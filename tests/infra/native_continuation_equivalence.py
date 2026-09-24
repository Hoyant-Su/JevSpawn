import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from environments.maze import MazeEnvironment
from jev_spawn.algo.structured import padded
from jev_spawn.algo.token_paths import compile_token_paths
from jev_spawn.infra.cached_suffix import ragged_suffix
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.native_cache_batch import export_native_rows
from jev_spawn.runtime.token_stream import CapturedTokenStream


def edges(table, state):
    return {table['token_ids'][token]: target for token, target, valid in zip(
        table['edge_token_indices'][state], table['edge_next_states'][state],
        table['edge_valid'][state], strict=True) if valid}


@torch.inference_mode()
def main(config):
    prompts = load_prompt(config['prompts'])
    feedback_prompts = load_prompt(config['feedback_prompts'])
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(config['parallel_settings']).read_text()))
    tasks = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    settings = json.loads(Path(config['environment']).read_text())
    source = json.loads(Path(config['source_result']).read_text())
    episode, = [record for record in source['records'] if record['variant'] == config['source_variant']]
    assert len(episode['turns']) == shared.runtime.max_turns
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    environments = [MazeEnvironment(task, settings, output, deadline=lambda: None,
        configuration=config['maze_configuration']) for task in tasks]
    actions = list(environments[0].native_actions)
    assert all(list(environment.native_actions) == actions for environment in environments)
    terminator = getattr(backend.tokenizer, config['terminator'])
    paths = [backend.tokenizer(action, add_special_tokens=False)['input_ids'] + [terminator] for action in actions]
    table = compile_token_paths(paths)
    decoder = CapturedTokenStream(backend, len(tasks), shared.model.max_input_tokens + shared.generation.max_new_tokens,
                                  table, None, None, config['stream'])
    prefix = decoder.forced_prefix.tolist()
    branch_tokens = list(edges(table, decoder.prefix_state))
    branch_indices = [table['token_ids'].index(token) for token in branch_tokens]
    rendered = [backend.tokenizer.apply_chat_template([
        {'role': 'system', 'content': prompts['system']},
        {'role': 'user', 'content': prompts['user'].format(
            context=environment.context(False, config['serialization']),
            actions=json.dumps(actions, **config['serialization']))}], tokenize=False,
        add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking) for environment in environments]
    histories = [backend.tokenizer(text, add_special_tokens=False)['input_ids'] for text in rendered]
    assert list(map(len, histories)) == episode['turns'][0]['work']['initial_input_tokens']
    histories = [history + prefix for history in histories]
    previous, feedback, records = [], [], []
    for turn in episode['turns']:
        observations = turn['observations']
        assert [row['task_id'] for row in observations] == [task['task_id'] for task in tasks]
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        if previous:
            states, metadata = zip(*previous, strict=True)
            tails = [[record['pending_token'], *tokens, *prefix]
                     for record, tokens in zip(metadata, feedback, strict=True)]
            work = {'computed_input_tokens': 0, 'padded_input_tokens': 0}
            extended = ragged_suffix(backend, list(states), [tail[:-1] for tail in tails], work)
            decoder.load(extended, [tail[-1] for tail in tails])
            decoder.start_state = decoder.prefix_state
            decoder.reset_path()
            decoder.graph.replay()
        else:
            ids, mask = padded(histories, backend.tokenizer.pad_token_id, backend.device, 'left')
            decoder.start_state = decoder.prefix_state
            decoder.prefill({'input_ids': ids, 'attention_mask': mask})
        torch.cuda.synchronize(backend.device)
        cached_seconds = time.perf_counter() - started
        cached = decoder.logits[:, branch_indices].clone()
        started = time.perf_counter()
        ids, mask = padded(histories, backend.tokenizer.pad_token_id, backend.device, 'left')
        fresh = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
        logits = F.linear(fresh.last_hidden_state[:, -1].float(), decoder.selected_weights)[:, branch_indices]
        torch.cuda.synchronize(backend.device)
        fresh_seconds = time.perf_counter() - started
        diagnostic = None
        if turn['turn'] in config['diagnostic_turns']:
            reference = json.loads(Path(config['source_comparison_result']).read_text())
            original, = [record for record in reference['records'] if record['turn'] == turn['turn']]
            diagnostic = {'input_token_ids': ids.tolist(), 'attention_mask': mask.tolist(),
                'input_token_counts': mask.sum(-1).tolist(),
                'original_run_fresh_logits': [row['fresh_logits'] for row in original['rows']],
                'fresh_repeats': []}
            for repeat in range(config['diagnostic_repeats']):
                repeat_output = backend.model.model(input_ids=ids, attention_mask=mask,
                    position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
                repeated = F.linear(repeat_output.last_hidden_state[:, -1].float(),
                                    decoder.selected_weights)[:, branch_indices]
                diagnostic['fresh_repeats'].append({'repeat': repeat, 'logits': repeated.tolist(),
                    'maxdiff_current_fresh': (repeated - logits).abs().amax(-1).tolist(),
                    'maxdiff_cached': (repeated - cached).abs().amax(-1).tolist()})
                del repeat_output, repeated
        rows = []
        for task, left, right, observed in zip(tasks, cached.tolist(), logits.tolist(), observations, strict=True):
            a = branch_tokens[max(range(len(left)), key=left.__getitem__)]
            b = branch_tokens[max(range(len(right)), key=right.__getitem__)]
            rows.append({'task_id': task['task_id'], 'branch_token_ids': branch_tokens,
                'cached_logits': left, 'fresh_logits': right, 'cached_choice': a, 'fresh_choice': b,
                'choice_agreement': a == b, 'recorded_action': observed['action'],
                'max_absolute_logit_difference': max(abs(x-y) for x,y in zip(left, right, strict=True))})
        del fresh, logits, cached, ids, mask
        forced = [row['token_ids'] for row in observations]
        assert all(path in paths and path[:len(prefix)] == prefix for path in forced)
        assert len(set(map(len, forced))) == 1
        states = [decoder.prefix_state] * len(tasks)
        for offset in range(len(prefix), len(forced[0])):
            tokens = [path[offset] for path in forced]
            states = [edges(table, state)[token] for state, token in zip(states, tokens, strict=True)]
            decoder.ids[:, 0].copy_(torch.tensor(tokens, device=backend.device))
            decoder.transitions.state.copy_(torch.tensor(states, device=backend.device))
            decoder.transitions.done.copy_(decoder.transitions.terminal[decoder.transitions.state])
            if offset != len(forced[0]) - 1:
                if decoder.graph is None:
                    decoder.capture(shared.runtime.graph_warmup_steps)
                decoder.graph.replay()
        caches, metadata = export_native_rows(decoder, list(range(len(tasks))))
        assert metadata == [row['cache_metadata'] for row in observations]
        previous = list(zip(caches, metadata, strict=True))
        feedback = []
        for environment, observed in zip(environments, observations, strict=True):
            text = feedback_prompts['separator'] + backend.tokenizer.apply_chat_template([
                {'role': 'user', 'content': feedback_prompts['user'].format(
                    tool=environment.configuration['tool_name'],
                    feedback=json.dumps(observed['feedback'], **config['serialization']))}], tokenize=False,
                add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
            feedback.append(backend.tokenizer(text, add_special_tokens=False)['input_ids'])
        histories = [history + path[len(prefix):] + tokens + prefix
                     for history, path, tokens in zip(histories, forced, feedback, strict=True)]
        records.append({'turn': turn['turn'], 'cached_seconds': cached_seconds,
                        'fresh_seconds': fresh_seconds, 'rows': rows, 'diagnostic': diagnostic})
        (output / f'rank-{dist.get_rank()}.json').write_text(json.dumps({
            'scope': config['scope'], 'startup': startup, 'records': records}, **config['serialization']) + '\n')
        if commands.is_leader:
            print(json.dumps(records[-1]), flush=True)
    decoder.graph = None
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
