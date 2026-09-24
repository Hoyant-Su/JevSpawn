import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from environments.maze import MazeEnvironment
from jev_spawn.algo.token_paths import compile_token_paths
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.token_stream import CapturedTokenStream
from native_action_stream import FullHeadTokenStream


@torch.inference_mode()
def main(config):
    prompts = load_prompt(config['prompts'])
    feedback_prompt = load_prompt(config['feedback_prompts'])
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(config['parallel_settings']).read_text()))
    tasks = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    settings = json.loads(Path(config['environment']).read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    constructors = {'full_head': FullHeadTokenStream, 'candidate_head': CapturedTokenStream}
    records = []
    for variant in config['variants']:
        environments = [MazeEnvironment(task, settings, output, deadline=lambda: None,
            configuration=config['maze_configuration']) for task in tasks]
        actions = list(environments[0].native_actions)
        assert all(list(environment.native_actions) == actions for environment in environments)
        terminator = getattr(backend.tokenizer, config['terminator'])
        paths = [backend.tokenizer(action, add_special_tokens=False)['input_ids'] + [terminator] for action in actions]
        table = compile_token_paths(paths)
        budget = max(map(len, paths))
        assert budget <= shared.generation.max_new_tokens
        options = {'do_sample': False, 'max_new_tokens': budget, 'pad_token_id': backend.tokenizer.pad_token_id}
        rendered = [backend.tokenizer.apply_chat_template([
            {'role': 'system', 'content': prompts['system']},
            {'role': 'user', 'content': prompts['user'].format(
                context=environment.context(False, config['serialization']),
                actions=json.dumps(actions, **config['serialization']))}], tokenize=False,
            add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking) for environment in environments]
        for text in rendered:
            prefix = backend.tokenizer(text, add_special_tokens=False)['input_ids']
            for action, path in zip(actions, paths, strict=True):
                assert backend.tokenizer(text + action, add_special_tokens=False)['input_ids'] == prefix + path[:-1]
        inputs = backend.tokenizer(rendered, padding=True, add_special_tokens=False, return_tensors='pt').to(backend.device)
        assert max(inputs['attention_mask'].sum(-1).tolist()) <= shared.model.max_input_tokens
        active = list(range(len(tasks)))
        decoders, previous, feedback, turns = {}, [], [], []
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        for turn in range(shared.runtime.max_turns):
            assert time.perf_counter() - started < shared.runtime.sample_timeout_seconds
            size = len(active)
            if size not in decoders:
                decoders[size] = constructors[variant](backend, size,
                    shared.model.max_input_tokens + shared.generation.max_new_tokens,
                    table, None, None, config['stream'])
            decoder = decoders[size]
            torch.cuda.synchronize(backend.device)
            infer_started = time.perf_counter()
            if not previous:
                tokens, events, capture = decoder.generate_tokens(inputs, options,
                    lambda tokens, scores: torch.zeros(size, dtype=torch.bool, device=backend.device),
                    shared.runtime.graph_warmup_steps)
                tokens = tokens[:, inputs['input_ids'].shape[1]:]
                work = {'initial_input_tokens': inputs['attention_mask'].sum(-1).tolist()}
            else:
                assert all(record['cache_tokens'] + len(suffix) + 1 <= shared.model.max_input_tokens
                           for (_, record), suffix in zip(previous, feedback, strict=True))
                tokens, events, capture = decoder.continue_tokens(previous, feedback, options,
                    shared.runtime.graph_warmup_steps)
                work = decoder.continuation_work
            torch.cuda.synchronize(backend.device)
            inference_seconds = time.perf_counter() - infer_started
            next_active, next_previous, next_feedback, observations = [], [], [], []
            for row, (identity, values) in enumerate(zip(active, tokens.tolist(), strict=True)):
                emitted = values[:values.index(terminator) + 1]
                assert emitted in paths
                action = actions[paths.index(emitted)]
                environment = environments[identity]
                tool = environment.configuration['tool_name']
                observed = environment.observe(tool, {'action': action})
                observations.append({'task_id': tasks[identity]['task_id'], 'action': action,
                    'token_ids': emitted, 'feedback': observed, 'cache_metadata': decoder.completed[row][1]})
                if not environment.done:
                    text = feedback_prompt['separator'] + backend.tokenizer.apply_chat_template([
                        {'role': 'user', 'content': feedback_prompt['user'].format(tool=tool,
                            feedback=json.dumps(observed, **config['serialization']))}], tokenize=False,
                        add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking)
                    next_feedback.append(backend.tokenizer(text, add_special_tokens=False)['input_ids'])
                    next_previous.append(decoder.completed[row])
                    next_active.append(identity)
            turns.append({'turn': turn, 'batch_size': size, 'inference_seconds': inference_seconds,
                'capture_seconds': capture, 'work': work, 'observations': observations,
                'inter_token_ms': [left.elapsed_time(right) for left, right in zip(events, events[1:])]})
            active, previous, feedback = next_active, next_previous, next_feedback
            if commands.is_leader:
                print(json.dumps({'variant': variant, **turns[-1]}), flush=True)
            if not active:
                break
        torch.cuda.synchronize(backend.device)
        records.append({'variant': variant, 'elapsed_seconds': time.perf_counter() - started,
            'turns': turns, 'tasks': [{'task_id': task['task_id'], 'steps': len(environment.actions),
                'success': environment.answer is not None and environment.answer['success'],
                'environment_done': environment.done, 'turn_budget_exhausted': not environment.done,
                'actions': environment.actions} for task, environment in zip(tasks, environments, strict=True)]})
        (output / f'rank-{dist.get_rank()}.json').write_text(json.dumps({
            'scope': config['scope'], 'startup': startup, 'records': records}, **config['serialization']) + '\n')
        for decoder in decoders.values():
            decoder.graph = None
        del decoder, decoders, previous
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
