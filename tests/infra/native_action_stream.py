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


class FullHeadTokenStream(CapturedTokenStream):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_indices = torch.tensor(self.native_ids, device=self.backend.device, dtype=torch.long)

    def generate_tokens(self, inputs, options, stopping, warmup_steps, on_tokens=None, between_steps=None):
        return self.decode_path(inputs, options, stopping, warmup_steps, on_tokens, between_steps)

    def read_full_head(self, hidden):
        self.logits.copy_(self.backend.model.lm_head(hidden).index_select(-1, self.native_indices))

    @torch.inference_mode()
    def prefill(self, inputs):
        self.cache.reset()
        self.reset_path()
        width = inputs['input_ids'].shape[1]
        self.key_valid.fill_(True)
        self.key_valid[:, :width].copy_(inputs['attention_mask'])
        positions = (inputs['attention_mask'].cumsum(-1) - 1).clamp_min(0)
        output = self.trunk(**inputs, position_ids=positions, past_key_values=self.cache, use_cache=True)
        self.read_full_head(output.last_hidden_state[:, -1])
        self.transitions.advance(self.logits)
        self.positions.copy_(inputs['attention_mask'].sum(-1)[:, None])
        self.validate_attention()
        return self.ids[:, 0].clone()

    @torch.inference_mode()
    def step(self):
        valid = self.key_valid & (self.key_positions <= self.cache.get_seq_length())
        output = self.trunk(input_ids=self.ids, position_ids=self.positions,
            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
            past_key_values=self.cache, use_cache=True, **self.attention_arguments())
        self.read_full_head(output.last_hidden_state[:, -1])
        self.transitions.advance(self.logits)
        self.positions.add_(1)


@torch.inference_mode()
def main(config):
    shared = SharedConfig.load(config['shared_config'])
    prompts = load_prompt(config['prompts'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(config['parallel_settings']).read_text()))
    tasks = [json.loads(line) for line in Path(config['tasks']).read_text().splitlines()]
    environment_settings = json.loads(Path(config['environment']).read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    environments = [MazeEnvironment(task, environment_settings, output, deadline=lambda: None,
                                   configuration=config['maze_configuration']) for task in tasks]
    actions = list(environments[0].native_actions)
    assert all(list(environment.native_actions) == actions for environment in environments)
    terminator = getattr(backend.tokenizer, config['terminator'])
    paths = [backend.tokenizer(action, add_special_tokens=False)['input_ids'] + [terminator] for action in actions]
    table = compile_token_paths(paths)
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
    width = inputs['input_ids'].shape[1]
    budget = max(map(len, paths))
    assert budget <= shared.generation.max_new_tokens
    options = {'do_sample': False, 'max_new_tokens': budget, 'pad_token_id': backend.tokenizer.pad_token_id}
    constructors = {'full_head': FullHeadTokenStream, 'candidate_head': CapturedTokenStream}
    streams = {name: constructors[name](backend, len(tasks), width + budget, table, None, None,
                                      config['stream']) for name in config['variants']}
    records = []
    for phase in config['phases']:
        for name in config['variants']:
            decoder = streams[name]
            torch.cuda.synchronize(backend.device)
            started = time.perf_counter()
            tokens, events, capture_seconds = decoder.generate_tokens(inputs, options,
                lambda tokens, scores: torch.zeros(len(tasks), dtype=torch.bool, device=backend.device),
                shared.runtime.graph_warmup_steps)
            torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter() - started
            rows = []
            for task, row in zip(tasks, tokens[:, width:].tolist(), strict=True):
                emitted = row[:row.index(terminator) + 1]
                assert emitted in paths
                rows.append({'task_id': task['task_id'], 'token_ids': emitted,
                             'action': actions[paths.index(emitted)]})
            records.append({'phase': phase, 'variant': name, 'seconds': elapsed,
                'capture_seconds': capture_seconds, 'rows': rows,
                'inter_token_ms': [left.elapsed_time(right) for left, right in zip(events, events[1:])]})
            report = {'scope': config['scope'], 'startup': startup, 'batch_size': len(tasks),
                'input_tokens': inputs['attention_mask'].sum(-1).tolist(), 'native_paths': paths,
                'records': records}
            (output / f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, **config['serialization']) + '\n')
            if commands.is_leader:
                print(json.dumps(records[-1]), flush=True)
    comparisons = []
    for phase in config['phases']:
        reference, candidate = [record for record in records if record['phase'] == phase]
        comparisons.append({'phase': phase, 'actions_agree': reference['rows'] == candidate['rows'],
            'reference_seconds': reference['seconds'], 'candidate_seconds': candidate['seconds']})
    report['comparisons'] = comparisons
    report['scope_limit'] = 'Emitted action tokens only; no environment step or terminal-cache continuation is evaluated.'
    (output / f'rank-{dist.get_rank()}.json').write_text(json.dumps(report, **config['serialization']) + '\n')
    for decoder in streams.values():
        decoder.graph = None
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
