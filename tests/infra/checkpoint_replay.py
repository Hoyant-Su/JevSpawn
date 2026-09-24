import argparse
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import ParallelPrefixCache
from baselines.common.task_prefix_service import TaskPrefixService
from jev_spawn.infra.readout_labels import AdmittedPrompt
from jev_spawn.infra.finite_batch import score_finite_with_tail
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache


class CheckpointPrefixCache(PrefixCache):
    """Keep exact intermediate hybrid states at fixed token boundaries."""

    def __init__(self, capacity, block_tokens):
        super().__init__(capacity)
        self.block_tokens = block_tokens

    def get_many(self, sequences, compute):
        for stop in range(self.block_tokens, max(map(len, sequences)), self.block_tokens):
            checkpoints = [sequence[:stop] for sequence in sequences if len(sequence) > stop]
            super().get_many(checkpoints, compute)
        return super().get_many(sequences, compute)


class ObservedPrefix(ParallelPrefixCache):
    def __init__(self, capacity, commands):
        super().__init__(PrefixCache(capacity), commands)
        self.plans = []

    def resident_plan(self, keys):
        hits = super().resident_plan(keys)
        self.plans.append({'hits': hits, 'token_counts': [len(key[0]) for key in keys]})
        return hits


class ObservedCheckpoint(CheckpointPrefixCache, ParallelPrefixCache):
    def __init__(self, capacity, block_tokens, commands):
        PrefixCache.__init__(self, capacity)
        self.block_tokens, self.commands, self.plans = block_tokens, commands, []

    def resident_plan(self, keys):
        hits = ParallelPrefixCache.resident_plan(self, keys)
        self.plans.append({'hits': hits, 'token_counts': [len(key[0]) for key in keys]})
        return hits


def prepare(config, backend, shared):
    recorded = json.loads(Path(config['requests']).read_text())
    cohorts = []
    for cohort in recorded['cohorts']:
        requests, lengths = [], []
        for item in cohort['requests']:
            assert backend.tokenizer(item['rendered'], add_special_tokens=False)['input_ids'] == item['tokens']
            assert backend.tokenizer.apply_chat_template(item['messages'], tokenize=False,
                add_generation_prompt=True, enable_thinking=shared.generation.enable_thinking) == item['rendered']
            request = SimpleNamespace(task_id=item['task_id'], field=item['field'], input_ids=item['tokens'], root_tokens=item['root_tokens'],
                admitted=AdmittedPrompt(item['rendered'], tuple(item['tokens'])))
            assert TaskPrefixService.task_prefix_length(SimpleNamespace(backend=backend), [request]) == item['base_length']
            assert len(item['tokens']) <= shared.model.max_input_tokens
            requests.append(request)
            lengths.append(item['base_length'])
        cohorts.append((requests, lengths, cohort['physical_batch_size']))
    assert len(cohorts) == config['expected_cohorts']
    assert sum(len(requests) for requests, _, _ in cohorts) == config['expected_requests']
    return cohorts


def main(config):
    assert config['required_source_line'] in inspect.getsource(score_finite_with_tail), config['reproduction_instruction']
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    tails, banks = {}, {}
    states = {'reference': lambda: ObservedPrefix(config['state_capacity'], commands),
              'checkpoint': lambda: ObservedCheckpoint(config['state_capacity'], config['block_tokens'], commands)}
    for name in config['variants']:
        tails[name] = StableFiniteGraphTail(backend, shared.runtime,
            ObservedPrefix(shared.runtime.root_batch_size, commands), config['state_copy'], config['graph_shape'])
        banks[name] = {'base': ObservedPrefix(shared.runtime.root_batch_size, commands), 'state': states[name]()}

    @torch.inference_mode()
    def replay(payload):
        name = payload['variant']
        for tail in tails.values():
            tail.graphs.clear()
            tail.prefix_cache.clear()
        for bank in banks.values():
            for cache in bank.values():
                cache.clear()
        torch.cuda.synchronize(backend.device)
        torch.cuda.reset_peak_memory_stats(backend.device)
        initial = torch.cuda.memory_allocated(backend.device)
        tail, bank = tails[name], banks[name]
        records = []
        started = time.perf_counter()
        for index, (requests, lengths, physical_rows) in enumerate(payload['cohorts']):
            caches = {**bank, 'final': tail.prefix_cache}
            for cache in caches.values():
                cache.plans.clear()
            begin = time.perf_counter()
            result = tail.score(requests, lengths, bank['base'], bank['state'], physical_batch_size=physical_rows)
            torch.cuda.synchronize(backend.device)
            elapsed = time.perf_counter() - begin
            result.pop('device_probabilities')
            rankmax = torch.tensor(elapsed, device=backend.device, dtype=torch.float64)
            dist.all_reduce(rankmax, op=dist.ReduceOp.MAX)
            records.append({'cohort': index, 'task_ids': [r.task_id for r in requests],
                'node_ids': [r.field['id'] for r in requests], 'input_tokens': [len(r.admitted.tokens) for r in requests],
                'base_lengths': lengths, 'rankmax_seconds': rankmax.item(),
                'cache_plans': {key: list(cache.plans) for key, cache in caches.items()}, 'result': result})
            if commands.is_leader:
                print(json.dumps({'phase': payload['phase'], 'variant': name, 'cohort': index,
                    'requests': len(requests), 'seconds': rankmax.item()}), flush=True)
        elapsed = time.perf_counter() - started
        values = [record['result']['groups'] for record in records]
        rank_values = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(rank_values, values, group=commands.control_group)
        assert all(value == values for value in rank_values)
        report = {'phase': payload['phase'], 'variant': name, 'records': records,
            'rank_outputs_equal': True, 'elapsed_seconds': elapsed, 'initial_allocated_bytes': initial,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device)}
        (output / config['rank_file'].format(rank=dist.get_rank(), phase=payload['phase'], variant=name)).write_text(
            json.dumps(report, indent=2)+'\n')
        return report

    commands.register(config['command'], replay)
    if commands.is_leader:
        try:
            begin = time.perf_counter()
            cohorts = prepare(config, backend, shared)
            preparation = time.perf_counter() - begin
            reports = [commands.call(config['command'], {'phase': phase, 'variant': variant, 'cohorts': cohorts})
                       for phase in config['phases'] for variant in config['variants']]
            comparisons = []
            for phase in config['phases']:
                reference, candidate = [report for report in reports if report['phase'] == phase]
                pairs = [(a, b) for left, right in zip(reference['records'], candidate['records'], strict=True)
                         for a, b in zip(left['result']['groups'][0], right['result']['groups'][0], strict=True)]
                comparisons.append({'phase': phase, 'requests': len(pairs),
                    'choice_agreements': sum(a['choice'] == b['choice'] for a, b in pairs),
                    'maximum_absolute_logit_difference': max(abs(x-y) for a,b in pairs
                        for x,y in zip(a['option_logits'],b['option_logits'],strict=True)),
                    'maximum_absolute_probability_difference': max(abs(x-y) for a,b in pairs
                        for x,y in zip(a['probabilities'],b['probabilities'],strict=True)),
                    'reference_seconds': reference['elapsed_seconds'], 'checkpoint_seconds': candidate['elapsed_seconds'],
                    'speedup': reference['elapsed_seconds']/candidate['elapsed_seconds']})
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                'verified_preparation_seconds': preparation, 'reports': reports, 'comparisons': comparisons}, indent=2)+'\n')
            print(json.dumps({'comparisons': comparisons}), flush=True)
        finally:
            commands.finish()
    else:
        commands.serve()
    for tail in tails.values():
        tail.graphs.clear()
        tail.prefix_cache.clear()
    for bank in banks.values():
        for cache in bank.values():
            cache.clear()
    torch.cuda.synchronize(backend.device)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
