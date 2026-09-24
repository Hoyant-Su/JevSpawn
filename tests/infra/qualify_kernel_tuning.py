import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.kernel_tuning import load_tuning, save_tuning, tuners
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from replay_recorded_finite import ObservedPrefixCache, prepare


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    settings = json.loads(Path(config['tuning_settings']).read_text())
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(config['cache_directory']) / config['cache_file'].format(rank=dist.get_rank())
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loaded = load_tuning(cache_path, backend, settings) if config['mode'] == 'restore' else None
    cache = ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
    banks = {key: ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands) for key in config['caches']}
    tail = StableFiniteGraphTail(backend, shared.runtime, cache, config['state_copy'], config['graph_shape'])

    @torch.inference_mode()
    def compute(payload):
        before = {name: len(tuner.cache) for name, tuner in tuners(settings).items()}
        torch.cuda.synchronize(backend.device)
        begin = time.perf_counter()
        results = [tail.score(requests, lengths, banks['base'], banks['state']) for requests, lengths in payload['cohorts']]
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - begin
        rank_seconds = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(rank_seconds, elapsed, group=commands.control_group)
        after = {name: len(tuner.cache) for name, tuner in tuners(settings).items()}
        report = {'mode': config['mode'], 'rank': dist.get_rank(), 'seconds': elapsed,
                  'rankmax_seconds': max(rank_seconds), 'results': results,
                  'loaded': loaded, 'before_entries': before, 'after_entries': after,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device)}
        if config['mode'] == 'save':
            report['saved'] = save_tuning(cache_path, backend, settings)
        choices = [[row['choice'] for row in result['groups'][0]] for result in results]
        ranks = [None for _ in range(shared.runtime.world_size)]
        dist.all_gather_object(ranks, choices, group=commands.control_group)
        assert all(row == choices for row in ranks)
        report['all_rank_choices_equal'] = True
        (output / config['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        return report

    commands.register(config['command'], compute)
    if commands.is_leader:
        try:
            cohorts, preparation = prepare(config, backend, shared)
            selected = [cohorts[config['variant']][index] for index in config['cohort_indices']]
            report = commands.call(config['command'], {'cohorts': selected})
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                'preparation_seconds': preparation, 'report': report}, indent=2) + '\n')
        finally:
            commands.finish()
    else:
        commands.serve()
    tail.graphs.clear()
    cache.clear()
    for bank in banks.values():
        bank.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    run(json.loads(Path(parser.parse_args().config).read_text()))
