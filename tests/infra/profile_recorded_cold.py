import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from mixed_instrumentation import instrument
from replay_recorded_finite import ObservedPrefixCache, prepare


def run(config):
    shared = SharedConfig.load(config['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(config['parallel_settings']).read_text()))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    cache = ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
    banks = {key: ObservedPrefixCache(PrefixCache(shared.runtime.root_batch_size), commands)
             for key in config['caches']}
    tail = StableFiniteGraphTail(backend, shared.runtime, cache, config['state_copy'], config['graph_shape'])

    @torch.inference_mode()
    def compute(payload):
        phase = payload['phase']
        if phase['clear_prefix']:
            cache.clear()
            for bank in banks.values():
                bank.clear()
        profiling = phase['profile']
        observed = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                           profile_memory=False, with_stack=False) if profiling else nullcontext()
        scopes = instrument(backend, config['scopes']) if profiling else nullcontext()
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        with observed, scopes:
            results = [tail.score(requests, lengths, banks['base'], banks['state'])
                       for requests, lengths in payload['cohorts']]
            torch.cuda.synchronize(backend.device)
        report = {'phase': phase, 'rank': dist.get_rank(), 'elapsed_seconds': time.perf_counter() - started,
                  'results': results, 'timing_scope': config['timing_scope']}
        if profiling:
            trace = output / config['trace_file'].format(rank=dist.get_rank(), phase=phase['name'])
            observed.export_chrome_trace(str(trace))
            report['trace'] = str(trace)
            report['operators'] = [{'name': event.key, 'calls': event.count,
                'self_cpu_us': event.self_cpu_time_total, 'cpu_us': event.cpu_time_total,
                'device_us': event.device_time_total, 'self_device_us': event.self_device_time_total}
                for event in observed.key_averages()]
        (output / config['rank_file'].format(rank=dist.get_rank(), phase=phase['name'])).write_text(
            json.dumps(report, indent=2) + '\n')
        return report

    commands.register(config['command'], compute)
    if commands.is_leader:
        try:
            cohorts, preparation = prepare(config, backend, shared)
            selected = [cohorts[config['variant']][index] for index in config['cohort_indices']]
            reports = [commands.call(config['command'], {'phase': phase, 'cohorts': selected})
                       for phase in config['phases']]
            Path(config['result']).write_text(json.dumps({'config': config, 'startup': startup,
                'preparation_seconds': preparation, 'reports': reports}, indent=2) + '\n')
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
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
