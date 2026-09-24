import argparse
from importlib import import_module
import json
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix.matched_spawn import compare, measure, requests_for, seed_roots


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    prepared = json.loads(Path(settings['prepared']).read_text())
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'scope': prepared['measurement_scope'], 'workloads': []}
    reference, candidate = settings['modes']
    for workload in prepared['workloads']:
        inference = json.loads(Path(workload['protocol']).read_text())['inference']['settings']
        batches = requests_for(workload, backend, shared, settings)
        modes, logits = {}, {}
        for mode in settings['modes']:
            definition = settings['implementations'][mode]
            cls = getattr(import_module(definition['module']), definition['class'])
            tail = cls(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                       inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            caches = [roots, *[getattr(tail, name) for name in definition['caches']]]
            warmup, _ = measure(backend, tail, roots, batches, settings)
            print(json.dumps({'rank': dist.get_rank(), 'track': workload['track'],
                              'mode': mode, 'warmup_complete': True}), flush=True)
            repetitions, logits[mode] = [], []
            for repetition in range(settings['repetitions']):
                for cache in caches:
                    cache.clear()
                root_work = seed_roots(backend, roots, batches)
                measurement, values = measure(backend, tail, roots, batches, settings)
                assert all(batch['work']['graph_captures'] == 0 for batch in measurement['batches'])
                measurement.update(repetition=repetition, root_initialization=root_work)
                repetitions.append(measurement)
                logits[mode].append(values)
            modes[mode] = {'warmup': warmup, 'repetitions': repetitions}
            for cache in caches:
                cache.clear()
            tail.graphs.clear()
        comparisons = [compare(a, b, batches) for a, b in zip(logits[candidate], logits[reference], strict=True)]
        exact = [[torch.equal(a, b) for a, b in zip(left, right, strict=True)]
                 for left, right in zip(logits[candidate], logits[reference], strict=True)]
        report['workloads'].append({'track': workload['track'], 'source': workload['source'],
            'batch_shapes': [len(batch) for batch in batches], 'modes': modes,
            'numerical_comparisons': comparisons, 'bitwise_equal_batches': exact})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'rank': dist.get_rank(), 'track': workload['track'], 'complete': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
