import argparse
from functools import partial
from importlib import import_module
import json
from pathlib import Path

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra.qwen35.parallel import _reduce
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix.matched_spawn import measure, requests_for, seed_roots


def reduce_as(dtype, module, inputs, output):
    value = output.to(dtype)
    dist.all_reduce(value, group=module._tp_group)
    return output.copy_(value)


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(settings['parallel_settings']).read_text()))
    reductions = [(module, identity) for module in backend.model.modules()
                  for identity, hook in module._forward_hooks.items() if hook is _reduce]
    assert reductions
    prepared = json.loads(Path(settings['prepared']).read_text())
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'workloads': [],
              'scope': 'Exact recorded finite requests, synchronized warmed replay; shared TP4 model with configured collective accumulation precision.'}
    for workload in prepared['workloads']:
        inference = json.loads(Path(workload['protocol']).read_text())['inference']['settings']
        original_batches = requests_for(workload, backend, shared, settings)
        modes = {}
        report['workloads'].append({'track': workload['track'], 'modes': modes})
        for name, definition in settings['implementations'].items():
            for module, identity in reductions:
                module._forward_hooks[identity] = partial(reduce_as, getattr(torch, definition['reduction_dtype']))
            cls = getattr(import_module(definition['module']), definition['class'])
            tail = cls(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                       inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            caches = [roots, *[getattr(tail, key) for key in definition['caches']]]
            batches = [batch[::definition['row_stride']] for batch in original_batches]
            warmup, _ = measure(backend, tail, roots, batches, settings)
            repetitions = []
            for repetition in range(settings['repetitions']):
                for cache in caches:
                    cache.clear()
                root_work = seed_roots(backend, roots, batches)
                result, logits = measure(backend, tail, roots, batches, settings)
                assert all(batch['work']['graph_captures'] == 0 for batch in result['batches'])
                result.update(repetition=repetition, root_initialization=root_work,
                    logits=[value.tolist() for value in logits])
                repetitions.append(result)
            modes[name] = {'warmup': warmup, 'repetitions': repetitions,
                          'task_ids': [[request.task_id for request in batch] for batch in batches],
                          'option_counts': [[len(request.field['options']) for request in batch] for batch in batches]}
            for cache in caches:
                cache.clear()
            tail.graphs.clear()
            (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(
                json.dumps(report, indent=2) + '\n')
            print(json.dumps({'rank': dist.get_rank(), 'track': workload['track'],
                              'mode': name, 'completed': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
