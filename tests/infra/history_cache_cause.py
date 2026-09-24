import argparse
from importlib import import_module
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers.cache_utils import DynamicLayer

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix.matched_spawn import requests_for
from tests.infra.history_prefix import HistoryPrefixTail


def state_tensors(states):
    for state in states:
        for layer in state.layers:
            if isinstance(layer, DynamicLayer):
                yield layer.keys
                yield layer.values
            else:
                yield from layer.conv_states.values()
                yield from layer.recurrent_states.values()


class CheckedHistoryTail(HistoryPrefixTail):
    def extend_prefixes(self, backend, states, prefixes, bases, work, extend_states):
        sources = [*states, *[state for entries in self.history_cache.values() for _, state in entries]]
        tensors = list(state_tensors(sources))
        snapshots = [tensor.clone() for tensor in tensors]
        result = super().extend_prefixes(backend, states, prefixes, bases, work, extend_states)
        assert all(torch.equal(before, after) for before, after in zip(snapshots, tensors, strict=True))
        return result


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared,
        json.loads(Path(settings['parallel_settings']).read_text()))
    prepared = json.loads(Path(settings['prepared']).read_text())
    report = {'settings': settings, 'startup': startup, 'workloads': [],
              'scope': 'Numerical diagnosis with GPU state snapshots; timings are not speed measurements.'}
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    for workload in prepared['workloads']:
        inference = json.loads(Path(workload['protocol']).read_text())['inference']['settings']
        batches = requests_for(workload, backend, shared, settings)
        modes = {}
        for name, definition in settings['diagnostics'].items():
            cls = getattr(import_module(definition['module']), definition['class'])
            tail = cls(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                       inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            records = []
            for repetition in range(settings['repetitions']):
                for index, batch in enumerate(batches):
                    requests = batch[::definition['row_stride']]
                    for cache in definition['clear_before_call']:
                        getattr(tail, cache).clear()
                    result = tail.score(requests, [len(request.root_tokens) for request in requests], roots)
                    records.append({'repetition': repetition, 'batch_index': index,
                        'task_ids': [request.task_id for request in requests],
                        'fields': result['groups'][0],
                        'reused_history_tokens': result['reused_state_tokens'],
                        'computed_input_tokens': result['computed_input_tokens']})
            modes[name] = records
            roots.clear()
            for cache in definition['caches']:
                getattr(tail, cache).clear()
            tail.graphs.clear()
        report['workloads'].append({'source_track': workload['track'], 'modes': modes})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(
            json.dumps(report, indent=2) + '\n')
        print(json.dumps({'rank': dist.get_rank(), 'track': workload['track'],
                          'completed': True, 'source_states_unchanged': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
