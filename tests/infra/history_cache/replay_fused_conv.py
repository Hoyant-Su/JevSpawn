import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.history_cache import finite_checkpoint
from tests.infra.history_cache.replay_checkpoint import ObservedCheckpoint, ObservedReference
from tests.infra.history_cache.replay_finite import reset_history
from tests.infra.native_chunk_checkpoint.fused_conv import FusedConvRecorder
from tests.infra.native_history_cache.replay import requests_from_record
from tests.infra.native_suffix_graph.qualify import compare, measure, tensors


class FusedCheckpoint(ObservedCheckpoint):
    def extend_with_checkpoint(self, *args, **kwargs):
        with patch.object(finite_checkpoint, 'CheckpointRecorder', FusedConvRecorder):
            return super().extend_with_checkpoint(*args, **kwargs)


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    service = json.loads(Path(settings['service']).read_text())['settings']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {'settings': settings, 'startup': startup, 'sequences': []}
    for sources in settings['sequences']:
        sequence = [requests_from_record(backend, source) for source in sources]
        roots = PrefixCache(shared.runtime.root_batch_size)
        constructors = (ObservedReference, ObservedCheckpoint, FusedCheckpoint)
        tails = {}
        for name, constructor in zip(settings['arms'], constructors, strict=True):
            arguments = (backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                         service['state_copy'], service['graph_shape'])
            tails[name] = constructor(*arguments) if constructor is ObservedReference else constructor(
                *arguments, settings['history_cache'])

        def reset():
            for tail in tails.values():
                tail.prefix_cache.clear()
                if isinstance(tail, ObservedCheckpoint):
                    reset_history(tail)

        reference = tails[settings['arms'][0]]
        for requests, lengths in sequence:
            reference.score(requests, lengths, roots)
        reset()
        diagnostics = []
        for source, (requests, lengths) in zip(sources, sequence, strict=True):
            values, logits, states = {}, {}, {}
            for name, tail in tails.items():
                result = tail.score(requests, lengths, roots)
                values[name] = {key: result[key] for key in settings['workload_fields']}
                values[name]['choices'] = [answer['choice'] for group in result['groups'] for answer in group]
                logits[name] = tail.last_logits.clone()
                states[name] = [tensor.clone() for cache in tail.prefix_cache.entries.values()
                                for tensor in tensors(cache)]
            checks = {}
            for left, right in settings['comparisons']:
                checks[f'{left}:{right}'] = {'logits': compare([logits[left]], [logits[right]]),
                    'states': compare(states[left], states[right]),
                    'choices_equal': values[left]['choices'] == values[right]['choices']}
            diagnostics.append({'source': source, 'task_ids': [request.task_id for request in requests],
                                'workload': values, 'checks': checks})
        timings = []
        for repetition in range(settings['repetitions']):
            reset()
            order = settings['arms'][repetition % len(tails):] + settings['arms'][:repetition % len(tails)]
            cohorts = []
            for requests, lengths in sequence:
                cohorts.append({name: measure(lambda: tails[name].score(requests, lengths, roots),
                    backend.device, settings['measurements_per_cohort']) for name in order})
            timings.append(cohorts)
        report['sequences'].append({'sources': sources, 'diagnostics': diagnostics, 'timings': timings})
        (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
        for tail in tails.values():
            tail.graphs.clear()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
