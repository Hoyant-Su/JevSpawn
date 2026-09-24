import argparse
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from importlib import import_module
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.infra import cached_suffix
from jev_spawn.infra.finite_graph import FiniteGraphTail
from jev_spawn.runtime.finite_decoding import CapturedFinite
from jev_spawn.runtime.prefix_cache import PrefixCache
from tests.infra.action_prefix.matched_spawn import measure, requests_for, seed_roots
from tests.infra.action_prefix import post_readout
from tests.infra.action_prefix.profile_events import kernel_report


class PhaseRecorder:
    def __init__(self, names):
        self.names, self.events = names, []

    def begin(self, name):
        scope = record_function(self.names[name])
        scope.__enter__()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return scope, start, end

    def finish(self, name, record):
        scope, start, end = record
        end.record()
        scope.__exit__(None, None, None)
        self.events.append((self.names[name], start, end))

    def wrap(self, name, function):
        @wraps(function)
        def measured(*args, **kwargs):
            record = self.begin(name)
            result = function(*args, **kwargs)
            self.finish(name, record)
            return result
        return measured

    def install(self, stack, backend, tail):
        stack.enter_context(patch.object(post_readout, 'snapshot_prefixes',
            self.wrap('snapshot', post_readout.snapshot_prefixes)))
        stack.enter_context(patch.object(cached_suffix, 'pack_native_caches',
            self.wrap('pack', cached_suffix.pack_native_caches)))
        stack.enter_context(patch.object(cached_suffix, 'split_native_cache_at',
            self.wrap('split', cached_suffix.split_native_cache_at)))
        stack.enter_context(patch.object(backend.model.model, 'forward',
            self.wrap('prefill', backend.model.model.forward)))
        stack.enter_context(patch.object(CapturedFinite, 'load', self.wrap('load', CapturedFinite.load)))
        stack.enter_context(patch.object(tail, 'extend_states', self.wrap('extension', tail.extend_states)))
        for decoder in tail.graphs.values():
            stack.enter_context(patch.object(decoder.graph, 'replay', self.wrap('graph', decoder.graph.replay)))
        original = FiniteGraphTail.__call__

        def finite_then_readout(*args, **kwargs):
            result = original(*args, **kwargs)
            self.readout = self.begin('readout')
            return result

        stack.enter_context(patch.object(FiniteGraphTail, '__call__', finite_then_readout))

    def report(self):
        phases = defaultdict(list)
        for name, start, end in self.events:
            phases[name].append(start.elapsed_time(end))
        return {name: {'calls': len(values), 'cuda_span_ms': values,
                       'total_cuda_span_ms': sum(values)} for name, values in phases.items()}


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    prepared = json.loads(Path(settings['prepared']).read_text())
    workloads = {row['track']: row for row in prepared['workloads']}
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    classes = {name: getattr(import_module(definition['module']), definition['class'])
               for name, definition in settings['implementations'].items()}
    report = {'settings': settings, 'startup': startup, 'profiles': [],
        'scope': prepared['measurement_scope'],
        'timing_note': 'CUDA event spans include launch gaps. Nested phases overlap and must not be summed. '
            'Kernel duration and device busy union come from CUPTI events. Profiled latency is not used as a speed claim.'}
    for track in settings['tracks']:
        workload = workloads[track]
        inference = json.loads(Path(workload['protocol']).read_text())['inference']['settings']
        batches = requests_for(workload, backend, shared, settings)
        for mode in settings['modes']:
            tail = classes[mode](backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                                 inference['state_copy'], inference['graph_shape'])
            roots = PrefixCache(shared.runtime.root_batch_size)
            cold, _ = measure(backend, tail, roots, batches, settings)
            tail.prefix_cache.clear()
            roots.clear()
            seed_roots(backend, roots, batches)
            unprofiled, _ = measure(backend, tail, roots, batches, settings)
            tail.prefix_cache.clear()
            roots.clear()
            seed_roots(backend, roots, batches)
            recorder = PhaseRecorder(settings['phase_names'])
            shapes, results = [], []

            def observe(module, args, kwargs):
                shapes.append(list(kwargs['input_ids'].shape))

            hook = backend.model.model.register_forward_pre_hook(observe, with_kwargs=True)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=settings['record_shapes'], with_stack=settings['with_stack']) as prof:
                with ExitStack() as stack:
                    recorder.install(stack, backend, tail)
                    torch.cuda.synchronize(backend.device)
                    started = time.perf_counter()
                    for requests in batches:
                        with record_function(settings['phase_names']['score']):
                            result = tail.score(requests, [len(request.root_tokens) for request in requests], roots)
                            recorder.finish('readout', recorder.readout)
                            results.append({key: result[key] for key in settings['workload_fields']})
                    torch.cuda.synchronize(backend.device)
                    wall = time.perf_counter() - started
            hook.remove()
            row = {'track': track, 'mode': mode, 'cold_warmup': cold, 'unprofiled': unprofiled,
                'profiled_wall_seconds': wall, 'phase_cuda_events': recorder.report(),
                'eager_forward_shapes': shapes, 'batches': results,
                'operators': [{'name': event.key, 'count': event.count, 'self_cpu_time_total_us': event.self_cpu_time_total,
                               'cpu_time_total_us': event.cpu_time_total, 'self_device_time_total_us': event.self_device_time_total,
                               'device_time_total_us': event.device_time_total} for event in prof.key_averages()]}
            if dist.get_rank() in settings['trace_ranks']:
                path = output / settings['trace_file'].format(track=track, mode=mode, rank=dist.get_rank())
                prof.export_chrome_trace(str(path))
                row['kernel_profile'] = kernel_report(path, settings)
                row['trace'] = str(path)
            report['profiles'].append(row)
            (output / settings['rank_file'].format(rank=dist.get_rank())).write_text(json.dumps(report, indent=2) + '\n')
            tail.graphs.clear()
            tail.prefix_cache.clear()
            roots.clear()
            print(json.dumps({'rank': dist.get_rank(), 'track': track, 'mode': mode, 'profiled': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
