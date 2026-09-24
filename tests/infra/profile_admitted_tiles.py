import argparse
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from functools import partial
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from baselines.common.parallel_service import decode_request, encode_request
from baselines.common.runtime import InferenceRuntime
from baselines.common.shared_state_prefix_service import SharedStatePrefixService
from jev_spawn.algo import structured
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.schema import CONTROLLER, controller_prompts
from qwen35_graph_profile import summarize_trace


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


@contextmanager
def instrument(service, scopes):
    clone = structured.deepcopy
    state = next(iter(service.state_cache.entries.values()))
    cache_type = type(state)
    reorder = cache_type.reorder_cache
    active = []

    def copied(*args, **kwargs):
        with record_function(scopes['clone']):
            return clone(*args, **kwargs)

    def expanded(cache, *args, **kwargs):
        with record_function(scopes['expand']):
            return reorder(cache, *args, **kwargs)

    def enter(module, args, kwargs):
        scope = record_function(scopes['model'])
        active.append(scope)
        scope.__enter__()

    def leave(module, args, result):
        active.pop().__exit__(None, None, None)

    structured.deepcopy, cache_type.reorder_cache = copied, expanded
    handles = [service.backend.model.model.register_forward_pre_hook(enter, with_kwargs=True),
               service.backend.model.model.register_forward_hook(leave)]
    try:
        yield
    finally:
        structured.deepcopy, cache_type.reorder_cache = clone, reorder
        for handle in handles:
            handle.remove()


def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    parallel = json.loads(Path(settings['parallel_settings']).read_text())
    backend, commands, startup = initialize_parallel(shared, parallel)
    source = json.loads(Path(settings['source_protocol']).read_text())
    method = json.loads(Path(settings['method']).read_text())
    runtime = InferenceRuntime(settings['shared_config'], partial(SharedStatePrefixService,
        settings=method['settings'], prompts=load_prompt(method['prompts'])), backend=backend)
    CONTROLLER.clear()
    CONTROLLER.update(source['controller'])
    output = Path(settings['output'])
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=False)
        save(output / 'protocol.json', {'settings': settings, 'startup': startup, 'source_groups': source['groups']})
    dist.barrier()
    records = []

    @torch.inference_mode()
    def score(payload):
        batch = [decode_request(value) for value in payload['requests']]
        profiling = payload['phase'] == settings['profile_phase']
        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                           profile_memory=False, with_stack=False) if profiling else nullcontext()
        instrumentation = instrument(runtime.service, settings['scopes']) if profiling else nullcontext()
        torch.cuda.synchronize(backend.device)
        started = time.perf_counter()
        with profiler as observed, instrumentation:
            result = runtime.service._score(batch)
            torch.cuda.synchronize(backend.device)
        local_wall = time.perf_counter() - started
        metrics = torch.tensor([local_wall, *result['timings'].values()], device=backend.device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        values = metrics.tolist()
        reference = json.loads((Path(settings['reference_run']) /
            f"measured-admitted-{payload['batch_index']}.json").read_text())
        assert result['groups'] == reference['result']['groups']
        record = {'phase': payload['phase'], 'batch_index': payload['batch_index'],
            'rank': dist.get_rank(), 'local_scoring_seconds': local_wall, 'rankmax_scoring_seconds': values[0],
            'rankmax_phase_seconds': dict(zip(result['timings'], values[1:])), 'result': result,
            'exact_original_logits_choices': True}
        if profiling:
            path = output / f"trace-rank{dist.get_rank()}-batch{payload['batch_index']}.json"
            observed.export_chrome_trace(str(path))
            record['trace'] = summarize_trace(path, settings['trace'])
            record['operator_scopes'] = [{'name': event.key, 'count': event.count,
                'cpu_time_total_us': event.cpu_time_total, 'self_cpu_time_total_us': event.self_cpu_time_total,
                'device_time_total_us': event.device_time_total, 'self_device_time_total_us': event.self_device_time_total}
                for event in observed.key_averages() if event.key in settings['scopes'].values()]
            save(output / f"profile-rank{dist.get_rank()}-batch{payload['batch_index']}.json", record)
        records.append(record)
        return record

    commands.register(settings['command'], score)
    if commands.is_leader:
        identity = json.loads(Path(settings['source_settings']).read_text())['task_id']
        runtime.deadlines.start(identity)
        try:
            for phase in settings['phases']:
                for index, group in enumerate(source['groups'], start=1):
                    started = time.perf_counter()
                    batch = []
                    for node in group:
                        prompt = controller_prompts([node['state']], node['question'], node['options'],
                            list(backend.answer_labels[:len(node['options'])]), CONTROLLER['output_instruction'])[0]
                        batch.append(runtime.service.request_type(
                            [{'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': prompt}],
                            settings['decision_tokens'], shared.generation.temperature, (), identity,
                            time.perf_counter(), Future(), field=node))
                    batch = runtime.service._validate_inputs(batch)
                    assert len(batch) == len(group)
                    admission = time.perf_counter() - started
                    record = commands.call(settings['command'], {'phase': phase, 'batch_index': index,
                        'requests': [encode_request(request) for request in batch]})
                    record.update(admission_seconds=admission,
                                  inclusive_seconds=time.perf_counter() - started)
                    save(output / f'{phase}-batch{index}.json', record)
            runtime.close()
        finally:
            commands.finish()
    else:
        commands.serve()
        runtime.close()
    if commands.is_leader:
        save(output / 'completion.json', {'settings': settings, 'records': records})
        print(json.dumps({'completed_cohorts': len(records), 'all_original_logits_choices_exact': True}), flush=True)
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
