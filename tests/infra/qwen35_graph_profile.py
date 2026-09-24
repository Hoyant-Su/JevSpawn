import argparse
from collections import defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import re
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.configuration import load_resource
from jev_spawn.runtime.decoding import CapturedDecode


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def summarize_trace(path, settings):
    trace = json.loads(path.read_text())
    events = [event for event in trace['traceEvents'] if event.get('cat') in settings['device_categories']]
    assert events
    totals = defaultdict(lambda: {'calls': 0, 'device_microseconds': 0.0})
    categories = defaultdict(lambda: {'calls': 0, 'device_microseconds': 0.0})
    for event in events:
        name = event['name']
        category = next((group['name'] for group in settings['kernel_groups']
                         if any(re.search(pattern, name, re.IGNORECASE) for pattern in group['patterns'])),
                        settings['unmatched_category'])
        for row in [totals[name], categories[category]]:
            row['calls'] += 1
            row['device_microseconds'] += event['dur']
    intervals = sorted((event['ts'], event['ts'] + event['dur']) for event in events)
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(stop, merged[-1][1])
        else:
            merged.append([start, stop])
    span = merged[-1][1] - merged[0][0]
    busy = sum(stop - start for start, stop in merged)
    summed = sum(event['dur'] for event in events)
    return {'profiled_steps': settings['profile_steps'], 'device_event_count': len(events),
            'summed_device_ms_per_step': summed / settings['profile_steps'] / settings['microseconds_per_ms'],
            'instrumented_device_span_ms': span / settings['microseconds_per_ms'],
            'instrumented_union_busy_ms': busy / settings['microseconds_per_ms'],
            'instrumented_interkernel_gap_ms': (span - busy) / settings['microseconds_per_ms'],
            'instrumented_gap_fraction': (span - busy) / span,
            'kernels': [{'name': name, **row, 'fraction_of_summed_device_time': row['device_microseconds'] / summed}
                        for name, row in sorted(totals.items(), key=lambda item: item[1]['device_microseconds'], reverse=True)],
            'categories': [{'name': name, **row, 'fraction_of_summed_device_time': row['device_microseconds'] / summed}
                           for name, row in sorted(categories.items(), key=lambda item: item[1]['device_microseconds'], reverse=True)]}


def measure(decoder, forced, steps):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(steps + 1)]
    torch.cuda.synchronize()
    started = time.perf_counter()
    events[0].record()
    for step in range(steps):
        decoder.ids.copy_(forced[step])
        decoder.graph.replay()
        events[step + 1].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    intervals = [left.elapsed_time(right) for left, right in zip(events, events[1:])]
    return {'wall_seconds': wall, 'per_step_ms': intervals,
            'median_itl_ms': statistics.median(intervals), 'mean_itl_ms': statistics.mean(intervals),
            'per_row_tokens_per_second': steps / wall}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['workload_config']).read_text())
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    backend = Backend(shared.backend())
    record = json.loads(Path(source['single_source']).read_text())
    token_ids = record['input_ids'][source['single_row']]
    ids = torch.tensor([token_ids], device=backend.device)
    single = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    batch_record = json.loads(Path(source['batch_source']).read_text())[source['batch_index']]
    rendered = backend.tokenizer.apply_chat_template(batch_record['messages'], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    batch = backend.tokenizer(rendered, padding=True, add_special_tokens=False,
                              return_tensors='pt', truncation=False).to(backend.device)
    assert batch['attention_mask'].sum(-1).tolist() == batch_record['input_tokens']
    workloads = [(single, [record['fields'][source['single_row']]['id']]), (batch, batch_record['task_ids'])]
    execution_settings = load_resource('qwen35_execution')
    language = backend.model.get_submodule(execution_settings['language_model_path'])
    linear_modules = [(name, module) for name, module in language.named_modules()
                      if isinstance(module, torch.nn.Linear)] + [('lm_head', backend.model.lm_head)]
    linear = [{'name': name, 'weight_shape': list(module.weight.shape),
               'weight_bytes': module.weight.numel() * module.weight.element_size()}
              for name, module in linear_modules]
    gpu = torch.cuda.get_device_properties(backend.device)
    hardware = {'gpu_name': gpu.name, 'total_memory_bytes': gpu.total_memory,
                'multiprocessor_count': gpu.multi_processor_count,
                'loaded_parameter_bytes': sum(parameter.numel() * parameter.element_size()
                                              for parameter in backend.model.parameters()),
                'active_linear_weight_bytes': sum(row['weight_bytes'] for row in linear),
                'active_linear_weight_scope': 'Language transformer Linear modules and vocabulary head; excludes vision and embedding lookup.'}
    save(output / 'protocol.json', {'settings': settings, 'shared': asdict(shared),
         'execution_settings': execution_settings, 'hardware': hardware,
         'linear_weights': linear,
         'scope': 'Unprofiled timing and instrumented attribution are separate; no algorithm, precision, or workload changes.'})
    reports = []
    for inputs, identities in workloads:
        batch_size = len(identities)
        reference = json.loads((Path(settings['reference_run']) / f'native_b{batch_size}.json').read_text())
        forced = torch.tensor([reference['initial_token_ids'], *zip(*reference['output_token_ids'])],
                              device=backend.device).unsqueeze(-1)
        assert reference['task_ids'] == identities
        capacity = inputs['input_ids'].shape[-1] + shared.generation.max_new_tokens
        decoder = CapturedDecode(backend, batch_size, capacity)
        decoder.prefill(inputs)
        decoder.capture(shared.runtime.graph_warmup_steps)
        snapshot = decoder.capture_snapshot()
        unprofiled = measure(decoder, forced, settings['timed_steps'])
        decoder.restore_capture_snapshot(snapshot)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False, profile_memory=False, with_stack=False) as profiler:
            for step in range(settings['profile_steps']):
                decoder.ids.copy_(forced[step])
                decoder.graph.replay()
            torch.cuda.synchronize()
        trace_path = output / f'trace-b{batch_size}.json'
        profiler.export_chrome_trace(str(trace_path))
        summary = {'batch_size': batch_size, 'task_ids': identities,
                   'input_tokens': inputs['attention_mask'].sum(-1).tolist(),
                   'padded_width': inputs['input_ids'].shape[-1], 'cache_capacity': capacity,
                   'unprofiled': unprofiled, 'profile': summarize_trace(trace_path, settings)}
        reports.append(summary)
        save(output / f'b{batch_size}.json', summary)
        print(json.dumps({key: value for key, value in summary.items() if key != 'profile'}), flush=True)
        del decoder, snapshot, profiler
        gc.collect()
        torch.cuda.empty_cache()
    summary_path = Path(settings['summary_output'])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    save(summary_path, {'settings': settings, 'hardware': hardware, 'reports': reports})


if __name__ == '__main__':
    main()
