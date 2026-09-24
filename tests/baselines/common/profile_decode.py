import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch
from transformers import GenerationConfig

from jev_spawn.infra.backend import Backend
from baselines.common.tasks import read, rows, render
from baselines.common.resources import ADAPTER_SETTINGS


def union_duration(intervals):
    total, end = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


@torch.inference_mode()
def measure(backend, inputs, mode, tokens, trace):
    events, host_times = [], []

    def record(module, args, output):
        if mode == 'synchronize':
            torch.cuda.synchronize()
            host_times.append(time.perf_counter())
        else:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.append(event)

    hook = backend.model.register_forward_hook(record) if mode != 'uninstrumented' else None
    options = GenerationConfig(do_sample=False, max_new_tokens=tokens, use_cache=True,
                               eos_token_id=backend.eos_ids, pad_token_id=backend.tokenizer.pad_token_id,
                               bos_token_id=backend.tokenizer.bos_token_id)
    torch.cuda.synchronize()
    started = time.perf_counter()
    context = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                     record_shapes=True) if trace else nullcontext()
    try:
        with context as profile:
            output = backend.model.generate(**inputs, generation_config=options, logits_to_keep=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
    finally:
        if hook is not None:
            hook.remove()
    output = output[:, inputs['input_ids'].shape[1]:].tolist()
    counts = [next((i + 1 for i, token in enumerate(sequence) if token in backend.eos_ids), len(sequence))
              for sequence in output]
    intervals = [1000 * (b - a) for a, b in zip(host_times, host_times[1:])] if mode == 'synchronize' else [
        a.elapsed_time(b) for a, b in zip(events, events[1:])]
    result = dict(mode=mode, profiled=bool(trace), elapsed_seconds=elapsed,
                  batch_size=len(output), output_tokens=counts,
                  aggregate_tokens_per_second=sum(counts) / elapsed,
                  output_token_ids=[sequence[:count] for sequence, count in zip(output, counts)],
                  itl_median_ms=statistics.median(intervals) if intervals else None,
                  itl_max_ms=max(intervals) if intervals else None)
    if trace:
        profile.export_chrome_trace(str(trace))
        trace_data = read(trace)['traceEvents']
        kernels = [(event['ts'], event['ts'] + event['dur']) for event in trace_data
                   if event.get('cat') == 'kernel' and event.get('ph') == 'X']
        assert kernels
        span = max(stop for _, stop in kernels) - min(start for start, _ in kernels)
        result['gpu_kernel_busy_fraction'] = union_duration(kernels) / span
        result['gpu_kernel_count'] = len(kernels)
        result['gpu_timeline_seconds'] = span / 1e6
        result['top_cpu_operators'] = profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=ADAPTER_SETTINGS['profile_decode']['operator_row_limit'])
        result['top_gpu_operators'] = profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=ADAPTER_SETTINGS['profile_decode']['operator_row_limit'])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(read(config['native_config']))
    tasks = rows(config['tasks'])[config['offset']:config['offset'] + config['batch_size']]
    assert len(tasks) == config['batch_size']
    messages = [[{'role': 'user', 'content': render(task)}] for task in tasks]
    texts = backend.tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
    inputs, lengths = backend._encode(texts)
    assert inputs['input_ids'].shape[0] == config['batch_size']
    measure(backend, inputs, 'uninstrumented', config['max_new_tokens'], None)
    measurements = []
    for repeat in range(config['repeats']):
        modes = config['modes'][repeat % len(config['modes']):] + config['modes'][:repeat % len(config['modes'])]
        for mode in modes:
            result = measure(backend, inputs, mode, config['max_new_tokens'], None)
            result['repeat'] = repeat
            measurements.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != 'output_token_ids'}), flush=True)
    assert all(row['output_token_ids'] == measurements[0]['output_token_ids'] for row in measurements)
    trace = measure(backend, inputs, 'events', config['profile_tokens'], args.output / 'trace.json')
    output = dict(config=config, backend=backend.metadata, task_ids=[row['task_id'] for row in tasks],
                  input_tokens=lengths, measurements=measurements, trace=trace, exact_token_parity=True)
    (args.output / 'result.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: value for key, value in trace.items() if key != 'output_token_ids'}), flush=True)


if __name__ == '__main__':
    main()
