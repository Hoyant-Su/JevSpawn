import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
from threading import Barrier
import time

import torch

from baselines.common.runtime import InferenceRuntime
from baselines.common.tasks import read, rows, render


def summarize(records, elapsed):
    intervals = [value for batch in records for row in batch['decode']
                 for value in row['inter_token_seconds']]
    executed = sum(row['executed_token_slots'] for row in records)
    return {'elapsed_seconds': elapsed,
            'actual_batch_sizes': [row['batch_size'] for row in records],
            'model_sequences': sum(row['batch_size'] for row in records),
            'active_token_slots': sum(row['active_token_slots'] for row in records),
            'executed_token_slots': executed,
            'finished_row_fraction': 1 - sum(row['active_token_slots'] for row in records) / executed,
            'median_itl_ms': 1000 * statistics.median(intervals) if intervals else None,
            'max_itl_ms': 1000 * max(intervals) if intervals else None,
            'mean_queue_seconds': statistics.mean(value for row in records for value in row['queue_seconds']),
            'mean_row_tail_wait_seconds': statistics.mean(value for row in records for value in row['row_tail_wait_seconds'])}


def measure(runtime, tasks, name, stops, budgets=None):
    barrier = Barrier(len(tasks))

    def complete(index):
        identity = name + '/' + tasks[index]['task_id']
        barrier.wait()
        runtime.deadlines.start(identity)
        return runtime.service.complete(
            [{'role': 'user', 'content': render(tasks[index])}],
            runtime.config.generation.max_new_tokens if budgets is None else budgets[index],
            runtime.config.generation.temperature,
            stop=stops[index], task_id=identity)[0]

    begin = len(runtime.service.records)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        texts = list(pool.map(complete, range(len(tasks))))
    elapsed = time.perf_counter() - started
    records = runtime.service.records[begin:]
    return {'phase': name, 'task_ids': [task['task_id'] for task in tasks], 'texts': texts,
            'batches': records, 'metrics': summarize(records, elapsed)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-config', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = read(args.profile)
    tasks = rows(profile['tasks'])[profile['offset']:profile['offset'] + profile['samples']]
    assert len(tasks) == profile['samples']
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = InferenceRuntime(args.shared_config)
    assert len(tasks) == runtime.config.runtime.batch_size
    results = []

    def save_measurements():
        (args.output / 'measurements.json').write_text(json.dumps(results, indent=2) + '\n')

    try:
        for size in [1, len(tasks)]:
            result = measure(runtime, tasks[:size], f'warmup-{size}', [None] * size)
            results.append(result)
            save_measurements()
        for repeat in range(profile['repeats']):
            single = [measure(runtime, [task], f'single-{repeat}-{index}', [None])
                      for index, task in enumerate(tasks)]
            batch = measure(runtime, tasks, f'batch-{repeat}', [None] * len(tasks))
            results.extend([*single, batch])
            save_measurements()
            parity = [row['texts'][0] == text for row, text in zip(single, batch['texts'])]
            print(json.dumps({'repeat': repeat, 'single_total_seconds': sum(row['metrics']['elapsed_seconds'] for row in single),
                              'single_batch_text_parity': parity, 'batch': batch['metrics']}), flush=True)
        stops = [profile['stop_string'] if index % 2 else None for index in range(len(tasks))]
        stopped = measure(runtime, tasks, 'mixed-stops', stops)
        results.append(stopped)
        save_measurements()
        expected = [text.split(stop)[0] if stop else text for text, stop in zip(batch['texts'], stops)]
        assert stopped['texts'] == expected, 'Per-row stop boundaries differ from the greedy prefix.'
        assert stopped['metrics']['actual_batch_sizes'] == [len(tasks)], 'Mixed stops split the GPU batch.'
        reused = measure(runtime, tasks, 'mixed-stops-reused', stops)
        results.append(reused)
        save_measurements()
        assert reused['texts'] == expected
        budgets = [profile['budget_prefix_tokens'] if index % 2 else runtime.config.generation.max_new_tokens
                   for index in range(len(tasks))]
        limited = measure(runtime, tasks, 'mixed-budgets', [None] * len(tasks), budgets)
        results.append(limited)
        save_measurements()
        original_ids = {identity.removeprefix(batch['phase'] + '/'): ids
                        for record in batch['batches'] for identity, ids in
                        zip(record['task_ids'], record['output_token_ids'])}
        expected = runtime.backend.tokenizer.batch_decode(
            [original_ids[task['task_id']][:budget] for task, budget in zip(tasks, budgets)], skip_special_tokens=True)
        assert limited['texts'] == expected, 'Per-row token budgets changed valid prefixes.'
        assert limited['metrics']['actual_batch_sizes'] == [len(tasks)], 'Token budgets split the GPU batch.'
        generate = runtime.service._generate_tokens
        trace_events = []

        def profiled(inputs, options, stopping):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                result = generate(inputs, options, stopping)
                torch.cuda.synchronize()
            profiler.export_chrome_trace(str(args.output / 'service-trace.json'))
            trace_events.extend(event.name for event in profiler.events() if 'graphlaunch' in event.name.lower())
            return result

        runtime.service._generate_tokens = profiled
        traced = measure(runtime, tasks, 'trace', [None] * len(tasks))
        traced['profiled'] = True
        results.append(traced)
        save_measurements()
        if runtime.config.runtime.decode_engine == 'cuda_graph':
            assert trace_events, 'No CUDA graph replay was observed in the profiler.'
        output = {'profile': profile, **runtime.metadata(), 'measurements': results,
                  'single_batch_text_parity': parity, 'mixed_stop_parity': True,
                  'mixed_budget_parity': True, 'cuda_graph_launch_events': len(trace_events),
                  'scope': 'Inference infrastructure qualification, not a task-quality benchmark.'}
        (args.output / 'result.json').write_text(json.dumps(output, indent=2) + '\n')
    finally:
        runtime.close()


if __name__ == '__main__':
    main()
