from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import statistics
import sys
from threading import Barrier
import time
import traceback

import torch

from baselines.lats.adapter import ChatModel, SandboxExecutor, run_task


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def timing(records):
    intervals = sorted(value for batch in records for row in batch['decode']
                       for value in row['inter_token_seconds'])
    return {'batches': len(records), 'batch_sizes': [r['batch_size'] for r in records],
            'interval_count': len(intervals), 'median_ms': statistics.median(intervals) * 1000,
            'p95_ms': intervals[int(.95 * (len(intervals) - 1))] * 1000,
            'maximum_ms': max(intervals) * 1000,
            'intervals_over_100ms': sum(value >= .1 for value in intervals)}


def run_stage(core, result_type, service, rows, settings, output, stage):
    directory = output / stage
    directory.mkdir()
    barrier = Barrier(len(rows))

    def task(row):
        task_dir = directory / row['task_id'].replace('/', '_')
        task_dir.mkdir()
        calls, events = [], []
        executor = SandboxExecutor(Path(settings['upstream']) / 'programming/executors', settings['evaluation_script'], settings['sandbox'],
                                   task_dir / 'tools', settings['sandbox_memory_mb'], result_type)

        def complete(messages, max_tokens, temperature, n, stop):
            if len(calls) >= settings['generation_call_cap']:
                raise RuntimeError('Declared generation call budget exhausted')
            assert max_tokens == settings['max_tokens_per_call']
            assert temperature in settings['temperatures']
            record = {'index': len(calls), 'messages': messages, 'max_tokens': max_tokens,
                      'temperature': temperature, 'n': n, 'stop': stop}
            calls.append(record)
            write(task_dir / 'calls.json', calls)
            started = time.perf_counter()
            record['responses'] = service.complete(messages, max_tokens, temperature, n=n, stop=stop,
                                                    task_id=row['task_id'])
            record['elapsed_seconds'] = time.perf_counter() - started
            write(task_dir / 'calls.json', calls)
            return record['responses']

        def profile(frame, event, value):
            if event != 'return' or frame.f_code.co_filename != core.__file__:
                return
            node = frame.f_locals.get('self')
            if isinstance(node, core.Node):
                events.append({'function': frame.f_code.co_name, 'depth': node.depth,
                               'visits': node.visits, 'value': node.value,
                               'children': len(node.children)})

        barrier.wait()
        started = time.perf_counter()
        sys.setprofile(profile)
        try:
            result = run_task(core, row, ChatModel(settings['model_name'], complete), executor,
                              max_iters=settings['max_iters'], expansion_factor=settings['expansion_factor'],
                              number_of_tests=settings['number_of_tests'], log_path=task_dir / 'upstream.jsonl',
                              verbose=False)
            result['status'] = 'completed'
        except Exception:
            result = {'task_id': row['task_id'], 'status': 'failed', 'error': traceback.format_exc(),
                      'public_evaluations': executor.evaluations, 'tool_calls': executor.calls}
        finally:
            sys.setprofile(None)
        result.update(elapsed_seconds=time.perf_counter() - started, model_calls=len(calls), node_events=events)
        write(task_dir / 'result.json', result)
        print(json.dumps({'stage': stage, 'task_id': row['task_id'], 'status': result['status'],
                          'calls': len(calls), 'seconds': result['elapsed_seconds']}), flush=True)
        return result

    torch.manual_seed(settings['seed'])
    random.seed(settings['seed'])
    index = len(service.records)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(rows)) as pool:
        results = list(pool.map(task, rows))
    seconds = time.perf_counter() - started
    records = service.records[index:]
    write(directory / 'batches.json', records)
    write(directory / 'summary.json', {'elapsed_seconds': seconds, 'task_count': len(rows),
                                      'completed': sum(r['status'] == 'completed' for r in results),
                                      'model_calls': sum(r['model_calls'] for r in results),
                                      'timing': timing(records)})
    with (directory / 'solutions.jsonl').open('w') as stream:
        for result in results:
            if result['status'] == 'completed':
                stream.write(json.dumps({'task_id': result['task_id'], 'solution': result['solution']}) + '\n')
    return results
