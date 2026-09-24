from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import json
import statistics
from threading import Barrier
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def run_phase(service, rows, settings, prompts, output, phase):
    solve = importlib.import_module(settings['adapter_module']).solve
    directory = output / phase
    directory.mkdir()
    barrier = Barrier(len(rows))

    def run_task(row):
        barrier.wait()
        started = time.perf_counter()
        complete = partial(service.complete, task_id=row['task_id'])
        try:
            result = solve(row, complete, settings, prompts)
            result['status'] = 'completed'
        except Exception as error:
            result = {'task_id': row['task_id'], 'status': 'failed',
                      'error': type(error).__name__ + ': ' + str(error)}
        result['elapsed_seconds'] = time.perf_counter() - started
        write(directory / (row['task_id'].replace('/', '_') + '.json'), result)
        print(json.dumps({'phase': phase, 'task_id': row['task_id'],
                          'status': result['status'], 'seconds': result['elapsed_seconds']}), flush=True)
        return result

    start = len(service.records)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=settings['batch_size']) as pool:
        results = list(pool.map(run_task, rows))
    elapsed = time.perf_counter() - started
    records = service.records[start:]
    write(directory / 'batches.json', records)
    intervals = [value for batch in records for row in batch['decode']
                 for value in row['inter_token_seconds']]
    summary = {'task_count': len(rows), 'completed': sum(r['status'] == 'completed' for r in results),
               'elapsed_seconds': elapsed, 'model_calls': sum(len(r['task_ids']) for r in records),
               'batch_sizes': [r['batch_size'] for r in records],
               'output_tokens': sum(sum(r['output_tokens']) for r in records),
               'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in records),
               'median_itl_ms': 1000 * statistics.median(intervals),
               'max_itl_ms': 1000 * max(intervals),
               'intervals_over_100ms': sum(t >= .1 for t in intervals)}
    write(directory / 'summary.json', summary)
    return results
