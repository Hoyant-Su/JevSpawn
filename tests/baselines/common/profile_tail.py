import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier
import time

import torch

from baselines.common.runtime import InferenceRuntime
from baselines.common.tasks import read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shared-config', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    profile = read(args.profile)
    source = read(profile['requests'])
    requests = source[profile['batch_index']]
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = InferenceRuntime(args.shared_config)
    size = runtime.config.runtime.batch_size
    assert len(requests['messages']) == size
    barrier = Barrier(size)
    results = [None] * size
    started = time.perf_counter()

    def solve(index):
        identity = requests['task_ids'][index]
        runtime.deadlines.start(identity)
        barrier.wait()
        calls = []
        for turn in range(profile['turns']):
            begin = time.perf_counter()
            output = runtime.service.complete(requests['messages'][index],
                        profile['max_new_tokens'], runtime.config.generation.temperature,
                        stop=requests['row_stops'][index], task_id=identity)
            calls.append({'turn': turn, 'text': output[0], 'seconds': time.perf_counter() - begin,
                          'finished_seconds': time.perf_counter() - started})
        results[index] = {'task_id': identity, 'calls': calls}

    try:
        with ThreadPoolExecutor(max_workers=size) as pool:
            list(pool.map(solve, range(size)))
        elapsed = time.perf_counter() - started
    finally:
        runtime.close()
        (args.output / 'batches.json').write_text(json.dumps(runtime.service.records, indent=2) + '\n')

    result = {'profile': profile, **runtime.metadata(), 'elapsed_seconds': elapsed,
              'results': results, 'batches': runtime.service.records,
              'scope': 'Fixed real model requests; repeated calls expose dependency scheduling, not task accuracy.'}
    if runtime.config.runtime.generation_scheduling == 'continuous':
        result['scheduling'] = runtime.service.scheduling_records
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'seconds': elapsed, 'completed_calls': sum(len(row['calls']) for row in results)}), flush=True)


if __name__ == '__main__':
    main()
