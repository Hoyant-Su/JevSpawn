import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import gc
import json
from pathlib import Path
from threading import Event, Barrier
import time

import torch
import yaml

from baselines.common.config import SharedConfig
from baselines.common.runtime import InferenceRuntime
from baselines.common.service import BatchService
from jev_spawn.infra.backend import Backend


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def measure(settings, shared_path, backend, output):
    source = json.loads(Path(settings['source_batches']).read_text())
    requests = settings['requests']
    ids = [request['request_id'] for request in requests]
    assert len(set(ids)) == len(ids)
    assert all(request['after'] is None or request['after'] in ids for request in requests)
    events = {identity: Event() for identity in ids}
    initial = [request for request in requests if request['after'] is None]
    barrier = Barrier(len(initial))
    runtime = InferenceRuntime(shared_path, partial(BatchService, settings={}, prompts={}), backend=backend)
    assert len(initial) == runtime.config.runtime.batch_size
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'runtime.json', runtime.metadata())
    save(output / 'inputs.json', [{'request': request, 'messages': source[request['source_batch']]['messages'][request['source_row']]}
                                for request in requests])

    def execute(request):
        identity = request['request_id']
        if request['after'] is None:
            barrier.wait()
        else:
            events[request['after']].wait()
        runtime.deadlines.start(identity)
        started = time.perf_counter()
        try:
            result, = runtime.service.complete(source[request['source_batch']]['messages'][request['source_row']], request['max_tokens'],
                runtime.config.generation.temperature, stop=request['stop'], task_id=identity, return_tokens=True)
            return {'request_id': identity, 'after': request['after'], 'started_monotonic': started,
                    'finished_monotonic': time.perf_counter(), **result}
        finally:
            events[identity].set()

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=len(requests)) as workers:
            futures = [workers.submit(execute, request) for request in requests]
            results = [future.result() for future in futures]
    finally:
        runtime.close()
        save(output / 'batches.json', runtime.service.records)
        save(output / 'input_failures.json', runtime.service.input_failures)
        save(output / 'scheduling.json', getattr(runtime.service, 'scheduling_records', []))
    elapsed = time.perf_counter() - started
    assert [result['request_id'] for result in results] == ids
    rows = {}
    for record in runtime.service.records:
        for row, identity in enumerate(record['task_ids']):
            assert identity not in rows
            rows[identity] = {'started_monotonic': record['started_monotonic'],
                'queue_seconds': record['queue_seconds'][row], 'decode': record['decode'][row],
                'stop': record['row_stops'][row], 'finish_reason': record['finish_reasons'][row],
                'output_tokens': record['output_tokens'][row], 'token_ids': record['output_token_ids'][row]}
    assert set(rows) == set(ids)
    for request, result in zip(requests, results):
        row = rows[request['request_id']]
        assert result['token_ids'] == row['token_ids']
        assert 0 < len(result['token_ids']) <= request['max_tokens']
        assert row['stop'] == request['stop']
        assert row['finish_reason'] in {'stop', 'length'}
    witness = settings['refill_witness']
    results_by_id = {result['request_id']: result for result in results}
    admitted_before_survivor_finished = (rows[witness['incoming']]['started_monotonic'] <
        results_by_id[witness['survivor']]['finished_monotonic'])
    summary = {'requests': len(results), 'elapsed_seconds': elapsed, 'rows': rows,
        'peak_allocated_bytes': max(record['peak_allocated_bytes'] for record in runtime.service.records),
        'peak_reserved_bytes': max(record['peak_reserved_bytes'] for record in runtime.service.records),
        'admitted_before_survivor_finished': admitted_before_survivor_finished,
        'refill_witness': witness, 'results': results}
    save(output / 'completion.json', summary)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def qualify(settings):
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    original = yaml.safe_load(Path(settings['shared_config']).read_text())
    assert original['runtime']['generation_scheduling'] == 'cohort'
    backend = Backend(SharedConfig.load(settings['shared_config']).backend())
    candidate_config = yaml.safe_load(Path(settings['candidate_config']).read_text())
    expected = yaml.safe_load(Path(settings['shared_config']).read_text())
    expected['runtime'].update(generation_scheduling='continuous', cache_allocation='shared_refill_cache_arena_v1')
    assert candidate_config == expected
    paths = {'cohort': settings['shared_config'], 'continuous': settings['candidate_config']}
    save(output / 'specification.json', settings)
    reference = measure(settings, paths['cohort'], backend, output / 'cohort')
    candidate = measure(settings, paths['continuous'], backend, output / 'continuous')
    agreement = {identity: reference['rows'][identity]['token_ids'] == row['token_ids']
                 and reference['rows'][identity]['finish_reason'] == row['finish_reason']
                 for identity, row in candidate['rows'].items()}
    summary = {'exact_output_and_stop_agreement': agreement,
        'all_outputs_and_stops_equal': all(agreement.values()),
        'refill_before_survivor_finished': candidate['admitted_before_survivor_finished'],
        'cohort_seconds': reference['elapsed_seconds'], 'continuous_seconds': candidate['elapsed_seconds'],
        'cohort_peak_allocated_bytes': reference['peak_allocated_bytes'],
        'continuous_peak_allocated_bytes': candidate['peak_allocated_bytes'],
        'capacity': original['model']['max_input_tokens'] + original['generation']['max_new_tokens'],
        'scope': 'Real request service qualification with explicit profiling budgets; not task-level baseline accuracy.'}
    save(output / 'completion.json', summary)
    assert summary['refill_before_survivor_finished']
    assert summary['all_outputs_and_stops_equal']
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    args = parser.parse_args()
    qualify(json.loads(args.specification.read_text()))


if __name__ == '__main__':
    main()
