import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
from threading import Barrier, Lock
import time
from jev_spawn.infra.prompts import load_prompt


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_protocol(path):
    settings = read(path)
    native = read(settings['native_config'])
    tasks = rows(settings['tasks'])
    development = {r['task_id']: r for r in rows(settings['warmup_tasks'])}
    warmup = [development[task_id] for task_id in settings['warmup_task_ids']]
    assert len(tasks) == settings['task_count']
    assert len({r['task_id'] for r in tasks}) == len(tasks)
    assert len(warmup) == len({r['task_id'] for r in warmup})
    assert 1 <= len(warmup) <= settings['block_size']
    assert not {r['task_id'] for r in tasks} & {r['task_id'] for r in warmup}
    assert native['batch_size'] == settings['block_size'] == 8
    assert settings['interface'] in ['sync_fields', 'async_flat']
    for task in tasks + warmup:
        assert set(task['fields']) == {'q0'}
        ids = [o['id'] for o in task['fields']['q0']['options']]
        assert len(ids) == len(set(ids)) and len(ids) >= 2
    return {'settings': settings, 'native': native, 'prompts': load_prompt(settings['prompts']),
            'tasks': tasks, 'warmup': warmup}


def blocks(protocol):
    size = protocol['settings']['block_size']
    return [protocol['tasks'][start:start + size]
            for start in range(0, len(protocol['tasks']), size)]


def completed(directory, task_ids, seed):
    path = directory / 'complete.json'
    if not path.exists():
        return None
    block = read(path)
    assert block['task_ids'] == task_ids and block['seed'] == seed
    assert [r['task_id'] for r in block['results']] == task_ids
    return block


def flat_task(task):
    field = task['fields']['q0']
    return {'task_id': task['task_id'], 'state': task['state'], 'question': field['question'],
            'options': '\n'.join(o['id'] + ': ' + o['description'] for o in field['options'])}


def metrics(records, elapsed):
    intervals = [t for batch in records for row in batch['decode'] for t in row['inter_token_seconds']]
    return {'elapsed_seconds': elapsed, 'model_calls': sum(len(r['task_ids']) for r in records),
            'output_tokens': sum(sum(r['output_tokens']) for r in records),
            'actual_batch_sizes': [r['batch_size'] for r in records],
            'truncated_calls': sum(sum(r['truncated']) for r in records),
            'peak_allocated_bytes': max((r['peak_allocated_bytes'] for r in records), default=None),
            'median_itl_ms': statistics.median(intervals) * 1000 if intervals else None,
            'max_itl_ms': max(intervals) * 1000 if intervals else None,
            'intervals_over_100ms': sum(t >= .1 for t in intervals)}


def execute_block(service, solve, tasks, protocol, directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    attempts = list(directory.glob('attempt-*'))
    attempt = directory / f'attempt-{len(attempts):04d}'
    attempt.mkdir()
    settings = protocol['settings']
    barrier = Barrier(len(tasks))

    def run_task(task):
        requests = []
        lock = Lock()

        def complete(messages, max_tokens, temperature, n=1, stop=None):
            request = {'messages': messages, 'max_tokens': max_tokens, 'temperature': temperature,
                       'n': n, 'stop': stop, 'started': time.perf_counter()}
            with lock:
                request['call_id'] = len(requests)
                requests.append(request)
            try:
                request['responses'] = service.complete(messages, max_tokens, temperature,
                                                        n=n, stop=stop, task_id=task['task_id'])
                return request['responses']
            except Exception as error:
                request['error'] = type(error).__name__ + ': ' + str(error)
                raise
            finally:
                request['elapsed_seconds'] = time.perf_counter() - request['started']

        barrier.wait()
        started = time.perf_counter()
        try:
            if settings['interface'] == 'async_flat':
                result = asyncio.run(solve(flat_task(task), complete, settings['adapter_config']))
            else:
                result = solve(task, complete, settings['adapter_config'], protocol['prompts'])
            assert result['task_id'] == task['task_id']
            result['status'] = 'completed'
        except Exception as error:
            result = {'task_id': task['task_id'], 'status': 'failed',
                      'error': type(error).__name__ + ': ' + str(error)}
        result['elapsed_seconds'] = time.perf_counter() - started
        result['requests'] = requests
        save(attempt / (task['task_id'].replace('/', '_') + '.json'), result)
        return result

    start_record = len(service.records)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=settings['block_size']) as pool:
        results = list(pool.map(run_task, tasks))
    elapsed = time.perf_counter() - started
    records = service.records[start_record:]
    task_ids = [r['task_id'] for r in tasks]
    assert all(set(r['task_ids']) <= set(task_ids) for r in records)
    assert all(r['batch_size'] == len(r['task_ids']) <= settings['block_size'] for r in records)
    block = {'task_ids': [r['task_id'] for r in tasks], 'seed': seed, 'attempt': attempt.name,
             'results': results, 'batches': records, 'metrics': metrics(records, elapsed)}
    save(directory / 'complete.json', block)
    service.records.clear()
    print(json.dumps({'directory': str(directory), 'tasks': len(tasks), **block['metrics']}), flush=True)
    return block


def run(protocol, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'protocol.json').exists():
            assert read(output / 'protocol.json') == protocol, 'The resumed protocol or task contents changed.'
        else:
            save(output / 'protocol.json', protocol)
        settings = protocol['settings']
        pending = [(index, tasks) for index, tasks in enumerate(blocks(protocol))
                   if completed(output / f'block-{index:04d}', [r['task_id'] for r in tasks],
                                protocol['native']['seed'] + index + 1) is None]
        if not pending:
            return
        sys.path[:0] = settings['python_paths']
        solve = importlib.import_module(settings['adapter_module']).solve
        backend_class = importlib.import_module('jev_spawn.infra.backend').Backend
        service_class = importlib.import_module('baselines.official.model_service').GenerationService
        torch = importlib.import_module('torch')
        backend = backend_class(protocol['native'])
        service = service_class(backend, settings['block_size'], settings['batch_wait_seconds'])
        session = output / f'session-{len(list(output.glob("session-*"))):04d}'
        session.mkdir()
        save(session / 'backend.json', {'backend': backend.metadata,
             'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
             'pending_blocks': [index for index, tasks in pending]})
        try:
            seed = protocol['native']['seed']
            random.seed(seed)
            torch.manual_seed(seed)
            execute_block(service, solve, protocol['warmup'], protocol, session / 'warmup', seed)
            for index, tasks in pending:
                seed = protocol['native']['seed'] + index + 1
                random.seed(seed)
                torch.manual_seed(seed)
                execute_block(service, solve, tasks, protocol, output / f'block-{index:04d}', seed)
        finally:
            service.close()


def evaluate(protocol, output):
    assert read(output / 'protocol.json') == protocol
    results = []
    summaries = []
    for index, tasks in enumerate(blocks(protocol)):
        block = completed(output / f'block-{index:04d}', [r['task_id'] for r in tasks],
                          protocol['native']['seed'] + index + 1)
        assert block is not None, f'Incomplete block {index}'
        results.extend(block['results'])
        summaries.append(block['metrics'])
    labels = {r['task_id']: r['labels']['q0'] for r in rows(protocol['settings']['labels'])}
    assert set(labels) == {r['task_id'] for r in results}
    assert all(labels[r['task_id']] in [o['id'] for o in r['fields']['q0']['options']]
               for r in protocol['tasks'])
    summary = {'tasks': len(results), 'blocks': len(summaries),
               'correct': sum(r.get('answer') == labels[r['task_id']] for r in results),
               'valid': sum(r.get('answer') in [o['id'] for o in task['fields']['q0']['options']]
                            for r, task in zip(results, protocol['tasks'])),
               'adapter_exceptions': sum(r['status'] == 'failed' for r in results),
               'model_request_errors': sum('error' in c for r in results for c in r['requests']),
               'block_metrics': summaries,
               'measurement_scope': 'Completed held-out blocks only. Excludes startup, development warmup and incomplete attempts.',
               'budget': protocol['settings']['budget']}
    for key in ['elapsed_seconds', 'model_calls', 'output_tokens', 'truncated_calls', 'intervals_over_100ms']:
        summary[key] = sum(s[key] for s in summaries)
    for key in ['peak_allocated_bytes', 'max_itl_ms']:
        summary[key] = max((s[key] for s in summaries if s[key] is not None), default=None)
    summary['accuracy'] = summary['correct'] / summary['tasks']
    save(output / 'evaluation.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest='mode', required=True)
    for name in ['run', 'evaluate', 'validate']:
        command = modes.add_parser(name)
        command.add_argument('--settings', type=Path, required=True)
        if name != 'validate':
            command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.settings)
    if args.mode == 'run':
        run(protocol, args.output)
    elif args.mode == 'evaluate':
        print(json.dumps(evaluate(protocol, args.output), indent=2))
    else:
        print(json.dumps({'tasks': len(protocol['tasks']), 'block_sizes': list(map(len, blocks(protocol))),
                          'warmup_task_ids': [r['task_id'] for r in protocol['warmup']]}))


if __name__ == '__main__':
    main()
