import argparse
import importlib
import json
import multiprocessing as mp
from pathlib import Path
import time

from baselines.resource_trials import DeliveryTrace
from baselines.resource_budget.memory import ProcessMemory


def child(module, config, output, connection):
    importlib.import_module(module).run(config, output, connection)


def supervise(config, output, task_ids, *, module, memory=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'trial.json').write_text(json.dumps(config, indent=2) + '\n')
    context = mp.get_context('spawn')
    connection, worker_connection = context.Pipe()
    process = context.Process(target=child, args=(module, config, output, worker_connection))
    trace = DeliveryTrace(output / 'delivery.jsonl', task_ids)
    started = time.perf_counter()
    process.start()
    worker_connection.close()
    if memory is not None:
        memory.start(process.pid)
    origin = None
    status = 'failed'
    ready = None
    error = None
    try:
        while process.is_alive() or connection.poll():
            if memory is not None:
                memory.sample()
            now = time.perf_counter()
            cutoff = started + config['startup_timeout_seconds'] if origin is None else origin + config['deadline_seconds']
            if now >= cutoff:
                status = 'startup_timeout' if origin is None else 'deadline'
                break
            if not connection.poll(min(config['poll_seconds'], cutoff - now)):
                continue
            try:
                event = connection.recv()
            except EOFError:
                break
            if event['event'] == 'ready':
                assert origin is None
                ready = event
                trace.start()
                origin = trace.origin
                connection.send({'event': 'start'})
            elif event['event'] == 'delivery':
                assert origin is not None
                trace.deliver(event['results'])
            elif event['event'] in {'completed', 'out_of_memory', 'failed'}:
                status = event['event']
                error = event['error'] if status != 'completed' else None
                break
            else:
                raise ValueError('Unsupported worker event: ' + event['event'])
    except Exception as failure:
        status, error = "failed", type(failure).__name__ + ": " + str(failure)
    stopped = time.perf_counter()
    if status != 'completed' and process.is_alive():
        process.kill()
    process.join(config['join_timeout_seconds'])
    if process.is_alive():
        status, error = 'failed', 'Worker did not exit within the declared join timeout.'
        process.kill()
        process.join(config['join_timeout_seconds'])
    assert not process.is_alive(), 'Trial child survived forced termination.'
    if status == 'completed' and process.exitcode != 0:
        status, error = 'failed', 'Worker failed after reporting completion.'
    if origin is not None:
        trace.finish(status)
    else:
        trace.stream.close()
    finished = time.perf_counter()
    summary = {'status': status, 'error': error, 'worker_exitcode': process.exitcode,
               'startup_seconds': (origin or finished) - started,
               'service_seconds': None if origin is None else finished - origin,
               'stop_requested_seconds': None if origin is None else stopped - origin,
               'stopping_overshoot_seconds': max(0., finished - cutoff) if status == 'deadline' else None,
               'ready': ready, 'assigned_tasks': len(task_ids)}
    if memory is not None:
        summary['physical_memory'] = memory.close()
        assert status not in {'completed', 'deadline'} or summary['physical_memory']['samples'] > 0
    (output / 'terminal.json').write_text(json.dumps(summary, indent=2) + '\n')
    connection.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert set(config) == {'settings', 'study', 'policy', 'memory_gib', 'deadline_seconds',
                           'startup_timeout_seconds', 'poll_seconds', 'join_timeout_seconds'}
    assert config['policy'] in {'single_reasoning', 'formal_choices', 'agentprune', 'latentmas'}
    study = json.loads(Path(config['study']).read_text())
    assert config['memory_gib'] in study['memory_budget']['gib']
    assert config['deadline_seconds'] == max(study['time_budget']['seconds'])
    assert all(config[key] > 0 for key in ['memory_gib', 'deadline_seconds',
                                         'startup_timeout_seconds', 'poll_seconds', 'join_timeout_seconds'])
    settings = json.loads(Path(config['settings']).read_text())
    tasks = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()]
    assert len(tasks) == settings['task_count']
    result = supervise(config, args.output, [row['task_id'] for row in tasks],
                       module='baselines.resource_budget.worker',
                       memory=ProcessMemory(args.output / 'physical-memory.jsonl'))
    print(json.dumps(result))
    if result['status'] in {'failed', 'startup_timeout'}:
        raise RuntimeError(result['error'] or result['status'])


if __name__ == '__main__':
    main()
