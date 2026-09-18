"""Run one independent structured decision worker per input state."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from jev_spawn.backend import Backend
from jev_spawn.run import Journal, write_json


def invoke(backend, tasks, fields, config):
    states = [task['state'] for task in tasks]
    schemas = [task['fields'] for task in tasks] if fields is None else fields
    if config['method'] == 'compact_json':
        return backend.generate_array_fields(states, schemas, config['controller_max_new_tokens'])
    return backend.score_fields(states, schemas, mode=config['method'])


def schema_batches(tasks, batch_size):
    buckets = {}
    for task in tasks:
        signature = tuple((name, tuple(option['id'] for option in field['options'])) for name, field in task['fields'].items())
        buckets.setdefault(signature, []).append(task)
    return [bucket[start:start + batch_size] for bucket in buckets.values()
            for start in range(0, len(bucket), batch_size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    started_at = time.time()
    config = json.loads(args.config.read_text())
    assert config['method'] in {'shared', 'streamed', 'independent', 'compact_json'}
    assert type(config['warmup_batch']) is bool
    assert config['batch_size'] > 0 and 0 <= args.rank < config['world_size'] <= config['task_count']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(args.rank)
    data = Path(config['tasks_path']).read_bytes()
    rows = [json.loads(line) for line in data.splitlines()]
    if config['fields_path'] is None:
        fields = None
        keys = {'task_id', 'dataset', 'state', 'fields'}
    else:
        fields = json.loads(Path(config['fields_path']).read_text())
        assert fields
        keys = {'task_id', 'dataset', 'state'}
    selected = rows[config['task_offset']:config['task_offset'] + config['task_count']]
    assert len(selected) == config['task_count']
    assert all(set(task) == keys for task in selected)
    assert len({task['task_id'] for task in selected}) == len(selected)
    tasks = selected[args.rank::config['world_size']]
    if fields is not None:
        tasks = [dict(task, fields=fields) for task in tasks]
    assert all(task['fields'] for task in tasks)
    batches = schema_batches(tasks, config['batch_size'])
    directory = Path(config['run_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    done_path = directory / f'done-{args.rank}.json'
    if done_path.exists():
        return
    metadata = {'config': config, 'rank': args.rank, 'method': config['method'], 'fields': fields,
                'task_ids': [task['task_id'] for task in tasks],
                'task_fields': {task['task_id']: task['fields'] for task in tasks} if fields is None else None,
                'field_counts': {task['task_id']: len(task['fields']) for task in tasks},
                'batch_task_ids': [[task['task_id'] for task in batch] for batch in batches]}
    write_json(directory / f'run-{args.rank}.json', metadata)
    load_started = time.perf_counter()
    backend = Backend(config)
    torch.cuda.synchronize(backend.device)
    load_seconds = time.perf_counter() - load_started
    memory = {'resident_after_load_allocated_bytes': torch.cuda.memory_allocated(backend.device),
              'resident_after_load_reserved_bytes': torch.cuda.memory_reserved(backend.device)}
    write_json(directory / f'backend-{args.rank}.json', backend.metadata)
    warmup_seconds = 0.0
    if config['warmup_batch']:
        warmup_started = time.perf_counter()
        warmup = invoke(backend, batches[0], fields, config)
        torch.cuda.synchronize(backend.device)
        warmup_seconds = time.perf_counter() - warmup_started
        write_json(directory / f'warmup-{args.rank}-{os.getpid()}.json',
                   {'task_ids': metadata['batch_task_ids'][0],
                    'elapsed_seconds': warmup_seconds, 'result': warmup, 'scope': 'Warmup only; excluded from worker counts.'})
    memory.update(resident_before_work_allocated_bytes=torch.cuda.memory_allocated(backend.device),
                  resident_before_work_reserved_bytes=torch.cuda.memory_reserved(backend.device))
    torch.cuda.reset_peak_memory_stats(backend.device)
    journal = Journal(directory / f'live-{args.rank}.jsonl', args.rank)
    parent, role = config['job']['id'], config['worker_role']
    plan_path = directory / f'plan-{args.rank}.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        events = [journal.event('dispatch_started', parent=parent, gpu_id=args.rank, tasks=len(tasks))]
        events.extend(journal.event('agent_spawned', task['task_id'], f"{task['task_id']}:{role}", parent, role,
                                    state=task['state'], dataset=task['dataset'], field_names=list(task['fields']),
                                    field_schema=task['fields'],
                                    method=config['method'], gpu_id=args.rank) for task in tasks)
        plan = {'events': events}
        write_json(plan_path, plan)
    spawned = {event['task_id']: event['timestamp'] for event in plan['events'] if event['type'] == 'agent_spawned'}
    completed = 0
    for index, batch in enumerate(batches):
        completed += len(batch)
        path = directory / f'batch-{args.rank}-{index:04d}.json'
        ids = [task['task_id'] for task in batch]
        if path.exists():
            checkpoint = json.loads(path.read_text())
            assert checkpoint['task_ids'] == ids
            continue
        events = [journal.event('agent_started', task['task_id'], f"{task['task_id']}:{role}", parent, role,
                                gpu_id=args.rank, batch=index) for task in batch]
        starts = {event['task_id']: event['timestamp'] for event in events}
        compute_started = time.perf_counter()
        result = invoke(backend, batch, fields, config)
        torch.cuda.synchronize(backend.device)
        call_seconds = time.perf_counter() - compute_started
        events.append(journal.event('batch_completed', parent=parent, role=role, gpu_id=args.rank, batch=index,
                                   method=config['method'], batch_size=len(batch), field_count=len(batch[0]['fields']),
                                   field_decisions=sum(len(task['fields']) for task in batch),
                                   duration_seconds=call_seconds, logical_input_tokens=result['logical_input_tokens'],
                                   computed_input_tokens=result['computed_input_tokens'], padded_input_tokens=result['padded_input_tokens'],
                                   output_tokens=sum(result['output_tokens'])))
        predictions = []
        for position, task in enumerate(batch):
            answer = {name: {'choice': value['choices'][position], 'probabilities': value['probabilities'][position],
                             'option_ids': value['option_ids']} for name, value in result['fields'].items()}
            agent = f"{task['task_id']}:{role}"
            event = journal.event('agent_completed', task['task_id'], agent, parent, role,
                                  fields=answer, method=config['method'], gpu_id=args.rank, batch=index)
            events.append(event)
            predictions.append({'task_id': task['task_id'], 'dataset': task['dataset'], 'agent_id': agent, 'fields': answer,
                                'queue_seconds': starts[task['task_id']] - spawned[task['task_id']],
                                'execution_seconds': event['timestamp'] - starts[task['task_id']],
                                'task_latency_seconds': event['timestamp'] - spawned[task['task_id']]})
        write_json(path, {'task_ids': ids, 'predictions': predictions, 'events': events, 'result': result})
        print(json.dumps({'rank': args.rank, 'batch': index, 'workers_completed': completed,
                          'workers_total': len(tasks), 'seconds': call_seconds}), flush=True)
    torch.cuda.synchronize(backend.device)
    memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(backend.device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(backend.device))
    write_json(done_path, {'rank': args.rank, 'tasks': len(tasks), 'process_started_at': started_at,
                          'ended_at': time.time(), 'model_load_seconds': load_seconds, 'warmup_seconds': warmup_seconds,
                          'cuda_memory': memory, 'memory_scope': 'Current rank process after optional warmup and peak reset. '
                          'Includes resident model; excludes other processes and previous preempted processes.'})


if __name__ == '__main__':
    main()
