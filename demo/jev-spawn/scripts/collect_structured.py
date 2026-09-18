"""Collect completed structured workers without reading evaluation labels."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = Path(config['run_dir'])
    runs = [json.loads((directory / f'run-{rank}.json').read_text()) for rank in range(config['world_size'])]
    done = [json.loads((directory / f'done-{rank}.json').read_text()) for rank in range(config['world_size'])]
    plans = [json.loads((directory / f'plan-{rank}.json').read_text()) for rank in range(config['world_size'])]
    batches = [json.loads(path.read_text()) for path in sorted(directory.glob('batch-*.json'))]
    expected = [task for run in runs for task in run['task_ids']]
    predictions = [prediction for batch in batches for prediction in batch['predictions']]
    assert len(predictions) == len(expected) == len(set(expected)) == config['task_count']
    by_id = {prediction['task_id']: prediction for prediction in predictions}
    assert set(by_id) == set(expected)
    fields = runs[0]['fields']
    task_fields = ({task: schema for run in runs for task, schema in run['task_fields'].items()}
                   if fields is None else None)
    assert all(list(prediction['fields']) == list(task_fields[prediction['task_id']] if fields is None else fields)
               for prediction in predictions)
    # Interleave deterministic rank shards to restore original input order.
    order = [task for group in zip(*[run['task_ids'] for run in runs]) for task in group]
    tail = min(len(run['task_ids']) for run in runs)
    order.extend(task for run in runs for task in run['task_ids'][tail:])
    assert len(order) == len(expected)
    predictions = [by_id[task] for task in order]
    events = sorted([event for part in plans + batches for event in part['events']],
                    key=lambda event: (event['timestamp'], event['event_id']))
    running = queued = peak_running = peak_queued = 0
    for event in events:
        kind = event['type']
        queued += int(kind == 'agent_spawned') - int(kind == 'agent_started')
        running += int(kind == 'agent_started') - int(kind == 'agent_completed')
        assert min(running, queued) >= 0
        peak_running, peak_queued = max(peak_running, running), max(peak_queued, queued)
    assert running == queued == 0
    calls = [event['payload'] for event in events if event['type'] == 'batch_completed']
    elapsed = events[-1]['timestamp'] - events[0]['timestamp']
    count = len(predictions)
    field_counts = {prediction['task_id']: len(prediction['fields']) for prediction in predictions}
    decisions = sum(field_counts.values())
    summary = {
        'run_id': config['run_id'], 'job': config['job'], 'method': config['method'], 'model': config['model_path'],
        'worker_count': count, 'workers_completed': count, 'field_count': len(fields) if fields is not None else None,
        'field_count_distribution': dict(collections.Counter(field_counts.values())), 'field_decisions': decisions,
        'worker_definition': 'One independent state and role-scoped model call within a batch; fields are decisions, not additional agents.',
        'datasets': dict(collections.Counter(row['dataset'] for row in predictions)),
        'gpu_replicas': config['world_size'], 'coordinator_extra_model_replicas': 0,
        'batch_size_per_gpu': config['batch_size'], 'observed_batch_sizes': sorted({call['batch_size'] for call in calls}),
        'batching': 'Fixed rank shards, then schema-structure buckets in first-seen order; input order preserved within each bucket.',
        'peak_running_workers': peak_running, 'peak_queued_workers': peak_queued,
        'elapsed_seconds': elapsed, 'workers_per_second': count / elapsed, 'field_decisions_per_second': decisions / elapsed,
        'time_scope': 'First worker dispatch through last worker completion, including queue and host time and restart gaps. '
                      'Each rank loads and optionally warms its model before dispatch; other ranks may still be loading.',
        'current_process_end_to_end_seconds': max(rank['ended_at'] for rank in done) - min(rank['process_started_at'] for rank in done),
        'end_to_end_scope': 'Current rank invocations after argument parsing through final checkpoint; includes model loading and warmup, '
                            'excludes Python import time and pre-restart processes.',
        'model_load_seconds_per_gpu': [rank['model_load_seconds'] for rank in done],
        'warmup_seconds_per_gpu': [rank['warmup_seconds'] for rank in done],
        'model_call_seconds_sum': sum(call['duration_seconds'] for call in calls),
        'logical_input_tokens': sum(call['logical_input_tokens'] for call in calls),
        'computed_input_tokens': sum(call['computed_input_tokens'] for call in calls),
        'padded_input_tokens': sum(call['padded_input_tokens'] for call in calls),
        'generated_tokens': sum(call['output_tokens'] for call in calls),
        'mean_queue_seconds': statistics.mean(row['queue_seconds'] for row in predictions),
        'mean_task_latency_seconds': statistics.mean(row['task_latency_seconds'] for row in predictions),
        'cuda_memory': {'per_gpu': [{'gpu_id': rank, **record['cuda_memory']} for rank, record in enumerate(done)],
                        'scope': done[0]['memory_scope'], 'measurement': 'PyTorch allocated/reserved bytes, not whole-device memory.'},
        'evaluation_status': 'Not evaluated; no gold labels were read.',
    }
    write_json(directory / 'run.json', {'config': config, 'task_ids': order, 'fields': fields,
                                      'task_fields': task_fields, 'field_counts': field_counts, 'method': config['method']})
    write_json(directory / 'summary.json', summary)
    (directory / 'predictions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in predictions))
    events.insert(0, {'event_id': 'root', 'timestamp': events[0]['timestamp'], 'type': 'run_started', 'agent_id': config['job']['id'],
                      'payload': {'run_id': config['run_id'], 'job': config['job'], 'method': config['method']}})
    events.append({'event_id': 'done', 'timestamp': events[-1]['timestamp'], 'type': 'run_completed', 'payload': summary})
    (directory / 'events.jsonl').write_text(''.join(json.dumps(event, ensure_ascii=False) + '\n' for event in events))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
