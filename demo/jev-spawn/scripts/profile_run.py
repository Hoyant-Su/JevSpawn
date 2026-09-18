"""Separate aggregate throughput from observed worker latency and batch packing."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in (args.run_dir / 'events.jsonl').read_text().splitlines()]
    summary = json.loads((args.run_dir / 'summary.json').read_text())
    starts, task_starts, task_latencies = {}, {}, []
    latencies, batches = collections.defaultdict(list), collections.defaultdict(list)
    for event in events:
        if event['type'] == 'agent_started':
            starts[event['agent_id']] = event['timestamp']
            if event['role'] not in {'review', 'repair'}:
                task_starts[event['task_id']] = event['timestamp']
        if event['type'] == 'agent_completed':
            role = event['role'] if event['role'] in {'review', 'repair'} else 'implement'
            latencies[role].append(event['timestamp'] - starts[event['agent_id']])
        if event['type'] == 'batch_completed':
            batches[event['role']].append(event['payload'])
        if event['type'] == 'task_completed':
            task_latencies.append(event['timestamp'] - task_starts[event['task_id']])
    result = {
        'run_id': summary['run_id'],
        'tasks_per_second': summary['task_count'] / summary['elapsed_seconds'],
        'task_flow_seconds': {'count': len(task_latencies), 'median': statistics.median(task_latencies),
                              'minimum': min(task_latencies), 'maximum': max(task_latencies)},
        'worker_latency_seconds': {role: {'count': len(values), 'median': statistics.median(values),
                                          'minimum': min(values), 'maximum': max(values)}
                                   for role, values in latencies.items()},
        'batches': {role: {'count': len(rows), 'mean_size': statistics.mean(r['batch_size'] for r in rows),
                           'summed_gpu_seconds': sum(r['duration_seconds'] for r in rows)}
                    for role, rows in batches.items()},
        'interpretation': 'Task throughput is a batch aggregate, not individual latency. Worker latency runs from start to batch-result return and excludes queue waiting. Task flow runs from initial coder start through recorded task completion, including subsequent decisions, workers and stage barriers. Summed GPU durations overlap across replicas; they are not wall-clock time.'}
    (args.run_dir / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
