import argparse
import json
from pathlib import Path
import statistics

from methods.latent_readout.inputs import condition_id, read_rows


def interval_summary(values):
    return {'count': len(values), 'min_seconds': min(values), 'median_seconds': statistics.median(values),
            'max_seconds': max(values), 'at_least_100ms': sum(value >= 0.1 for value in values)} if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.output / 'protocol.json').read_text())
    settings, tasks = protocol['settings'], protocol['tasks']
    blocks = [json.loads(path.read_text()) for path in sorted((args.output / 'measured').glob('block-*.json'))]
    conditions = [condition_id(value) for value in settings['conditions']]
    records = {condition: {} for condition in conditions}
    for block in blocks:
        group = records[condition_id(block['condition'])]
        for row in block['predictions']:
            assert row['task_id'] not in group
            group[row['task_id']] = row
    expected = {task['task_id'] for task in tasks}
    assert all(set(group) == expected for group in records.values())
    assert sum(map(len, records.values())) == len(tasks) * len(conditions)
    labels = {row['task_id']: row['labels'] for source in settings['sources'] for row in read_rows(source['labels'])}
    results = {}
    for condition, group in records.items():
        datasets = {}
        for source in settings['sources']:
            rows = [row for row in group.values() if row['dataset'] == source['dataset']]
            datasets[source['dataset']] = {'count': len(rows), 'valid': sum(row['valid'] for row in rows),
                'correct': sum(row['valid'] and row['answer'] == labels[row['task_id']][row['field_id']] for row in rows)}
        chosen = [block for block in blocks if condition_id(block['condition']) == condition]
        text_intervals = [value for row in group.values() for value in row['text_itl_seconds']]
        latent_intervals = [call['seconds'] for block in chosen for call in block['forward_calls'] if call['phase'] == 'latent']
        results[condition] = {'datasets': datasets,
            'compute_seconds': sum(block['compute_seconds'] for block in chosen),
            'peak_allocated_bytes': max(block['peak_allocated_bytes'] for block in chosen),
            'peak_reserved_bytes': max(block['peak_reserved_bytes'] for block in chosen),
            'batch_sizes': [block['batch_size'] for block in chosen],
            'latent_steps': sum(row['latent_steps'] for row in group.values()),
            'generated_tokens': sum(row['generated_tokens'] for row in group.values()),
            'text_itl': interval_summary(text_intervals),
            'latent_trunk_intervals': interval_summary(latent_intervals),
            'interval_scope': 'Latent intervals measure native trunk forwards only; alignment, rendering, and residual work remain in other_compute and complete latency.',
            'phase_seconds': {phase: sum(block['phase_seconds'][phase] for block in chosen)
                              for phase in chosen[0]['phase_seconds']}}
    paired = []
    for task in tasks:
        task_id, field_id = task['task_id'], task['field_id']
        paired.append({'task_id': task_id, 'dataset': task['dataset'],
                       'correct': {condition: row[task_id]['valid'] and row[task_id]['answer'] == labels[task_id][field_id]
                                   for condition, row in records.items()}})
    promotion = {}
    for condition in ['latent-10', 'latent-30']:
        data = results[condition]['datasets']
        improves = sum(value['correct'] for value in data.values()) > sum(value['correct'] for value in results['latent-0']['datasets'].values())
        preserves = all(value['correct'] >= results['text']['datasets'][name]['correct'] for name, value in data.items())
        faster = results[condition]['compute_seconds'] < results['text']['compute_seconds']
        promotion[condition] = {'improves_zero': improves, 'preserves_text_by_dataset': preserves,
                                'faster_than_text': faster, 'passes': improves and preserves and faster}
    attempts = [json.loads(path.read_text()) for path in sorted(args.output.glob('attempt-*/setup.json'))]
    startup = sum(row['startup_seconds'] for row in attempts)
    warmup = sum(json.loads(path.read_text())['compute_seconds'] for path in args.output.glob('attempt-*/warmup/block-*.json'))
    compute = sum(block['compute_seconds'] for block in blocks)
    result = {'scope': 'Development capability only; no spawning or new-method claim.', 'outputs': len(tasks) * len(conditions),
              'conditions': results, 'paired_correctness': paired, 'promotion': promotion,
              'startup_seconds': startup, 'completed_warmup_compute_seconds': warmup,
              'complete_measured_compute_seconds': compute, 'cold_start_total_accounted_seconds': startup + warmup + compute,
              'attempts': len(attempts),
              'restart_scope': 'Interrupted incomplete blocks are rerun. Uncommitted preemption time is not included in compute totals.'}
    destination = args.output / 'summary.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
