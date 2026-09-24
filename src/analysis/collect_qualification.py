import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def collect(manifest):
    expected = set(manifest['task_ids'])
    methods, configurations = {}, set()
    for lane in manifest['lanes'].values():
        for entry in lane['methods']:
            root, evaluation = Path(entry['run']), Path(entry['evaluation'])
            records = [json.loads(path.read_text()) for path in sorted(root.glob('task-*.json'))]
            assert len({row['task_id'] for row in records}) == len(records)
            assert {row['task_id'] for row in records} <= expected
            result = {'run': str(root), 'evaluation': str(evaluation), 'evaluated': evaluation.exists(),
                      'recorded_task_count': len(records),
                      'status_counts': dict(Counter(row['status'] for row in records))}
            methods[entry['method']] = result
            if not evaluation.exists():
                continue
            scores = json.loads(evaluation.read_text())
            assert {row['task_id'] for row in scores['scores']} == expected
            result['datasets'] = scores['datasets']
            result['failures'] = [{'task_id': row['task_id'], 'status': row['status'],
                                  'error': row['error'], 'elapsed_seconds': row['elapsed_seconds']}
                                 for row in records if row['status'] != 'completed']
            sessions, batches, covered, elapsed = [], [], [], []
            for directory in sorted(root.glob('session-*')):
                runtime = json.loads((directory / 'runtime.json').read_text())
                configurations.add(json.dumps(runtime['shared_config'], sort_keys=True))
                filenames = ('completion.json', 'batches.json', 'elapsed.json')
                missing = [name for name in filenames if not (directory / name).exists()]
                sessions.append({'session': directory.name, 'missing_telemetry': missing})
                if missing:
                    continue
                covered.extend(json.loads((directory / 'completion.json').read_text())['task_ids'])
                batches.extend(json.loads((directory / 'batches.json').read_text()))
                elapsed.append(json.loads((directory / 'elapsed.json').read_text())['seconds'])
            assert len(covered) == len(set(covered)) and set(covered) <= expected
            complete = set(covered) == expected and all(not row['missing_telemetry'] for row in sessions)
            result.update(sessions=sessions, timing_complete=complete, timing_covered_task_ids=covered,
                          observed_session_seconds=sum(elapsed), session_seconds=sum(elapsed) if complete else None,
                          batch_shapes=dict(Counter(batch['batch_size'] for batch in batches)), decode_itl={})
            result['batches_without_decode_telemetry'] = sum('decode' not in batch for batch in batches)
            for size in sorted({batch['batch_size'] for batch in batches}):
                intervals = [value for batch in batches if batch['batch_size'] == size and 'decode' in batch
                             for row in batch['decode'] for value in row['inter_token_seconds']]
                metric = {'observed_row_intervals': len(intervals)}
                if intervals:
                    metric.update(mean_itl_seconds=mean(intervals), median_itl_seconds=median(intervals))
                result['decode_itl'][str(size)] = metric
    assert len(configurations) == 1
    configuration, = configurations
    return {'task_ids': manifest['task_ids'], 'shared_config': json.loads(configuration), 'methods': methods,
            'evaluated_methods': sum(row['evaluated'] for row in methods.values()),
            'scope': 'Same declared qualification tasks. Official quality retains failed-task denominators. '
                     'Session timing includes tools and queue waits, excludes model loading. ITL pools observed '
                     'active-row token intervals by actual GPU batch size; interrupted coverage is explicit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = collect(json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'evaluated_methods': report['evaluated_methods']}))
