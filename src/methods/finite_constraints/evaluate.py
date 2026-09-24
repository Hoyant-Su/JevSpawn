import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from data.evaluate_zebralogic import score_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    assert (args.run / 'completion.json').is_file()
    completed = json.loads((args.run / 'completion.json').read_text())
    settings = json.loads((args.run / 'protocol.json').read_text())['settings']
    labels = {row['task_id']: row for row in map(json.loads, Path(settings['labels']).read_text().splitlines())}
    tasks = {row['task_id']: row for row in map(json.loads, Path(settings['tasks']).read_text().splitlines())}
    report = {'scope': 'Development only. Original labels first opened after every assigned phase and method completed.',
              'phases': {}}
    for phase in settings['phases']:
        records = json.loads((args.run / completed['phase_directories'][phase] / 'summary.json').read_text())
        assert len(records) == settings['task_count'] * len(settings['methods'])
        methods = {}
        for method in settings['methods']:
            rows = [record for record in records if record['method'] == method]
            assert {row['task_id'] for row in rows} == set(tasks)
            scores = [{'task_id': row['task_id'], **score_rows(
                tasks[row['task_id']]['puzzle'], labels[row['task_id']]['solution'], row['prediction_rows'])}
                for row in rows]
            methods[method] = {
                'tasks': len(rows), 'statuses': dict(Counter(row['status'] for row in rows)),
                'solved': sum(score['solved'] for score in scores),
                'puzzle_accuracy': sum(score['solved'] for score in scores) / len(rows),
                'cell_accuracy': sum(score['correct_cells'] for score in scores) / sum(score['total_cells'] for score in scores),
                'mean_seconds': statistics.mean(row['total_seconds'] for row in rows),
                'mean_peak_allocated_bytes': statistics.mean(row['peak_allocated_bytes'] for row in rows),
                'max_itl_ms': max(row['max_itl_ms'] for row in rows),
                'intervals_over_100ms': sum(row['intervals_over_100ms'] for row in rows),
                'finite_workers_per_task': {row['task_id']: row.get('finite_workers', 0) for row in rows},
                'scores': scores}
        report['phases'][phase] = methods
    (args.run / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['phases']['measured'], indent=2))


if __name__ == '__main__':
    main()
