import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.run / 'measured-results.json').read_text())
    labels = {row['task_id']: row['labels']['q0']
              for row in map(json.loads, args.labels.read_text().splitlines())}
    completed = [row for row in rows if row['status'] == 'valid']
    correct = sum(row['answer'] == labels[row['task_id']] for row in completed)
    report = {'tasks': len(rows), 'valid_answers': len(completed), 'correct': correct,
              'task_success_rate': correct / len(rows),
              'accuracy_conditional_on_valid_answer': correct / len(completed) if completed else None,
              'status_counts': dict(Counter(row['status'] for row in rows)),
              'task_results': [{'task_id': row['task_id'], 'status': row['status'],
                                'correct': row['status'] == 'valid' and row['answer'] == labels[row['task_id']]}
                               for row in rows]}
    (args.run / 'quality.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'task_results'}))


if __name__ == '__main__':
    main()
