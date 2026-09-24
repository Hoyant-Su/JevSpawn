"""Evaluate saved categorical predictions after inference."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    labels = {row['task_id']: row['labels'] for row in
              map(json.loads, args.labels.read_text().splitlines())}
    records = [json.loads(line) for line in (args.run / 'trials.jsonl').read_text().splitlines()]
    metadata = json.loads((args.run / 'run.json').read_text())
    required = {(repeat, batch, method) for repeat in range(metadata['repeats'])
                for batch in range(len(metadata['batch_task_ids'])) for method in metadata['methods']}
    assert {(row['repeat'], row['batch'], row['method']) for row in records} == required
    predictions, summary = {}, {}
    for method in metadata['methods']:
        selected = [row for row in records if row['method'] == method]
        times = defaultdict(float)
        estimates = {}
        for row in selected:
            times[row['repeat']] += row['elapsed_seconds']
            if row['repeat'] == 0:
                for name, field in row['result']['fields'].items():
                    for task_id, choice in zip(row['task_ids'], field['choices']):
                        estimates[(task_id, name)] = choice
        expected = {(task_id, name) for batch in metadata['batch_task_ids']
                    for task_id in batch for name in labels[task_id]}
        assert estimates.keys() == expected
        truth = {key: labels[key[0]][key[1]] for key in expected}
        support = Counter(truth.values())
        f1 = []
        for label in support:
            tp = sum(truth[key] == label and estimates[key] == label for key in expected)
            predicted = sum(value == label for value in estimates.values())
            f1.append(2 * tp / (support[label] + predicted))
        summary[method] = {
            'decisions': len(expected),
            'correct': sum(estimates[key] == truth[key] for key in expected),
            'accuracy': sum(estimates[key] == truth[key] for key in expected) / len(expected),
            'macro_f1': statistics.mean(f1), 'label_counts': dict(support),
            'seconds_per_repeat': dict(times),
            'median_seconds': statistics.median(times.values()),
            'peak_allocated_gib': max(row['peak_allocated_bytes'] for row in selected) / 2**30,
            'max_input_tokens': max(max(field['input_tokens']) for row in selected
                                    for field in row['result']['fields'].values()),
        }
        predictions[method] = estimates
    pairs = {}
    for method in metadata['methods']:
        reference = predictions[metadata['methods'][0]]
        pairs[method] = sum(reference[key] != predictions[method][key] for key in reference)
    result = {'methods': summary, 'disagreements_from_first_method': pairs,
              'timing_scope': metadata['timing_scope']}
    (args.run / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
