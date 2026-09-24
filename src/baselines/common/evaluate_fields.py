import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from baselines.common.tasks import read, rows
from baselines.formal_choices.run import save
from baselines.common.resources import ADAPTER_SETTINGS


def evaluate(directory, labels_path):
    contract = read(directory / 'protocol.json')
    tasks = contract['tasks']
    assert all(task['kind'] == 'fields' for task in tasks)
    assert read(directory / 'completion.json')['tasks'] == len(tasks)
    results = [read(directory / f'task-{index:05d}.json') for index in range(len(tasks))]
    assert [row['task_id'] for row in results] == [row['task_id'] for row in tasks]
    labels = {row['task_id']: row['labels'] for row in rows(labels_path)}
    decisions = []
    for task, result in zip(tasks, results):
        gold = labels[task['task_id']]
        assert set(gold) == set(task['source']['fields'])
        for name, answer in gold.items():
            prediction = result['answer'][name] if result['status'] == 'completed' and result['answer'] is not None else None
            decisions.append({'task_id': task['task_id'], 'field': name, 'predicted': prediction,
                              'expected': answer, 'correct': prediction == answer})
    batches = [row for path in sorted(directory.glob('session-*/batches.json')) for row in read(path)]
    intervals = [value for batch in batches for item in batch['decode'] for value in item['inter_token_seconds']]
    whole = sum(read(path)['elapsed_seconds'] for path in directory.glob('session-*/completion.json'))
    summary = {'stage': contract['specification']['stage'], 'metric': 'field_accuracy',
               'tasks': len(tasks), 'decisions': len(decisions),
               'correct': sum(row['correct'] for row in decisions),
               'accuracy': sum(row['correct'] for row in decisions) / len(decisions),
               'status_counts': dict(Counter(row['status'] for row in results)),
               'complete_elapsed_seconds': whole,
               'mean_sample_seconds': statistics.mean(row['elapsed_seconds'] for row in results),
               'actual_batch_sizes': dict(Counter(batch['batch_size'] for batch in batches)),
               'model_sequences': sum(batch['batch_size'] for batch in batches),
               'generated_tokens': sum(sum(batch['output_tokens']) for batch in batches),
               'median_itl_ms': 1000 * statistics.median(intervals) if intervals else None,
               'maximum_itl_ms': 1000 * max(intervals) if intervals else None,
               ADAPTER_SETTINGS['evaluation_fields']['slow_interval_metric']: sum(
                   value >= ADAPTER_SETTINGS['evaluation_fields']['slow_interval_seconds'] for value in intervals),
               'graph_capture_seconds': sum(batch['graph_capture_seconds'] for batch in batches),
               'peak_allocated_bytes': max(batch['peak_allocated_bytes'] for batch in batches),
               'per_field': decisions}
    save(directory / 'evaluation.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, args.labels), indent=2))


if __name__ == '__main__':
    main()
