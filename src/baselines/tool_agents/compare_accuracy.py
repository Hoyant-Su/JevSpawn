import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


def jsonl(path):
    return list(map(json.loads, path.read_text().splitlines()))


def agent_predictions(path, task_ids, repeat):
    metadata = json.loads((path / 'run.json').read_text())
    rows = jsonl(path / 'predictions.jsonl')
    assert len(rows) == len(task_ids) * metadata['repeats']
    assert {(row['repeat'], row['task_id']) for row in rows} == {
        (trial, task_id) for trial in range(metadata['repeats']) for task_id in task_ids}
    selected = [row for row in rows if row['repeat'] == repeat]
    return {row['task_id']: row['choice'] if row['status'] == 'complete' else None for row in selected}


def native_predictions(path, method, task_ids, repeat):
    metadata = json.loads((path / 'run.json').read_text())
    records = [row for row in jsonl(path / 'trials.jsonl') if row['method'] == method]
    assert len(records) == len(metadata['batch_task_ids']) * metadata['repeats']
    predictions = {}
    for row in records:
        if row['repeat'] == repeat:
            field, = row['result']['fields'].values()
            for task_id, choice in zip(row['task_ids'], field['choices'], strict=True):
                assert task_id not in predictions
                predictions[task_id] = choice
    assert predictions.keys() == task_ids
    return predictions


def paired_bootstrap(scores, names, samples, seed, confidence):
    indices = np.random.default_rng(seed).integers(scores.shape[1], size=(samples, scores.shape[1]))
    means = scores[:, indices].mean(axis=2)
    limits = [(1 - confidence) / 2, (1 + confidence) / 2]
    accuracies = {name: {'correct': int(scores[index].sum()), 'questions': scores.shape[1],
                         'accuracy': float(scores[index].mean()),
                         'interval': np.quantile(means[index], limits).tolist()}
                  for index, name in enumerate(names)}
    differences = []
    for left, right in combinations(range(len(names)), 2):
        differences.append({'left': names[left], 'right': names[right],
                            'difference': float(scores[left].mean() - scores[right].mean()),
                            'interval': np.quantile(means[left] - means[right], limits).tolist()})
    return accuracies, differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = args.config.resolve().parent
    labels = jsonl(directory / config['labels'])
    task_ids = {row['task_id'] for row in labels}
    assert len(task_ids) == len(labels)
    truth, question_ids = {}, {}
    for row in labels:
        truth[row['task_id']], = row['labels'].values()
        question_ids[row['task_id']], = row['question_ids'].values()
    assert len(set(question_ids.values())) == len(labels)
    ordered = sorted(task_ids, key=question_ids.__getitem__)
    predictions = {}
    for name, path in config['agents'].items():
        predictions[name] = agent_predictions(directory / path, task_ids, config['accuracy_repeat'])
    for name, source in config['native'].items():
        predictions[name] = native_predictions(directory / source['run'], source['method'], task_ids, config['accuracy_repeat'])
    assert all(values.keys() == task_ids for values in predictions.values())
    names = list(predictions)
    scores = np.array([[predictions[name][task_id] == truth[task_id] for task_id in ordered] for name in names])
    accuracies, differences = paired_bootstrap(scores, names, config['bootstrap_samples'],
                                              config['seed'], config['confidence_level'])
    result = {
        'questions': len(ordered), 'question_ids': [question_ids[task_id] for task_id in ordered],
        'accuracy_repeat': config['accuracy_repeat'],
        'bootstrap_samples': config['bootstrap_samples'], 'seed': config['seed'],
        'confidence_level': config['confidence_level'],
        'resampling_unit': 'Original AQuA question; the same resampled question indices are used for every method.',
        'interval_method': 'Paired nonparametric percentile bootstrap. Repeated timing executions are not independent question observations.',
        'failed_answer_policy': 'Truncation, invalid schema and missing final choices count as incorrect.',
        'accuracy': accuracies, 'paired_differences': differences,
    }
    (directory / config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'accuracy': accuracies, 'paired_differences': differences}, indent=2))


if __name__ == '__main__':
    main()
