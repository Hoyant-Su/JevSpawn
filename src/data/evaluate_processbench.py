import argparse
import json
from pathlib import Path


def metrics(predictions, labels, formulation):
    first_errors, local_matches = {}, []
    for task_id, row in labels.items():
        choices = predictions[task_id]
        assert choices.keys() == row['labels'].keys(), task_id
        if formulation == 'per_step':
            assert set(choices.values()) <= {'correct', 'incorrect'}, task_id
            errors = [index for index in range(row['step_count']) if choices[f'q{index}'] == 'incorrect']
            first_errors[task_id] = min(errors) if errors else -1
            local_matches.extend(choices[name] == answer for name, answer in row['labels'].items() if answer is not None)
        else:
            prediction = int(choices['q0'])
            assert -1 <= prediction < row['step_count'], task_id
            first_errors[task_id] = prediction
    matches = {task_id: prediction == labels[task_id]['first_error'] for task_id, prediction in first_errors.items()}
    errors = [match for task_id, match in matches.items() if labels[task_id]['first_error'] != -1]
    correct = [match for task_id, match in matches.items() if labels[task_id]['first_error'] == -1]
    error_accuracy = sum(errors) / len(errors) if errors else None
    correct_accuracy = sum(correct) / len(correct) if correct else None
    f1 = None
    if errors and correct and error_accuracy + correct_accuracy > 0:
        f1 = 2 * error_accuracy * correct_accuracy / (error_accuracy + correct_accuracy)
    return {
        'solutions': len(matches), 'error_solutions': len(errors), 'correct_solutions': len(correct),
        'decisions': sum(len(choices) for choices in predictions.values()),
        'first_error_accuracy': sum(matches.values()) / len(matches),
        'error_accuracy': error_accuracy, 'correct_accuracy': correct_accuracy, 'processbench_f1': f1,
        'labelled_steps': len(local_matches),
        'local_step_accuracy': sum(local_matches) / len(local_matches) if local_matches else None,
        'predicted_first_errors': first_errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--formulation', choices=['per_step', 'first_error'], required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / 'run.json').read_text())
    identities = {task_id for batch in metadata['batch_task_ids'] for task_id in batch}
    all_labels = {row['task_id']: row for row in map(json.loads, args.labels.read_text().splitlines())}
    labels = {task_id: all_labels[task_id] for task_id in sorted(identities)}
    records = list(map(json.loads, (args.run / 'trials.jsonl').read_text().splitlines()))
    result = {'formulation': args.formulation, 'metric_scale': 'fraction', 'methods': {},
              'undefined_metrics': 'Null denotes an absent subgroup or a zero denominator.'}
    for method in metadata['methods']:
        predictions = {task_id: {} for task_id in labels}
        trials = [row for row in records if row['method'] == method and row['repeat'] == 0]
        assert len(trials) == len(metadata['batch_task_ids'])
        for row in trials:
            for name, field in row['result']['fields'].items():
                for task_id, choice in zip(row['task_ids'], field['choices'], strict=True):
                    predictions[task_id][name] = choice
        result['methods'][method] = metrics(predictions, labels, args.formulation)
    (args.run / 'processbench_metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
