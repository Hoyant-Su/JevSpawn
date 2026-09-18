"""Score frozen answerability decisions without changing prompts or fitting thresholds."""

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def divide(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def metrics(rows):
    positive = sum(row['answerable'] for row in rows)
    negative = len(rows) - positive
    assert positive and negative, 'Balanced accuracy requires both gold classes.'
    tp = sum(row['choice'] == 'yes' and row['answerable'] for row in rows)
    tn = sum(row['choice'] == 'no' and not row['answerable'] for row in rows)
    fp = sum(row['choice'] == 'yes' and not row['answerable'] for row in rows)
    fn = sum(row['choice'] == 'no' and row['answerable'] for row in rows)
    invalid_yes = sum(row['choice'] not in ('yes', 'no') and row['answerable'] for row in rows)
    invalid_no = sum(row['choice'] not in ('yes', 'no') and not row['answerable'] for row in rows)
    yes_f1 = divide(2 * tp, 2 * tp + fp + fn + invalid_yes)
    no_f1 = divide(2 * tn, 2 * tn + fp + fn + invalid_no)
    paragraphs = {}
    for row in rows:
        paragraphs.setdefault(row['task_id'], []).append(row['choice'] == ('yes' if row['answerable'] else 'no'))
    probabilities = [row for row in rows if row['probability_yes'] is not None]
    assert len(probabilities) in (0, len(rows)), 'Probability coverage must be complete or explicitly absent.'
    return {
        'paragraphs': len(paragraphs), 'questions': len(rows), 'correct': tp + tn,
        'support': {'yes': positive, 'no': negative},
        'question_accuracy': (tp + tn) / len(rows),
        'balanced_accuracy': (tp / positive + tn / negative) / 2,
        'macro_f1': (yes_f1 + no_f1) / 2,
        'yes_precision': divide(tp, tp + fp), 'yes_recall': tp / positive, 'yes_f1': yes_f1,
        'no_precision': divide(tn, tn + fn), 'no_recall': tn / negative, 'no_f1': no_f1,
        'brier_score': sum((row['probability_yes'] - row['answerable']) ** 2 for row in probabilities) / len(probabilities) if probabilities else None,
        'probability_count': len(probabilities),
        'paragraph_all_correct': sum(all(values) for values in paragraphs.values()) / len(paragraphs),
        'paragraphs_all_correct_count': sum(all(values) for values in paragraphs.values()),
        'invalid_count': invalid_yes + invalid_no, 'invalid_output_rate': (invalid_yes + invalid_no) / len(rows),
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'invalid_gold_yes': invalid_yes, 'invalid_gold_no': invalid_no},
    }


def benchmark_predictions(run, metadata):
    assert isinstance(metadata['fields'], list), 'Answerability benchmark requires task-specific field schemas.'
    ids = metadata['task_ids']
    assert len(ids) == len(metadata['fields']) == metadata['batch_size'] * metadata['batch_count']
    schemas = dict(zip(ids, metadata['fields']))
    trials = read_jsonl(run / 'trials.jsonl')
    selected = {(row['method'], row['batch']): row for row in trials if row['sweep'] == 'latency' and row['repeat'] == 0}
    expected = {(method, batch) for method in metadata['methods'] for batch in range(metadata['batch_count'])}
    assert set(selected) == expected, 'Missing or unexpected measured method/batch records.'
    predictions = {}
    for method in metadata['methods']:
        predictions[method] = {}
        for batch in range(metadata['batch_count']):
            result = selected[(method, batch)]['result']
            assert result['batch_size'] == metadata['batch_size']
            for field in result['fields'].values():
                assert len(field['choices']) == len(field['probabilities']) == result['batch_size']
            for index in range(result['batch_size']):
                task_id = ids[batch * metadata['batch_size'] + index]
                predictions[method][task_id] = {
                    name: {'choice': field['choices'][index], 'probabilities': field['probabilities'][index], 'option_ids': field['option_ids']}
                    for name, field in result['fields'].items()
                }
    return schemas, predictions


def worker_predictions(run, metadata):
    schemas = metadata['task_fields']
    assert schemas is not None, 'Answerability workers require task-specific field schemas.'
    summary = json.loads((run / 'summary.json').read_text())
    assert summary['method'] == metadata['method']
    rows = read_jsonl(run / 'predictions.jsonl')
    predictions = {row['task_id']: row['fields'] for row in rows}
    assert len(rows) == len(predictions), 'Duplicate worker task IDs.'
    return schemas, {summary['method']: predictions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--format', choices=['benchmark', 'workers'], required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / 'run.json').read_text())
    json.loads((args.run / 'summary.json').read_text())
    label_rows = read_jsonl(args.labels)
    labels = {row['task_id']: row for row in label_rows}
    assert len(labels) == len(label_rows), 'Duplicate gold task IDs.'
    ids = metadata['task_ids']
    assert ids and len(ids) == len(set(ids)) and set(ids) <= labels.keys()
    loaders = {'benchmark': benchmark_predictions, 'workers': worker_predictions}
    schemas, predictions = loaders[args.format](args.run, metadata)
    assert schemas.keys() == set(ids)
    for task_id in ids:
        target = labels[task_id]
        assert schemas[task_id].keys() == target['labels'].keys() == target['question_ids'].keys()
        assert all(type(value) is bool for value in target['labels'].values())
    questions = [question_id for task_id in ids for question_id in labels[task_id]['question_ids'].values()]
    assert len(questions) == len(set(questions)), 'Duplicate official question IDs.'
    evaluated, methods = [], {}
    for method, tasks in predictions.items():
        assert tasks.keys() == set(ids), 'Prediction IDs must exactly match the run task selection.'
        rows = []
        for task_id in ids:
            assert tasks[task_id].keys() == schemas[task_id].keys(), 'Missing or extra predicted question fields.'
            for name, field in tasks[task_id].items():
                options = [option['id'] for option in schemas[task_id][name]['options']]
                assert set(options) == {'yes', 'no'} and len(options) == 2
                assert field['option_ids'] == options
                probabilities = field['probabilities']
                probability_yes = None
                if probabilities is not None:
                    assert len(probabilities) == len(options)
                    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
                    probability_yes = probabilities[options.index('yes')]
                answerable = labels[task_id]['labels'][name]
                rows.append({'method': method, 'task_id': task_id, 'field_id': name,
                             'question_id': labels[task_id]['question_ids'][name],
                             'answerable': answerable, 'choice': field['choice'], 'probability_yes': probability_yes,
                             'correct': field['choice'] == ('yes' if answerable else 'no')})
        methods[method] = metrics(rows)
        evaluated.extend(rows)
    targets = [{'task_id': task_id, 'answerable': answerable, 'probability_yes': None}
               for task_id in ids for answerable in labels[task_id]['labels'].values()]
    baselines = {f'all_{choice}': metrics([{**row, 'choice': choice} for row in targets]) for choice in ['yes', 'no']}
    summary = {
        'format': args.format, 'paragraphs': len(ids), 'questions': len(questions), 'methods': methods, 'baselines': baselines,
        'scope': 'SQuAD2.0 binary answerability only; not answer-span QA EM/F1. Benchmark format uses latency repeat zero, retaining the last complete record for each method/batch key. Worker format uses all frozen collected predictions.',
        'metric_conventions': 'Each original question has equal weight. Balanced accuracy averages yes/no recall. Macro F1 averages the two class F1 values. Undefined precision or F1 is zero. Invalid choices count as incorrect and as missed gold-class predictions, never as a default yes/no answer. Brier uses raw yes probabilities only when every question supplies them; no calibration or thresholds are fitted.',
    }
    (args.run / 'quality.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.run / 'answerability_predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in evaluated))
    print(json.dumps(summary['methods'], indent=2))


if __name__ == '__main__':
    main()
