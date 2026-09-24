"""Clustered quality intervals and measured runtime summaries for native runs."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def predictions(records, method, repeat=0):
    result = {}
    for record in records:
        if record['method'] != method or record['repeat'] != repeat:
            continue
        for name, field in record['result']['fields'].items():
            for index, task in enumerate(record['task_ids']):
                result[(task, name)] = {
                    'choice': field['choices'][index],
                    'probabilities': field.get('probabilities', [None] * len(record['task_ids']))[index],
                    'options': field.get('option_ids'),
                }
    return result


def interval(values):
    return np.quantile(values, [.025, .975]).tolist()


def probability_scores(values, gold, bins):
    confidence, hit, brier = [], [], []
    for key, value in values.items():
        if key not in gold or value['probabilities'] is None:
            continue
        probs = np.array(value['probabilities'])
        target = np.array([option == gold[key] for option in value['options']])
        confidence.append(float(probs.max()))
        hit.append(value['choice'] == gold[key])
        brier.append(float(np.square(probs - target).sum()))
    if not confidence:
        return {'brier': None, 'ece': None}
    confidence, hit = np.array(confidence), np.array(hit)
    bin_id = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = sum(float(np.mean(bin_id == b)) *
              abs(float(confidence[bin_id == b].mean() - hit[bin_id == b].mean()))
              for b in range(bins) if np.any(bin_id == b))
    return {'brier': float(np.mean(brier)), 'ece': ece}


def analyze(directory, label_path, samples, seed, bins, cluster_path=None):
    records = read(directory / 'trials.jsonl')
    meta = json.loads((directory / 'run.json').read_text())
    required = {(repeat, batch, method) for repeat in range(meta['repeats'])
                for batch in range(len(meta['batch_task_ids'])) for method in meta['methods']}
    assert {(r['repeat'], r['batch'], r['method']) for r in records} == required
    gold = {(r['task_id'], key): value for r in read(label_path)
            for key, value in r['labels'].items() if value is not None}
    task_ids = [task for batch in meta['batch_task_ids'] for task in batch]
    keys = {task: [key for key in gold if key[0] == task] for task in task_ids}
    counts = np.array([len(keys[task]) for task in task_ids])
    cluster_map = (json.loads(cluster_path.read_text()) if cluster_path is not None
                   else {'unit': 'Original task/paragraph', 'clusters': {task: task for task in task_ids}})
    groups = defaultdict(list)
    for index, task in enumerate(task_ids):
        groups[cluster_map['clusters'][task]].append(index)
    members = list(groups.values())
    unit_counts = np.array([counts[group].sum() for group in members])
    draws = np.random.default_rng(seed).integers(len(members), size=(samples, len(members)))
    denominator = unit_counts[draws].sum(axis=1)
    estimates = {method: predictions(records, method) for method in meta['methods']}
    correct = {method: np.array([sum(estimates[method][key]['choice'] == gold[key]
                                    for key in keys[task]) for task in task_ids])
               for method in meta['methods']}
    cluster_correct = {method: np.array([values[group].sum() for group in members])
                       for method, values in correct.items()}
    results = {}
    for method in meta['methods']:
        rows = [r for r in records if r['method'] == method]
        times = defaultdict(float)
        for r in rows:
            times[r['repeat']] += r['elapsed_seconds']
        dt = np.array([value * 1000 for r in rows for item in r['decode']
                       for value in item['inter_token_seconds']])
        ttft = [item['ttft_seconds'] * 1000 for r in rows for item in r['decode']
                if item['ttft_seconds'] is not None]
        p = estimates[method]
        classes = sorted(set(gold[key] for task in task_ids for key in keys[task]))
        used = [key for task in task_ids for key in keys[task]]
        f1 = []
        for label in classes:
            tp = sum(gold[key] == label and p[key]['choice'] == label for key in used)
            n = sum(gold[key] == label for key in used) + sum(p[key]['choice'] == label for key in used)
            f1.append(2 * tp / n)
        results[method] = {
            'tasks': len(task_ids), 'decisions': int(counts.sum()),
            'accuracy': float(correct[method].sum() / counts.sum()),
            'accuracy_ci95': interval(cluster_correct[method][draws].sum(axis=1) / denominator),
            'macro_f1': float(np.mean(f1)),
            'invalid': sum(p[key]['choice'] not in classes for key in used),
            'warm_seconds': list(times.values()), 'median_warm_seconds': float(np.median(list(times.values()))),
            'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in rows) / 2**30,
            'peak_reserved_gib': max(r['peak_reserved_bytes'] for r in rows) / 2**30,
            'itl_ms': {'p50': float(np.median(dt)), 'p95': float(np.quantile(dt, .95)),
                       'max': float(dt.max()), 'intervals_ge100ms': int(np.sum(dt >= 100))} if len(dt) else None,
            'median_ttft_ms': float(np.median(ttft)) if ttft else None,
            **probability_scores(p, gold, bins),
            'repeat_choice_changes': {str(repeat): sum(p[key]['choice'] != value['choice']
                for key, value in predictions(records, method, repeat).items())
                for repeat in range(1, meta['repeats'])},
        }
    pairs = {}
    for reference in meta['methods']:
        for method in meta['methods']:
            if reference == method:
                continue
            delta = correct[method] - correct[reference]
            cluster_delta = cluster_correct[method] - cluster_correct[reference]
            pairs[f'{method}-minus-{reference}'] = {
                'accuracy_difference': float(delta.sum() / counts.sum()),
                'accuracy_difference_ci95': interval(cluster_delta[draws].sum(axis=1) / denominator),
                'prediction_disagreements': sum(estimates[method][key]['choice'] != estimates[reference][key]['choice']
                                                for task in task_ids for key in keys[task]),
            }
    return {'run': str(directory), 'labels': str(label_path), 'methods': results, 'pairs': pairs,
            'timing_scope': meta['timing_scope'],
            'uncertainty': {'unit': cluster_map['unit'], 'clusters': len(members), 'bootstrap_samples': samples, 'seed': seed,
                            'interval': 'Paired percentile bootstrap, pointwise 95%; no independence assumed between fields.',
                            'runtime': f"{meta['repeats']} whole-workload values; no per-batch pseudo-replicated confidence interval."},
            'calibration': {'bins': bins, 'scores': 'Restricted categorical softmax; uncalibrated.',
                            'brier': 'Sum of squared probability error across all candidate classes.'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--bins', type=int, required=True)
    parser.add_argument('--clusters', type=Path)
    args = parser.parse_args()
    result = analyze(args.run, args.labels, args.samples, args.seed, args.bins, args.clusters)
    (args.run / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({method: {key: row[key] for key in ['accuracy', 'accuracy_ci95', 'itl_ms']}
                      for method, row in result['methods'].items()}))


if __name__ == '__main__':
    main()
