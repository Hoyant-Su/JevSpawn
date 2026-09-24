import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def interval(samples, confidence):
    tail = (1 - confidence) / 2
    return np.quantile(samples, [tail, 1 - tail]).tolist()


def paired_quality(directory, methods, reference, quality_repeat, replicates, seed, confidence):
    outcomes, task_ids = {}, None
    for method in methods:
        path = directory / f'repeat-{quality_repeat:02d}-{method}' / 'evaluation.jsonl'
        records = [json.loads(line) for line in path.read_text().splitlines()]
        ids = [row['task_id'] for row in records]
        assert len(ids) == len(set(ids))
        if task_ids is None:
            task_ids = ids
        assert ids == task_ids, 'Quality comparisons require identical ordered task IDs.'
        assert all(row['status'] in {'passed', 'failed', 'timeout'} for row in records)
        outcomes[method] = np.array([row['status'] == 'passed' for row in records], dtype=np.int8)
    indices = np.random.default_rng(seed).integers(len(task_ids), size=(replicates, len(task_ids)))
    estimates = {method: {
        'correct': int(values.sum()), 'accuracy_percent': float(values.mean() * 100),
        'accuracy_ci_percent': interval(values[indices].mean(axis=1) * 100, confidence),
    } for method, values in outcomes.items()}
    for method, values in outcomes.items():
        difference = values - outcomes[reference]
        estimates[method]['difference_percentage_points'] = float(difference.mean() * 100)
        estimates[method]['difference_ci_percentage_points'] = interval(
            difference[indices].mean(axis=1) * 100, confidence)
    comparisons = []
    for left, right in combinations(methods, 2):
        difference = outcomes[right] - outcomes[left]
        comparisons.append({
            'reference': left, 'method': right,
            'difference_percentage_points': float(difference.mean() * 100),
            'difference_ci_percentage_points': interval(difference[indices].mean(axis=1) * 100, confidence),
            'improved_tasks': int((difference == 1).sum()),
            'regressed_tasks': int((difference == -1).sum()),
        })
    return {'tasks': len(task_ids), 'quality_reference': reference,
            'quality_repeat': quality_repeat, 'bootstrap_replicates': replicates,
            'bootstrap_seed': seed, 'confidence': confidence, 'methods': estimates,
            'paired_comparisons': comparisons,
            'interpretation': 'Percentile intervals from paired task resampling. Timing repeats are excluded from the quality sample size. These intervals describe task-level uncertainty and are not multiplicity-adjusted significance tests.'}


def latex_text(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def comparison(stats, reference, method):
    if method == reference:
        return 0.0, [0.0, 0.0]
    for row in stats['paired_comparisons']:
        if (row['reference'], row['method']) == (reference, method):
            return row['difference_percentage_points'], row['difference_ci_percentage_points']
        if (row['reference'], row['method']) == (method, reference):
            lo, hi = row['difference_ci_percentage_points']
            return -row['difference_percentage_points'], [-hi, -lo]
    raise ValueError('Missing paired quality comparison.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='append', required=True, help='Dataset name=completed run directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--quality-repeat', type=int, required=True)
    parser.add_argument('--quality-reference', required=True)
    parser.add_argument('--fixed-reference', required=True)
    parser.add_argument('--bootstrap-replicates', type=int, required=True)
    parser.add_argument('--bootstrap-seed', type=int, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    args = parser.parse_args()
    assert args.bootstrap_replicates > 0 and 0 < args.confidence < 1
    rows, quality = [], {}
    for entry in args.run:
        name, directory = entry.split('=', 1)
        directory = Path(directory)
        summary = json.loads((directory / 'summary.json').read_text())
        methods = list(summary['methods'])
        assert args.quality_reference in methods
        stats = paired_quality(directory, methods, args.quality_reference, args.quality_repeat, args.bootstrap_replicates,
                               args.bootstrap_seed, args.confidence)
        assert stats['tasks'] == summary['tasks']
        quality[name] = stats
        for method, timing in summary['methods'].items():
            metric = stats['methods'][method]
            fixed_difference, fixed_ci = comparison(stats, args.fixed_reference, method)
            runs = timing['repeats']
            rows.append({
                'dataset': name, 'method': method, 'tasks': stats['tasks'],
                'correct': metric['correct'], 'accuracy_percent': metric['accuracy_percent'],
                'accuracy_ci_low': metric['accuracy_ci_percent'][0],
                'accuracy_ci_high': metric['accuracy_ci_percent'][1],
                'quality_reference': args.quality_reference,
                'difference_percentage_points': metric['difference_percentage_points'],
                'difference_ci_low': metric['difference_ci_percentage_points'][0],
                'difference_ci_high': metric['difference_ci_percentage_points'][1],
                'fixed_reference': args.fixed_reference,
                'difference_vs_fixed_percentage_points': fixed_difference,
                'difference_vs_fixed_ci_low': fixed_ci[0],
                'difference_vs_fixed_ci_high': fixed_ci[1],
                'seconds_mean': timing['elapsed_mean_seconds'],
                'seconds_sd': timing['elapsed_stdev_seconds'],
                'workers_mean': float(np.mean([r['worker_calls'] for r in runs])),
                'controller_decisions_mean': float(np.mean([r['controller_decisions'] for r in runs])),
                'input_tokens_mean': float(np.mean([r['input_tokens'] for r in runs])),
                'output_tokens_mean': float(np.mean([r['output_tokens'] for r in runs])),
                'peak_allocated_gib': max(r['peak_allocated_gib'] for r in runs),
                'itl_p95_ms': max(r['itl_p95_ms'] for r in runs),
                'itl_max_ms': timing['max_itl_ms'],
                'latency_qualified': timing['latency_qualified'],
                'identical_initial_drafts': timing['identical_initial_drafts'],
                'timing_repeats': summary['timing_repeats'],
                'passed_each_repeat': timing['passed_each_repeat'],
            })
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'T7_coding.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'T7_coding_quality.json').write_text(json.dumps(quality, indent=2) + '\n')
    table = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\resizebox{\linewidth}{!}{%',
             r'\begin{tabular}{llrrrrrrrr}', r'\hline',
             r'Dataset & Method & Pass (\%) & $\Delta$ vs single [CI] & $\Delta$ vs fixed [CI] & Time (s) & Workers & Decisions & Peak GiB & ITL p95 (ms) \\',
             r'\hline']
    for row in rows:
        deviation = row['seconds_sd']
        runtime = f"{row['seconds_mean']:.2f}"
        if deviation is not None:
            runtime += f" $\\pm$ {deviation:.2f}"
        table.append(' & '.join([
            latex_text(row['dataset']), latex_text(row['method']),
            f"{row['accuracy_percent']:.2f}",
            f"{row['difference_percentage_points']:+.2f} [{row['difference_ci_low']:.2f}, {row['difference_ci_high']:.2f}]",
            f"{row['difference_vs_fixed_percentage_points']:+.2f} [{row['difference_vs_fixed_ci_low']:.2f}, {row['difference_vs_fixed_ci_high']:.2f}]", runtime,
            f"{row['workers_mean']:.1f}", f"{row['controller_decisions_mean']:.1f}",
            f"{row['peak_allocated_gib']:.2f}", f"{row['itl_p95_ms']:.2f}",
        ]) + r' \\')
    table.extend([r'\hline', r'\end{tabular}', r'}',
                  r'\caption{Complete coding workflows with matched frozen models and batch sizes. '
                  r'Time reports the mean and sample standard deviation across complete workload repetitions. '
                  r'Accuracy uses the designated repetition, with each task counted once. Full workload warmup and evaluation time are excluded. '
                  r'Peak memory and ITL p95 report the largest value across measured repetitions. '
                  r'Worker counts differ because adaptive controllers select additional review and repair calls. '
                  f"Paired differences in percentage points relative to {latex_text(args.quality_reference)} and {latex_text(args.fixed_reference)} use "
                  f"{args.confidence * 100:g} percent confidence intervals from {args.bootstrap_replicates} task bootstrap draws with seed {args.bootstrap_seed}.}}",
                  r'\label{tab:coding-workflows}', r'\end{table*}'])
    (args.output / 'T7_coding.tex').write_text('\n'.join(table) + '\n')
    print(json.dumps({'rows': len(rows), 'datasets': list(quality), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
