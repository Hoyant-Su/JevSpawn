"""Build manuscript tables from completed experiment artifacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_native import predictions, probability_scores, read


DATASETS = {'aqua': 'AQuA', 'race_middle': 'RACE-middle', 'pubmedqa': 'PubMedQA',
            'scifact_cited': 'SciFact cited', 'squad2_answerability': 'SQuAD2 answerability'}
METHODS = {'independent': 'Independent', 'shared': 'Shared cache',
           'streamed': 'Streamed', 'compact_json': 'Compact JSON'}


def analysis(directory):
    return json.loads((directory / 'analysis.json').read_text())


def main_run(root, dataset):
    return root / 'runs' / ('squad-main-001' if dataset == 'squad2_answerability'
                           else 'native-main-001/' + dataset)


def write_table(root, name, headers, rows):
    output = root / 'manuscript/tables'
    output.mkdir(parents=True, exist_ok=True)
    with (output / (name + '.csv')).open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    escape = lambda value: str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    lines = ['\\begin{tabular}{' + 'l' * len(headers) + '}', r'\toprule',
             ' & '.join(map(escape, headers)) + r' \\', r'\midrule']
    lines += [' & '.join(map(escape, row)) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / (name + '.tex')).write_text('\n'.join(lines) + '\n')


def quality(row):
    lo, hi = row['accuracy_ci95']
    return f"{100 * row['accuracy']:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]"


def timing(row):
    return f"{row['median_warm_seconds']:.2f} [{min(row['warm_seconds']):.2f}, {max(row['warm_seconds']):.2f}]"


def structured(root):
    rows = []
    for dataset, name in DATASETS.items():
        result = analysis(main_run(root, dataset))
        for method, row in result['methods'].items():
            rows.append([name, METHODS[method], quality(row), f"{100 * row['macro_f1']:.2f}",
                         timing(row), f"{row['peak_allocated_gib']:.2f}"])
    write_table(root, 'T1_structured', ['Dataset', 'Method', 'Accuracy [95% CI]', 'Macro-F1',
                                      'Warm s [min,max]', 'GiB'], rows)


def mechanism(root):
    data = json.loads((root / 'results/readout_real_activations.json').read_text())
    rows = []
    for precision, row in data['precisions'].items():
        candidate = row['timing_summary']['candidate_rows']['cuda_ms']['median']
        full = row['timing_summary']['full_vocabulary_then_gather']['cuda_ms']['median']
        rows.append([precision, f'{candidate:.4f}', f'{full:.4f}', f'{full / candidate:.2f}',
                     row['equivalence']['argmax_disagreements']])
    write_table(root, 'T2_projection', ['Precision', 'Candidate ms', 'Full ms', 'Ratio', 'Changed / 24'], rows)
    run = main_run(root, 'race_middle')
    data = analysis(run)
    records = read(run / 'trials.jsonl')
    rows = []
    for method in ['independent', 'shared', 'streamed']:
        row = data['methods'][method]
        tokens = sum(record['result']['computed_input_tokens'] for record in records
                     if record['method'] == method and record['repeat'] == 0)
        rows.append([METHODS[method], tokens, timing(row), f"{row['peak_allocated_gib']:.2f}"])
    write_table(root, 'T2_cache', ['RACE execution', 'Computed tokens', 'Warm s [min,max]', 'GiB'], rows)


def batch_tile(root):
    configs = [(b, 128, f'squad-batch{b}-001') for b in [1, 4, 16]]
    configs += [(8, tile, f'squad-tile{tile}-001') for tile in [16, 64, 128]]
    rows = []
    for batch, tile, run in sorted(configs):
        for method, row in analysis(root / 'runs' / run)['methods'].items():
            rows.append([batch, tile, METHODS[method], f"{100 * row['accuracy']:.2f}",
                         timing(row), f"{row['peak_allocated_gib']:.2f}"])
    write_table(root, 'T3_batch_tile', ['Root batch', 'Branch tile', 'Method', 'Accuracy',
                                     'Warm s [min,max]', 'GiB'], rows)


def robustness(root):
    rows = []
    exclusions = json.loads((root / 'configs/schema/option_order_exclusions.json').read_text())['datasets']
    for dataset, name in DATASETS.items():
        original = root / 'runs' / (dataset + '-neutral-evaluation-001')
        rotated = root / 'runs' / (dataset + '-neutral-rotated-001')
        base = analysis(original)
        first, second = read(original / 'trials.jsonl'), read(rotated / 'trials.jsonl')
        labels = Path(base['labels'])
        labels = labels if labels.is_absolute() else root / labels
        gold = {(row['task_id'], field): answer for row in read(labels)
                for field, answer in row['labels'].items()}
        excluded = set(exclusions.get(dataset, {}).get('excluded_roots', []))
        task_ids = sorted({key[0] for key in predictions(first, 'independent')} - excluded)
        clusters = (json.loads(labels.with_name('clusters.json').read_text())['clusters']
                    if dataset == 'scifact_cited' else {task: task for task in task_ids})
        cluster_ids = sorted({clusters[task] for task in task_ids})
        positions = {key: index for index, key in enumerate(cluster_ids)}
        draws = np.random.default_rng(base['uncertainty']['seed']).integers(
            len(cluster_ids), size=(base['uncertainty']['bootstrap_samples'], len(cluster_ids)))
        for method in METHODS:
            before, after = predictions(first, method), predictions(second, method)
            before = {key: value for key, value in before.items() if key[0] not in excluded}
            after = {key: value for key, value in after.items() if key[0] not in excluded}
            assert before.keys() == after.keys()
            agreement = sum(before[key]['choice'] == after[key]['choice'] for key in before) / len(before)
            delta, counts = np.zeros(len(cluster_ids)), np.zeros(len(cluster_ids))
            for key in before:
                index = positions[clusters[key[0]]]
                delta[index] += int(after[key]['choice'] == gold[key]) - int(before[key]['choice'] == gold[key])
                counts[index] += 1
            lo, hi = np.quantile(delta[draws].sum(axis=1) / counts[draws].sum(axis=1), [.025, .975])
            change = f'{100 * delta.sum() / counts.sum():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]'
            accuracy_before = sum(value['choice'] == gold[key] for key, value in before.items()) / len(before)
            accuracy_after = sum(value['choice'] == gold[key] for key, value in after.items()) / len(after)
            row = probability_scores(before, gold, base['calibration']['bins'])
            rows.append([f'{name} ({len(before)})', METHODS[method], f'{100 * accuracy_before:.2f}',
                         f'{100 * accuracy_after:.2f}', change, f'{100 * agreement:.2f}',
                         f"{row['brier']:.3f}" if row['brier'] is not None else '--',
                         f"{row['ece']:.3f}" if row['ece'] is not None else '--'])
    write_table(root, 'T5_robustness', ['Dataset', 'Method', 'Original acc.', 'Rotated acc.',
                                     'Delta [95% CI]', 'Choice agreement', 'Brier', 'ECE'], rows)


def scaling(root):
    rows = []
    for policy, sizes in [('per_gpu', [1, 2, 4]), ('global', [2, 4])]:
        for size in sizes:
            run = root / 'runs' / f'scaling-{policy}-{size}gpu-001'
            data = json.loads((run / 'summary.json').read_text())
            for method in METHODS:
                trials = [row for row in data['trials'] if row['method'] == method]
                rows.append([policy, size, data['batch_size_per_gpu'], METHODS[method],
                             f"{np.median([r['dispatch_seconds'] for r in trials]):.2f}",
                             f"{np.median([r['compute_span_seconds'] for r in trials]):.2f}",
                             f"{max(max(r['peak_allocated_gib_per_gpu']) for r in trials):.2f}",
                             f"{data['startup_and_full_warmup_seconds']:.2f}"])
    write_table(root, 'T4_scaling', ['Batch policy', 'GPUs', 'Batch/GPU', 'Method',
                                  'Dispatch s', 'Compute s', 'Max GiB/GPU', 'Load+warmup s'], rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--tables', nargs='+', choices=['T1', 'T2', 'T3', 'T4', 'T5'], required=True)
    args = parser.parse_args()
    functions = {'T1': structured, 'T2': mechanism, 'T3': batch_tile, 'T4': scaling, 'T5': robustness}
    for table in args.tables:
        functions[table](args.root)
        print(table)


if __name__ == '__main__':
    main()
