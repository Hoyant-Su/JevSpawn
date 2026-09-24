import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def predictions(entry):
    run = ROOT / entry['run']
    protocol = read(run / 'protocol.json')
    tasks = [json.loads(line) for line in Path(protocol['settings']['tasks']).read_text().splitlines()]
    labels = [json.loads(line) for line in Path(protocol['settings']['labels']).read_text().splitlines()]
    gold = {row['task_id']: row['labels']['q0'] for row in labels}
    if entry['format'] == 'latentmas':
        records = [row for p in sorted(run.glob('block-*.json')) for row in read(p)['predictions']]
        answers = [row['answer'] for row in records]
    elif entry['format'] == 'agentprune':
        records = [row for p in sorted(run.glob('block-*/complete.json')) for row in read(p)['result']['records']]
        answers = [row['answer'] for row in records]
    else:
        assert entry['format'] in {'formal_choices', 'single_reasoning'}
        records = [row for p in sorted(run.glob('block-*/complete.json')) for row in read(p)['results']]
        if entry['format'] == 'single_reasoning':
            answers = [row['choice'] if row['status'] == 'complete' else None for row in records]
        else:
            answers = [row['answer'] if row['status'] == 'completed' else None for row in records]
    ids = [task['task_id'] for task in tasks]
    assert [row['task_id'] for row in records] == ids and set(gold) == set(ids)
    correct = np.array([answer == gold[task_id] for task_id, answer in zip(ids, answers)], dtype=np.int8)
    assert len(correct) == entry['tasks'] and int(correct.sum()) == entry['correct']
    return tasks, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--published', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    assert args.resamples > 0
    records = [dict(row, group='Published methods') for row in read(args.published)['records']]
    records += [dict(row, group='Single agent control') for row in read(args.controls)['records']]
    rows, pairs = [], []
    generator = np.random.default_rng(args.seed)
    lines = [r'\begin{table}[t]', r'\centering\small',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Method & Accuracy & 95 percent CI & Valid & Time & Memory \\', r'\midrule']
    for dataset in dict.fromkeys(row['dataset'] for row in records):
        selected = [row for row in records if row['dataset'] == dataset]
        tasks, _ = predictions(selected[0])
        indices = generator.integers(len(tasks), size=(args.resamples, len(tasks)))
        vectors = {}
        lines.append(r'\multicolumn{6}{l}{\textit{' + dataset + r'}} \\')
        for entry in selected:
            current, correct = predictions(entry)
            assert current == tasks
            vectors[entry['method']] = correct
            low, high = np.quantile(correct[indices].mean(axis=1) * 100, [.025, .975])
            rows.append({**entry, 'accuracy_interval_percent': [float(low), float(high)]})
            method = entry['method']
            if entry['group'] == 'Published methods':
                method += r'~\cite{' + entry['citation'] + '}'
            else:
                lines.append(r'\cmidrule(lr){1-6}')
            lines.append(f"{method} & {entry['accuracy'] * 100:.2f} & "
                         f"$[{low:.2f},{high:.2f}]$ & {entry['valid']}/{entry['tasks']} & "
                         f"{entry['elapsed_seconds'] / 60:.2f} & {entry['peak_allocated_bytes'] / 2**30:.2f} " + r'\\')
        for left, right in combinations(vectors, 2):
            delta = vectors[left] - vectors[right]
            interval = np.quantile(delta[indices].mean(axis=1) * 100, [.025, .975])
            pairs.append({'dataset': dataset, 'left': left, 'right': right,
                          'difference_pp': float(delta.mean() * 100),
                          'interval_95_pp': interval.tolist(),
                          'left_only_correct': int((delta == 1).sum()),
                          'right_only_correct': int((delta == -1).sum())})
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}',
              r'\caption{Question answering on fixed subsets of 256 SuperGPQA and 256 MedXpertQA Text questions. '
              r'All methods use Qwen3.5-4B, BF16, one H100, and a batch capacity of eight. '
              r'Accuracy is a percentage, time is measured workload time in minutes, and memory is peak allocated GiB. '
              r'Intervals use ' + f'{args.resamples:,}' + r' question bootstrap samples. '
              r'Time excludes loading, warmup, and topology learning. Reasoning budgets differ across methods. '
              r'The single agent control uses one call with an output budget of 2,048 tokens.}',
              r'\label{tab:cross-domain-choices}', r'\end{table}']
    args.output.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    args.output.with_suffix('.json').write_text(json.dumps({
        'seed': args.seed, 'resamples': args.resamples,
        'scope': 'Completed operating points only. Pointwise, unadjusted question bootstrap intervals conditional on each measured execution. Not equal-compute comparisons.',
        'records': rows, 'paired_differences': pairs}, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'paired_comparisons': len(pairs)}))


if __name__ == '__main__':
    main()
