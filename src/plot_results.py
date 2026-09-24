"""Plot measured systems gains beside paired quality uncertainty."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from build_tables import main_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    datasets = [('squad2_answerability', 'SQuAD2 answerability'), ('race_middle', 'RACE-middle'),
                ('scifact_cited', 'SciFact cited'), ('pubmedqa', 'PubMedQA'), ('aqua', 'AQuA')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.8), sharey=True,
                             gridspec_kw={'width_ratios': [1, 1, 1.1]}, layout='constrained')
    y = np.arange(len(datasets))
    sources = []
    for index, (dataset, name) in enumerate(datasets):
        path = main_run(args.root, dataset) / 'analysis.json'
        data = json.loads(path.read_text())
        baseline, method = data['methods']['independent'], data['methods']['streamed']
        ratios = np.array(baseline['warm_seconds']) / np.array(method['warm_seconds'])
        median = np.median(ratios)
        axes[0].errorbar(median, index, xerr=[[median - ratios.min()], [ratios.max() - median]],
                         fmt='o', color='#0072B2', capsize=3, markersize=5)
        axes[0].text(median + .05, index - .14, f'{median:.2f}', fontsize=7, color='#0072B2')
        saving = 100 * (1 - method['peak_allocated_gib'] / baseline['peak_allocated_gib'])
        axes[1].plot(saving, index, 's', color='#009E73', markersize=5)
        axes[1].text(saving + 1, index - .14, f'{saving:.1f}', fontsize=7, color='#009E73')
        pair = data['pairs']['streamed-minus-independent']
        delta = 100 * pair['accuracy_difference']
        lo, hi = np.array(pair['accuracy_difference_ci95']) * 100
        axes[2].errorbar(delta, index, xerr=[[delta - lo], [hi - delta]], fmt='D',
                         color='#D55E00', capsize=3, markersize=4)
        sources.append({'dataset': dataset, 'source': str(path), 'speed_ratios': ratios.tolist(),
                        'memory_reduction_percent': saving, 'accuracy_difference_pp': delta,
                        'accuracy_difference_ci95_pp': [lo, hi]})
    axes[0].set_yticks(y, [name for _, name in datasets])
    axes[0].invert_yaxis()
    labels = ['Independent / streamed time', 'Peak memory reduction (%)', 'Accuracy change (pp)']
    for index, ax in enumerate(axes):
        ax.set_xlabel(labels[index])
        ax.axvline(1 if index == 0 else 0, color='0.65', linewidth=.8, linestyle='--', zorder=0)
        ax.text(0, 1.05, 'ABC'[index], transform=ax.transAxes, weight='bold', fontsize=10)
        ax.tick_params(axis='y', length=0)
    for ax in axes:
        ax.margins(x=.2)
    output = args.root / 'manuscript/figures'
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / 'structured_tradeoffs.pdf')
    fig.savefig(output / 'structured_tradeoffs.png', dpi=300)
    (output / 'structured_tradeoffs.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
