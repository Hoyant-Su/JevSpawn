import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    names = {'compact_json': 'Finite JSON', 'independent': 'Independent readout',
             'shared': 'Shared readout', 'streamed': 'Streamed readout'}
    comparisons = {r['method']: r for r in report['paired_comparisons'] if r['reference'] == 'compact_json'}
    rows, latex = [], []
    for method, result in report['methods'].items():
        quality = result['quality']
        row = {'method': method, 'available_solutions': report['available_solutions'],
               'requested_solutions': report['requested_solutions'], 'coverage': report['coverage'],
               'problem_clusters': report['problem_clusters'],
               'service_mean_seconds': result['verification_service_mean_seconds'],
               'service_sd_seconds': result['verification_service_sd_seconds'],
               'max_allocated_gib_per_gpu': max(r['peak_allocated_gib_per_gpu'] for r in result['repeats'])}
        for name in ['accuracy', 'false_accept_rate', 'false_reject_rate', 'balanced_accuracy']:
            row[name + '_percent'] = 100 * quality[name]['estimate']
            row[name + '_ci_low_percent'], row[name + '_ci_high_percent'] = [100 * x for x in quality[name]['ci']]
        difference, ci = (0.0, [0.0, 0.0]) if method == 'compact_json' else (
            comparisons[method]['accuracy_difference'], comparisons[method]['accuracy_difference_ci'])
        row.update(accuracy_difference_pp=100 * difference,
                   accuracy_difference_ci_low_pp=100 * ci[0], accuracy_difference_ci_high_pp=100 * ci[1])
        rows.append(row)
        delta = '--' if method == 'compact_json' else f'{100 * difference:+.2f} [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]'
        latex.append(' & '.join([names[method]] + [f'{row[name + "_percent"]:.2f}' for name in
                     ['accuracy', 'balanced_accuracy', 'false_accept_rate', 'false_reject_rate']] +
                     [f'{row["service_mean_seconds"]:.2f} $\\pm$ {row["service_sd_seconds"]:.2f}',
                      f'{row["max_allocated_gib_per_gpu"]:.3f}', delta]) + r' \\')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'ADv2_component.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = r'''\begin{table*}[t]
\centering
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrrl}
\toprule
Verifier & Acc. (\%) & Bal. acc. (\%) & FAR (\%) & FRR (\%) & Service (s) & GiB/GPU & $\Delta$ Acc. [95\% CI] \\
\midrule
''' + '\n'.join(latex) + r'''
\bottomrule
\end{tabular}
}
\caption{Verification with the ADv2 indicator pool on fixed ProcessBench GSM8K solutions using frozen Qwen3.5-4B.
Accuracy is conditional on valid retrieved indicators for 398 of 400 solutions, corresponding to 99.5 percent coverage, 374 distinct problems, and 830 indicator decisions.
Two retrieval abstentions arise from an empty selection and a selection enclosed in Markdown fences that violates the specified output interface. Neither receives a mathematical verdict.
All methods share the same original indicator pool, actual retrieved indicators, and input states.
FAR is acceptance among flawed solutions, and FRR is rejection among solutions with no annotated reasoning error.
Service time is the mean $\pm$ sample standard deviation over three repetitions of summed synchronized verification calls across two ranks. It does not measure parallel dispatch or complete agent latency.
Memory is maximum allocated CUDA memory on any one GPU. Verification retains only the 4B model, while retrieval also retains Qwen3-Embedding-0.6B.
Quality uses repetition zero. Paired percentile confidence intervals versus finite JSON use 2,000 bootstrap draws over problem clusters with seed zero.
The comparison evaluates a verification component rather than the complete ADv2 multiagent workflow.}
\label{tab:adv2-component}
\end{table*}
'''
    (args.output / 'ADv2_component.tex').write_text(table)
    print(json.dumps({'output': str(args.output), 'methods': len(rows)}))


if __name__ == '__main__':
    main()
