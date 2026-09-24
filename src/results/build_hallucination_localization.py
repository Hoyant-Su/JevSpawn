import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    stage = json.loads(args.stage.read_text())
    source = Path(stage['run']) / 'evaluation.json'
    report = json.loads(source.read_text())
    count = report['task_count']
    assert count == stage['data']['task_count']
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrrrrrr}',
             r'\toprule', r'Method & Valid & Pooled F1 & Mean F1 & Time & Memory & Workers & Depth \\',
             r'\midrule']
    rows = []
    for arm, name in [('direct', 'Direct spans'), ('tiled_independent', 'Independent decisions'),
                      ('streamed', 'Streamed decisions'), ('json', 'JSON decisions')]:
        value = report['methods'][arm]
        assert len(value['queries']) == count
        lines.append(f'{name} & {value["completed"]}/{count} & {100 * value["character"]["f1"]:.1f} & '
                     f'{100 * value["mean_response_character_f1"]:.1f} & {value["elapsed_seconds"]:.2f} & '
                     f'{value["peak_allocated_bytes"] / 2**30:.2f} & {value["mean_workers"]:.1f} & '
                     f'{value["maximum_depth"]}' + r' \\')
        rows.append({'arm': arm, **{key: item for key, item in value.items() if key != 'queries'}})
    fixed = stage['fixed']
    caption = (
        f'Hallucination localization on {count} RAGTruth development responses, balanced across '
        'question answering, summarization, and data to text generation. '
        'Pooled F1 measures character overlap across responses. Mean F1 averages response scores, '
        'assigning zero to invalid outputs and one to valid empty predictions with empty annotations. '
        'Both scores are percentages. Time includes partition generation, worker decisions, and span reconstruction '
        'in seconds. Memory is peak allocated GiB including weights. Workers is the mean number of local '
        'span decisions per response, and depth is the maximum realized refinement depth. '
        'Direct generation returns exact hallucinated substrings without a partition. '
        f'All methods use {Path(fixed["model_path"]).name}, BF16, one H100, model batch capacity '
        f'{fixed["model_batch_capacity"]}, and one response in flight. '
        'Each method receives a complete warmup pass and one measured pass. All assigned responses '
        'contribute to quality and workload time, including failed predictions.')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
                  r'\label{tab:hallucination-localization}', r'\end{table}'])
    args.target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    args.target.with_suffix('.json').write_text(json.dumps({
        'source': str(source), 'rows': rows, 'paired_comparisons': report['paired_comparisons'],
        'uncertainty': report['uncertainty']}, indent=2) + '\n')
    print(f'Wrote {args.target} for {count} responses')


if __name__ == '__main__':
    main()
