import argparse
import json
from pathlib import Path



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    stage = json.loads(args.stage.read_text())
    source = Path(stage['run']) / 'evaluation.json'
    report = json.loads(source.read_text())
    count = report['task_count']
    assert count == stage['data']['task_count']
    lines = [r'\begin{table}[t]', r'\centering\small', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Method & Valid & EM & F1 & Time & Memory & Workers \\', r'\midrule']
    rows = []
    for arm, name in [('direct', 'Direct RAG'), ('tiled_independent', 'Independent references'),
                      ('streamed', 'Streamed references'), ('json', 'JSON references')]:
        value = report['methods'][arm]
        assert len(value['queries']) == count
        lines.append(f'{name} & {value["completed"]}/{count} & {100 * value["exact_match"]:.1f} & '
                     f'{100 * value["token_f1"]:.1f} & {value["elapsed_seconds"]:.2f} & '
                     f'{value["peak_allocated_bytes"] / 2**30:.2f} & {value["mean_workers"]:.1f}' + r' \\')
        rows.append({'arm': arm, **{key: item for key, item in value.items() if key != 'queries'}})
    fixed = stage['fixed']
    caption = (
        f'Source reference interfaces on {count} MultiHop-RAG {report["dataset_split"]} questions. '
        'The model defines evidence requests and workers hierarchically select source references. '
        'Final synthesis reads the selected original passages. EM and token F1 are percentages over all questions. '
        'Time includes planning, retrieval, source selection, dereferencing and final generation, in seconds. '
        'Memory is peak allocated GiB including model weights. Workers is the mean number of finite source selections per question. '
        f'All methods use {Path(fixed["model_path"]).name}, BF16, {fixed["gpu_count"]} H100, '
        f'batch capacity {fixed["model_batch_capacity"]} and {fixed["root_tasks_in_flight"]} original question in flight. '
        f'Direct RAG reads {fixed["initial_search_count"]} passages retrieved from the original question. '
        f'Reference policies retrieve {fixed["field_search_count"]} passages per generated request, '
        'so retrieval policy and realized evidence differ. '
        'Each method receives full workload warmup and one measured pass.')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\caption{' + caption + '}',
                  r'\label{' + args.label + '}', r'\end{table}'])
    args.target.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    args.target.with_suffix('.json').write_text(json.dumps({
        'source': str(source), 'rows': rows, 'paired_comparisons': report['paired_comparisons'],
        'mean_reference_utility': report['mean_reference_utility'],
        'reference_utility_interval95': report['reference_utility_interval95'],
        'paired_exact_match_difference': report['paired_exact_match_difference'],
        'paired_exact_match_interval95': report['paired_exact_match_interval95']}, indent=2) + '\n')
    print(f'Wrote {args.target} for {count} original tasks')


if __name__ == '__main__':
    main()
