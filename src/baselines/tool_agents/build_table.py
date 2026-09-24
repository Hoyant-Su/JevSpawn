import argparse
import csv
import json
from pathlib import Path
import statistics

from compare_accuracy import agent_predictions, jsonl, native_predictions


def agent_row(name, run, memory_run, ids):
    metadata = json.loads((run / 'run.json').read_text())
    predictions = [agent_predictions(run, ids, repeat) for repeat in range(metadata['repeats'])]
    assert metadata['repeats'] == 3
    rows = jsonl(run / 'predictions.jsonl')
    first = [row for row in rows if row['repeat'] == 0]
    times = [sum({row['batch_index']: row['batch_elapsed_seconds'] for row in rows
                  if row['repeat'] == repeat}.values()) for repeat in range(3)]
    memory = jsonl(memory_run / 'predictions.jsonl')
    memory_meta = json.loads((memory_run / 'run.json').read_text())
    assert memory_meta['measurement_kind'] == 'memory'
    assert memory_meta['repeats'] == 1 and memory_meta['config'] == metadata['config']
    assert len(memory) == len(ids) and {row['task_id'] for row in memory} == ids
    assert all(row['stage'] == 'memory_measurement' for row in memory)
    for row in memory:
        assert max(event['peak_cuda_memory_bytes'] for event in row['trace']
                   if event['event'] == 'generation') <= row['peak_allocated_bytes']
    memory_predictions = {row['task_id']: row['choice'] if row['status'] == 'complete' else None for row in memory}
    return {
        'method': name, 'valid': sum(row['status'] == 'complete' for row in first),
        'output_tokens': sum(row['output_tokens'] for row in first),
        'consumed_input_tokens': sum(row['input_tokens'] for row in first),
        'model_sequence_calls': sum(row['model_calls'] for row in first),
        'tool_calls': sum(row['tool_calls'] for row in first),
        'seconds_median': statistics.median(times), 'seconds_min': min(times), 'seconds_max': max(times),
        'peak_allocated_gib': max(row['peak_allocated_bytes'] for row in memory) / 2 ** 30,
        'max_itl_ms': 1000 * max(interval for row in rows for event in row['trace']
                                 if event['event'] == 'generation' for interval in event['decode']['inter_token_seconds']),
        'changed_repeat1': sum(predictions[1][key] != predictions[0][key] for key in ids),
        'changed_repeat2': sum(predictions[2][key] != predictions[0][key] for key in ids),
        'memory_run': str(memory_run), 'memory_tasks': len(memory),
        'memory_prediction_matches': sum(memory_predictions[key] == predictions[0][key] for key in ids),
        'call_token_cap': metadata['config']['max_new_tokens'],
        'task_token_cap': metadata['config']['max_output_tokens_per_task'],
        'task_call_cap': metadata['config']['max_model_calls_per_task'],
    }


def native_row(name, run, method, ids):
    predictions = [native_predictions(run, method, ids, repeat) for repeat in range(3)]
    rows = [row for row in jsonl(run / 'trials.jsonl') if row['method'] == method]
    first = [row for row in rows if row['repeat'] == 0]
    times = [sum(row['elapsed_seconds'] for row in rows if row['repeat'] == repeat) for repeat in range(3)]
    return {
        'method': name, 'valid': len(ids), 'output_tokens': 0,
        'consumed_input_tokens': sum(row['result']['computed_input_tokens'] for row in first),
        'model_sequence_calls': None, 'tool_calls': 0,
        'seconds_median': statistics.median(times), 'seconds_min': min(times), 'seconds_max': max(times),
        'peak_allocated_gib': max(row['peak_allocated_bytes'] for row in rows) / 2 ** 30,
        'max_itl_ms': None,
        'changed_repeat1': sum(predictions[1][key] != predictions[0][key] for key in ids),
        'changed_repeat2': sum(predictions[2][key] != predictions[0][key] for key in ids),
        'memory_run': str(run), 'memory_tasks': len(ids) * 3,
        'memory_prediction_matches': None, 'call_token_cap': 0, 'task_token_cap': 0, 'task_call_cap': None,
    }


def difference(comparison, left, right):
    if left == right:
        return None
    pair, = [row for row in comparison['paired_differences'] if {row['left'], row['right']} == {left, right}]
    sign = 1 if pair['left'] == left else -1
    return [100 * sign * pair['difference'], *sorted(100 * sign * value for value in pair['interval'])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--memory-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = args.config.resolve().parent
    comparison = json.loads((directory / config['output']).read_text())
    ids = {row['task_id'] for row in jsonl(directory / config['labels'])}
    rows = [agent_row(name, directory / path, args.memory_root / Path(path).name, ids)
            for name, path in config['agents'].items()]
    rows += [native_row(name, directory / source['run'], source['method'], ids)
             for name, source in config['native'].items()]
    for row in rows:
        row.update(questions=len(ids), correct=comparison['accuracy'][row['method']]['correct'],
                   accuracy_percent=100 * comparison['accuracy'][row['method']]['accuracy'])
        for reference in ['single_reasoning', 'jevspawn_streamed']:
            delta = difference(comparison, row['method'], reference)
            for suffix, value in zip(['difference_pp', 'ci_low_pp', 'ci_high_pp'], delta or [None] * 3):
                row[f'vs_{reference}_{suffix}'] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    names = {'react': 'Interleaved tool policy', 'compiler_adapter': 'Planned tool policy',
             'single_reasoning': 'Single call reasoning', 'jevspawn_streamed': 'JevSpawn categorical readout'}
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Method & Correct / valid & Acc. (\%) & Time (s), median [range] & Peak GiB & Max ITL (ms) \\', r'\midrule']
    for row in rows:
        itl = '--' if row['max_itl_ms'] is None else f"{row['max_itl_ms']:.2f}"
        lines.append(f"{names[row['method']]} & {row['correct']} / {row['valid']} & {row['accuracy_percent']:.2f} & {row['seconds_median']:.2f} [{row['seconds_min']:.2f}, {row['seconds_max']:.2f}] & {row['peak_allocated_gib']:.2f} & {itl} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\par\medskip',
              r'\begin{tabular}{lrrr}', r'\toprule',
              r'Method & Output tokens & Model / tool calls & Changed answers \\', r'\midrule']
    for row in rows:
        calls = '--' if row['model_sequence_calls'] is None else str(row['model_sequence_calls'])
        lines.append(f"{names[row['method']]} & {row['output_tokens']:,} & {calls} / {row['tool_calls']} & {row['changed_repeat1']} / {row['changed_repeat2']} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\medskip',
              r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lrr}', r'\toprule',
              r'Method & $\Delta$ vs. single reasoning (pp), 95\% CI & $\Delta$ vs. categorical readout (pp), 95\% CI \\', r'\midrule']
    for row in rows:
        cells = []
        for reference in ['single_reasoning', 'jevspawn_streamed']:
            delta = difference(comparison, row['method'], reference)
            cells.append('--' if delta is None else f'{delta[0]:+.2f} [{delta[1]:+.2f}, {delta[2]:+.2f}]')
        lines.append(names[row['method']] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}',
              r'\caption{Reasoning and tool use on the 254 AQuA test questions. All policies use Qwen3.5-4B on one H100 with batch size eight and greedy decoding. The two tool policies allocate 2,048 output tokens across at most four calls, whereas single call reasoning uses the same budget in one call. Time gives the median and range over three repetitions. Agent times cover complete workflows, and categorical time measures decision execution. Memory includes model weights. Accuracy intervals use paired question bootstrap samples. Changed answers count disagreements with the first repetition. Measurement details appear in Appendix~\ref{sec:setup}.}',
              r'\label{tab:tool-agents}', r'\end{table*}']
    args.output.with_suffix('.tex').write_text('\n'.join(lines) + '\n')
    args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
