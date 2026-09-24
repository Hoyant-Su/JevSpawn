import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    rows = list(map(json.loads, (args.run / 'predictions.jsonl').read_text().splitlines()))
    labels = {row['task_id']: next(iter(row['labels'].values())) for row in map(json.loads, args.labels.read_text().splitlines())}
    metadata = json.loads((args.run / 'run.json').read_text())
    assert len(rows) == metadata['task_count'] * metadata['repeats']
    assert len({(row['repeat'], row['task_id']) for row in rows}) == len(rows)
    assert {row['repeat'] for row in rows} == set(range(metadata['repeats']))
    batches, generations, intervals, ttft = {}, {}, [], []
    for row in rows:
        assert row['model_calls'] <= metadata['config']['max_model_calls_per_task']
        assert row['output_tokens'] <= metadata['config']['max_output_tokens_per_task']
        batches[(row['repeat'], row['batch_index'])] = row['batch_elapsed_seconds']
        events = [event for event in row['trace'] if event['event'] == 'generation']
        assert len(events) == row['model_calls']
        assert sum(event['output_tokens'] for event in events) == row['output_tokens']
        for event in events:
            decode = event['decode']
            assert len(decode['forward_completion_seconds']) == event['output_tokens']
            assert len(decode['inter_token_seconds']) == event['output_tokens'] - 1
            generations[(row['repeat'], row['batch_index'], event['stage'])] = event['batch_seconds']
            intervals.extend(decode['inter_token_seconds'])
            ttft.append(decode['ttft_seconds'])
    correct = sum(row['status'] == 'complete' and row['choice'] == labels[row['task_id']] for row in rows)
    result = {
        'method': metadata['method'], 'tasks': metadata['task_count'], 'repeats': metadata['repeats'], 'task_observations': len(rows),
        'complete': sum(row['status'] == 'complete' for row in rows), 'correct': correct,
        'accuracy': correct / len(rows),
        'valid_rate': sum(row['status'] == 'complete' for row in rows) / len(rows),
        'failure_reasons': dict(Counter(row['error'] for row in rows if row['status'] == 'failed')),
        'root_batch_sizes': sorted({row['root_batch_size'] for row in rows}),
        'elapsed_seconds': sum(batches.values()), 'generation_seconds': sum(generations.values()),
        'generation_batches': len(generations), 'model_sequence_calls': sum(row['model_calls'] for row in rows),
        'tool_calls': sum(row['tool_calls'] for row in rows),
        'input_tokens': sum(row['input_tokens'] for row in rows), 'output_tokens': sum(row['output_tokens'] for row in rows),
        'itl_ms': dict(zip(['p50', 'p95', 'p99', 'max'], (np.quantile(intervals, [0.5, 0.95, 0.99, 1]) * 1000).tolist())),
        'ttft_ms': dict(zip(['p50', 'p95', 'max'], (np.quantile(ttft, [0.5, 0.95, 1]) * 1000).tolist())),
        'itl_scope': metadata['decode_timing_scope'],
        'timing_scope': metadata['timing_scope'],
    }
    result['per_repeat'] = {}
    for repeat in range(metadata['repeats']):
        selected = [row for row in rows if row['repeat'] == repeat]
        result['per_repeat'][repeat] = {
            'tasks': len(selected),
            'complete': sum(row['status'] == 'complete' for row in selected),
            'correct': sum(row['status'] == 'complete' and row['choice'] == labels[row['task_id']] for row in selected),
            'elapsed_seconds': sum(seconds for (trial, _), seconds in batches.items() if trial == repeat),
        }
    result['accuracy_scope'] = 'Accuracy and valid_rate aggregate all measured task observations; per_repeat preserves separate counts.'
    if 'measurement_kind' in metadata:
        result.update(measurement_kind=metadata['measurement_kind'],
                      peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows),
                      peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in rows),
                      memory_scope=metadata['memory_scope'])
    (args.run / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
