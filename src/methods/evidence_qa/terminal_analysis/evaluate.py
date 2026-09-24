import argparse
import importlib.util
from pathlib import Path
import statistics

import numpy as np

from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.evidence_qa.evaluate import answer_metrics
from jev_spawn.infra.prompts import resolve_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    rows = read_jsonl(stage['inputs'])
    identities = [row['task_id'] for row in rows]
    outcomes = {}
    for arm in stage['arms']:
        assert read(args.output / arm / 'completed.json') == {'task_ids': identities, 'arm': arm}
        outcomes[arm] = [read(args.output / arm / 'measured' / f'{index:03d}' / 'outcome.json')
                         for index in range(len(rows))]
        assert [row['task_id'] for row in outcomes[arm]] == identities
    labels = {row['task_id']: row for row in read_jsonl(stage['labels'])}
    original_stage = resolve_prompts(read(stage['source_stage']))
    specification = importlib.util.spec_from_file_location('multihoprag_qa', original_stage['evaluation']['upstream'])
    upstream = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(upstream)
    methods = {}
    for arm, values in outcomes.items():
        queries = [{**value, 'scores': answer_metrics(value, labels[value['task_id']]['answer'], upstream)}
                   for value in values]
        calls = [call for index in range(len(rows))
                 for call in read(args.output / arm / 'measured' / f'{index:03d}' / 'calls.json')]
        intervals = [interval for call in calls for sequence in call['decode']
                     for interval in sequence['inter_token_seconds']]
        methods[arm] = {'queries': queries, 'completed': sum(row['status'] == 'completed' for row in queries),
                        'unavailable': sum(row['status'] == 'unavailable' for row in queries),
                        'exact_match': statistics.mean(row['scores']['exact_match'] for row in queries),
                        'token_f1': statistics.mean(row['scores']['token_f1'] for row in queries),
                        'elapsed_seconds': sum(row['elapsed_seconds'] for row in queries if row['elapsed_seconds'] is not None),
                        'peak_allocated_bytes': max(call['peak_allocated_bytes'] for call in calls),
                        'output_tokens': sum(sum(call['result']['output_tokens']) for call in calls),
                        'max_itl_seconds': max(intervals),
                        'intervals_over_100ms': sum(value >= .1 for value in intervals)}
    difference = np.array([a['scores']['exact_match'] - b['scores']['exact_match']
                          for a, b in zip(methods['values']['queries'], methods['no_values']['queries'])])
    rng = np.random.default_rng(original_stage['evaluation']['seed'])
    samples = difference[rng.integers(len(rows), size=(original_stage['evaluation']['bootstrap_samples'], len(rows)))].mean(axis=1)
    original = read(Path(original_stage['run']) / 'evaluation.json')
    report = {'methods': methods, 'mean_message_effect': float(difference.mean()),
              'message_effect_interval95': np.quantile(samples, [.025,.975]).tolist(),
              'original_streamed_exact_match': original['methods']['streamed']['metrics']['exact_match'],
              'direct_rag_exact_match': original['methods']['direct']['metrics']['exact_match'],
              'scope': 'Terminal receiver only. No reconstructed task latency or architecture promotion.'}
    write(args.output / 'evaluation.json', report)
    print({arm: {key: value for key, value in result.items() if key != 'queries'}
           for arm, result in methods.items()})


if __name__ == '__main__':
    main()
