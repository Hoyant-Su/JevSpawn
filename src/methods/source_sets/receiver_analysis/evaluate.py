import argparse
from collections import Counter
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
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    parent = resolve_prompts(read(stage['source_stage']))
    inputs = read(stage['inputs'])
    tasks = read_jsonl(stage['data']['tasks'])
    identities = [row['task_id'] for row in tasks]
    labels = {row['task_id']: row for row in read_jsonl(stage['data']['labels'])}
    assert len(tasks) == stage['data']['task_count'] and set(labels) == set(identities)
    spec = importlib.util.spec_from_file_location('multihoprag_qa', parent['evaluation']['upstream'])
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng = np.random.default_rng(parent['evaluation']['seed'])
    indices = rng.integers(len(tasks), size=(parent['evaluation']['bootstrap_samples'], len(tasks)))
    methods = {}
    for arm in stage['arms']:
        directory = Path(stage['output']) / arm
        assigned = [row for row in inputs if row['arm'] == arm]
        assert [row['task_id'] for row in assigned] == identities
        assert read(directory / 'completed.json') == {'task_ids': identities, 'arm': arm}
        assert read(directory / 'protocol.json') == {
            'fixed': stage['fixed'], 'system': stage['system'], 'inputs': assigned, 'arm': arm}
        queries, sessions = [], set()
        for index, row in enumerate(assigned):
            outcome = read(directory / 'measured' / f'{index:03d}' / 'outcome.json')
            result = outcome['result']
            assert result['task_id'] == row['task_id']
            assert outcome['source_outcome'] == row['source_outcome']
            call, = read(Path(outcome['attempt']) / 'calls.json')
            assert call['kind'] == 'answer' and call['result']['batch_size'] == 1
            assert call['context'] == {'system': stage['system'], 'prompt': row['prompt'],
                                       'schema': row['schema'], 'token_budget': stage['fixed']['final_tokens']}
            original = read(row['source_outcome'])
            original_call, = [c for c in read(Path(original['attempt']) / 'calls.json') if c['kind'] == 'answer']
            intervals = [value for sequence in call['decode'] for value in sequence['inter_token_seconds']]
            gold = labels[row['task_id']]['answer']
            queries.append({'task_id': row['task_id'], 'status': result['status'],
                'failure': result.get('failure'), 'answer_only': answer_metrics(result, gold, upstream),
                'original': answer_metrics(original['primary'], gold, upstream),
                'receiver_call_seconds': call['elapsed_seconds'],
                'receiver_wrapper_seconds': result['elapsed_seconds'],
                'original_receiver_call_seconds': original_call['elapsed_seconds'],
                'generated_tokens': sum(call['result']['output_tokens']),
                'original_generated_tokens': sum(original_call['result']['output_tokens']),
                'peak_allocated_bytes': call['peak_allocated_bytes'],
                'maximum_itl_seconds': max(intervals, default=None),
                'intervals_over_100ms': sum(value >= .1 for value in intervals)})
            sessions.add(outcome['session'])
        startup = []
        for name in sorted(sessions):
            session = Path(name)
            warmup = [read(session / 'warmup' / f'{index:03d}' / 'outcome.json')['result']
                      for index in range(len(tasks))]
            assert [row['task_id'] for row in warmup] == identities
            startup.append({'session': name, 'load_seconds': read(session / 'backend.json')['load_seconds'],
                            'warmup_tasks': len(warmup),
                            'warmup_seconds': sum(row['elapsed_seconds'] for row in warmup)})
        metrics = {mode: {metric: statistics.mean(row[mode][metric] for row in queries)
                         for metric in ['exact_match', 'token_f1']}
                   for mode in ['answer_only', 'original']}
        timing = {key: sum(row[key] for row in queries)
                  for key in ['receiver_call_seconds', 'receiver_wrapper_seconds',
                              'original_receiver_call_seconds', 'generated_tokens', 'original_generated_tokens']}
        delta = np.array([row['answer_only']['exact_match'] - row['original']['exact_match'] for row in queries])
        conditions = {
            'exact_match_preserved': metrics['answer_only']['exact_match'] >= metrics['original']['exact_match'],
            'token_f1_preserved': metrics['answer_only']['token_f1'] >= metrics['original']['token_f1'],
            'receiver_faster': timing['receiver_call_seconds'] < timing['original_receiver_call_seconds']}
        methods[arm] = {'queries': queries, 'metrics': metrics, **timing,
            'completed': sum(row['status'] == 'completed' for row in queries),
            'failures': dict(Counter(row['failure'] for row in queries if row['status'] != 'completed')),
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in queries),
            'intervals_over_100ms': sum(row['intervals_over_100ms'] for row in queries),
            'paired_exact_match_difference': float(delta.mean()),
            'paired_exact_match_interval95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
            'conditions': conditions, 'promoted': all(conditions.values()), 'startup': startup}
    report = {'task_count': len(tasks), 'methods': methods,
              'promoted': all(row['promoted'] for row in methods.values()),
              'scope': 'Receiver-only development comparison on identical recorded evidence. Original call timings are historical measurements. No reconstructed end-to-end speedup.',
              'decision_scope': 'Predeclared point estimates on eight development tasks. Passing permits a complete workflow rerun, not heldout evaluation or a claim of quality equivalence.'}
    write(Path(stage['output']) / 'evaluation.json', report)
    print({arm: {key: row[key] for key in ['metrics', 'receiver_call_seconds',
                                        'original_receiver_call_seconds', 'conditions']}
           for arm, row in methods.items()})
    print({'promoted': report['promoted']})


if __name__ == '__main__':
    main()
