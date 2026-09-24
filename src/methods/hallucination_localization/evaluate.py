import argparse
from collections import Counter
from pathlib import Path
import statistics

import numpy as np

from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.hallucination_localization.analysis import overlap, validate_tree
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def ratios(tp, fp, fn):
    return {'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def summarize(rows):
    character = ratios(*(sum(row['overlap'][key] for row in rows) for key in ['tp', 'fp', 'fn']))
    response = ratios(sum(row['overlap']['predicted_present'] is True and row['overlap']['gold_present'] for row in rows),
                      sum(row['overlap']['predicted_present'] is True and not row['overlap']['gold_present'] for row in rows),
                      sum(row['overlap']['predicted_present'] is not True and row['overlap']['gold_present'] for row in rows))
    return {'character': character, 'response_presence': response,
            'mean_response_character_f1': statistics.mean(row['overlap']['f1'] for row in rows),
            'response_accuracy': statistics.mean(row['overlap']['response_correct'] for row in rows),
            'completed': sum(row['status'] == 'completed' for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    tasks = read_jsonl(stage['data']['tasks'])
    labels = {row['task_id']: row['labels'] for row in read_jsonl(stage['data']['labels'])}
    identities = [row['task_id'] for row in tasks]
    assert len(tasks) == stage['data']['task_count'] and set(identities) == set(labels)
    prompts, operators = load_prompt(stage['prompts']), read(stage['operators'])
    methods = {}
    for arm in stage['arms']:
        directory = Path(stage['run']) / arm
        assert read(directory / 'completed.json') == {'task_ids': identities, 'arm': arm}
        protocol = read(directory / 'protocol.json')
        assert protocol == {'fixed': stage['fixed'], 'data': stage['data'], 'arm': arm,
                            'task_ids': identities, 'prompts': prompts, 'operators': operators,
                            'native': resolve_prompts(read(stage['native_config']))}
        rows, sessions = [], set()
        for index, task in enumerate(tasks):
            outcome = read(directory / 'measured' / f'{index:03d}' / 'outcome.json')
            result = outcome['result']
            assert result['task_id'] == task['task_id']
            calls = read(Path(outcome['attempt']) / 'calls.json')
            tree = validate_tree(task, result, calls, prompts, operators) if arm != 'direct' and result['status'] == 'completed' else None
            intervals = [value for call in calls if 'decode' in call
                         for sequence in call['decode'] for value in sequence['inter_token_seconds']]
            rows.append({'task_id': task['task_id'], 'task_type': task['task_type'], 'status': result['status'],
                'failure': result.get('failure'), 'overlap': overlap(result, labels[task['task_id']], len(task['response'])),
                'elapsed_seconds': result['elapsed_seconds'], 'peak_allocated_bytes': result['peak_allocated_bytes'],
                'workers': result['worker_invocations'], 'tree': tree,
                'model_batches': len(calls),
                'generated_tokens': sum(sum(call['result']['output_tokens']) for call in calls if 'decode' in call),
                'maximum_itl_seconds': max(intervals, default=None),
                'intervals_over_100ms': sum(value >= .1 for value in intervals),
                'component_seconds': {kind: sum(call['elapsed_seconds'] for call in calls if call['kind'] == kind)
                                      for kind in ['planner', 'worker', 'answer']}})
            sessions.add(outcome['session'])
        startup = []
        for name in sorted(sessions):
            session = Path(name)
            warmup = [read(session / 'warmup' / f'{index:03d}' / 'outcome.json')['result'] for index in range(len(tasks))]
            assert [row['task_id'] for row in warmup] == identities
            startup.append({'session': name, 'load_seconds': read(session / 'backend.json')['load_seconds'],
                            'warmup_tasks': len(warmup), 'warmup_seconds': sum(row['elapsed_seconds'] for row in warmup)})
        methods[arm] = {'queries': rows, **summarize(rows), 'elapsed_seconds': sum(row['elapsed_seconds'] for row in rows),
            'peak_allocated_bytes': max((row['peak_allocated_bytes'] for row in rows if row['peak_allocated_bytes'] is not None), default=None),
            'mean_workers': statistics.mean(row['workers'] for row in rows), 'maximum_workers': max(row['workers'] for row in rows),
            'maximum_depth': max((node['tree']['maximum_depth'] for node in rows if node['tree']), default=0),
            'initial_spans': sum(row['tree']['initial_spans'] for row in rows if row['tree']),
            'refinements': sum(row['tree']['refinements'] for row in rows if row['tree']),
            'component_seconds': {kind: sum(row['component_seconds'][kind] for row in rows) for kind in ['planner', 'worker', 'answer']},
            'intervals_over_100ms': sum(row['intervals_over_100ms'] for row in rows),
            'failures': dict(Counter(row['failure'] for row in rows if row['status'] != 'completed')),
            'startup': startup, 'by_type': {kind: summarize([row for row in rows if row['task_type'] == kind])
                                          for kind in sorted({row['task_type'] for row in rows})}}
    ours, serial, direct = [methods[arm] for arm in ['streamed', 'json', 'direct']]
    quality = all(ours['character']['f1'] is not None and target['character']['f1'] is not None
                  and ours['character']['f1'] >= target['character']['f1']
                  and ours['mean_response_character_f1'] >= target['mean_response_character_f1']
                  for target in [serial, direct])
    conditions = {'all_tasks_valid': all(method['completed'] == len(tasks) for method in methods.values()),
                  'quality_preserved': quality,
                  'faster_than_json': ours['elapsed_seconds'] < serial['elapsed_seconds'],
                  'less_memory_than_json': ours['peak_allocated_bytes'] < serial['peak_allocated_bytes'],
                  'faster_than_direct': ours['elapsed_seconds'] < direct['elapsed_seconds']}
    rng = np.random.default_rng(stage['evaluation']['seed'])
    indices = rng.integers(len(tasks), size=(stage['evaluation']['bootstrap_samples'], len(tasks)))
    comparisons = {}
    for arm in ['json', 'direct', 'tiled_independent']:
        delta = np.array([a['overlap']['f1'] - b['overlap']['f1'] for a, b in zip(ours['queries'], methods[arm]['queries'])])
        sampled_scores = []
        for method in [ours, methods[arm]]:
            counts = np.array([[row['overlap'][key] for key in ['tp', 'fp', 'fn']] for row in method['queries']])
            pooled = counts[indices].sum(axis=1)
            denominator = 2 * pooled[:, 0] + pooled[:, 1] + pooled[:, 2]
            scores = np.full(len(indices), np.nan)
            np.divide(2 * pooled[:, 0], denominator, out=scores, where=denominator > 0)
            sampled_scores.append(scores)
        pooled_difference = sampled_scores[0] - sampled_scores[1]
        defined = np.isfinite(pooled_difference)
        ours_times = np.array([row['elapsed_seconds'] for row in ours['queries']])
        other_times = np.array([row['elapsed_seconds'] for row in methods[arm]['queries']])
        speedups = other_times[indices].sum(axis=1) / ours_times[indices].sum(axis=1)
        comparisons[arm] = {'mean_response_f1_difference': float(delta.mean()),
                            'interval95': np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist(),
                            'pooled_character_f1_difference': ours['character']['f1'] - methods[arm]['character']['f1'],
                            'pooled_character_interval95': np.quantile(pooled_difference[defined], [.025, .975]).tolist() if defined.any() else None,
                            'undefined_pooled_resamples': int((~defined).sum()),
                            'complete_speedup': float(other_times.sum() / ours_times.sum()),
                            'complete_speedup_interval95': np.quantile(speedups, [.025, .975]).tolist()}
    report = {'task_count': len(tasks), 'methods': methods, 'promotion': conditions, 'promoted': all(conditions.values()),
              'paired_comparisons': comparisons,
              'failure_semantics': 'Invalid outputs receive zero per-response F1 and incorrect response accuracy. They provide no predicted character positions and contribute all gold characters as false negatives. Invalid negative cases do not create hallucination true positives or false positives. All-task validity is mandatory for promotion.',
              'decision_scope': 'Development screening on predeclared point estimates. No equivalence or superiority claim from small-sample confidence intervals.'}
    report['uncertainty'] = 'Paired response bootstrap with the same source identities for all methods. Pooled character F1 resamples union-overlap counts and reports undefined zero-denominator replicates separately. Speedup intervals capture variation across responses under one timing realization, not repeated execution noise.'
    write(Path(stage['run']) / 'evaluation.json', report)
    print({arm: {key: method[key] for key in ['character', 'mean_response_character_f1', 'completed', 'elapsed_seconds', 'mean_workers']}
           for arm, method in methods.items()})
    print(conditions)


if __name__ == '__main__':
    main()
