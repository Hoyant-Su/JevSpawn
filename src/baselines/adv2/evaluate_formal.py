import argparse
import json
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


def interval(values, confidence):
    tail = (1 - confidence) / 2
    return np.quantile(values, [tail, 1 - tail]).tolist()


def cluster_totals(values, groups, count):
    return np.bincount(groups, weights=np.asarray(values, dtype=float), minlength=count)


def binary_quality(accepted, correct, groups, sampled, confidence):
    count = int(groups.max()) + 1
    matches = accepted == correct
    false_accept = accepted & ~correct
    false_reject = ~accepted & correct
    totals = {name: cluster_totals(values, groups, count) for name, values in {
        'n': np.ones(len(correct)), 'correct': correct, 'flawed': ~correct,
        'matches': matches, 'false_accept': false_accept, 'false_reject': false_reject}.items()}
    resampled = {name: values[sampled].sum(axis=1) for name, values in totals.items()}
    assert np.all(resampled['correct'] > 0) and np.all(resampled['flawed'] > 0)
    metrics = {
        'accuracy': (matches.mean(), resampled['matches'] / resampled['n']),
        'false_accept_rate': (false_accept.sum() / (~correct).sum(),
                              resampled['false_accept'] / resampled['flawed']),
        'false_reject_rate': (false_reject.sum() / correct.sum(),
                              resampled['false_reject'] / resampled['correct']),
    }
    metrics['balanced_accuracy'] = (
        1 - (metrics['false_accept_rate'][0] + metrics['false_reject_rate'][0]) / 2,
        1 - (metrics['false_accept_rate'][1] + metrics['false_reject_rate'][1]) / 2)
    return {'accepted': int(accepted.sum()), 'false_accepted': int(false_accept.sum()),
            'false_rejected': int(false_reject.sum()),
            **{name: {'estimate': float(point), 'ci': interval(values, confidence)}
               for name, (point, values) in metrics.items()}}


def main():
    parser = argparse.ArgumentParser()
    for name in ['run', 'labels', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--methods', nargs='+', required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--task-count', type=int, required=True)
    parser.add_argument('--quality-repeat', type=int, required=True)
    parser.add_argument('--bootstrap-replicates', type=int, required=True)
    parser.add_argument('--bootstrap-seed', type=int, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    args = parser.parse_args()
    assert 0 <= args.quality_repeat < args.repeats
    labels = [json.loads(line) for line in args.labels.read_text().splitlines()]
    assert len(labels) == args.task_count
    retrieval = [json.loads(p.read_text()) for p in sorted((args.run / 'retrieval').glob('batch-*.json'))]
    retrieved_ids = [i for r in retrieval for i in r['task_ids']]
    assert len(retrieved_ids) == len(labels) and set(retrieved_ids) == {r['task_id'] for r in labels}
    statuses = {task_id: status for r in retrieval for task_id, status in zip(r['task_ids'], r['status'])}
    available_ids = {task_id for task_id, status in statuses.items() if status['available']}
    unavailable = [dict(task_id=task_id, **status) for task_id, status in statuses.items() if not status['available']]
    full_cluster_count = len({r['problem_group'] for r in labels})
    labels = [row for row in labels if row['task_id'] in available_ids]
    assert labels
    ids = [row['task_id'] for row in labels]
    group_ids = {group: i for i, group in enumerate(sorted({row['problem_group'] for row in labels}))}
    groups = np.asarray([group_ids[row['problem_group']] for row in labels])
    assert set(groups) == set(range(len(set(groups))))
    correct = np.asarray([row['acceptable'] for row in labels], dtype=bool)
    count = len(set(groups))
    sampled = np.random.default_rng(args.bootstrap_seed).integers(
        count, size=(args.bootstrap_replicates, count))
    methods, predictions, indicators = {}, {}, {}
    for method in args.methods:
        trials, repeat_choices = [], []
        for repeat in range(args.repeats):
            files = sorted((args.run / 'trials').glob(f'repeat-{repeat:02d}-batch-*-{method}.json'))
            saved = [json.loads(path.read_text()) for path in files]
            rows = [row for trial in saved for row in trial['result']['rows']]
            assert len(rows) == len(ids) and {r['task_id'] for r in rows} == set(ids)
            lookup = {row['task_id']: row for row in rows}
            ordered = [lookup[task_id] for task_id in ids]
            accepted = np.asarray([row['accepted'] for row in ordered])
            repeat_choices.append(accepted)
            calls = [call['result'] for trial in saved for call in trial['result']['calls']]
            itl = [t for call in calls for row in call.get('decode', []) for t in row['inter_token_seconds']]
            trials.append({
                'repeat': repeat, 'verification_service_seconds': sum(r['elapsed_seconds'] for r in saved),
                'peak_allocated_gib_per_gpu': max(r['peak_allocated_bytes'] for r in saved) / 2**30,
                'computed_input_tokens': sum(r['computed_input_tokens'] for r in calls),
                'output_tokens': sum(sum(r['output_tokens']) for r in calls),
                'actual_bucket_batch_sizes': dict(Counter(r['batch_size'] for r in calls)),
                'actual_bucket_field_counts': dict(Counter(r['field_count'] for r in calls)),
                'indicator_decisions': sum(len(row['verdicts']) for row in ordered),
                'accuracy': float((accepted == correct).mean()),
                'itl_p95_ms': float(np.quantile(itl, .95) * 1000) if itl else None,
                'itl_max_ms': max(itl) * 1000 if itl else None,
            })
            if repeat == args.quality_repeat:
                predictions[method] = accepted
                indicators[method] = ordered
        times = [r['verification_service_seconds'] for r in trials]
        methods[method] = {
            'quality': binary_quality(predictions[method], correct, groups, sampled, args.confidence),
            'correct_decisions_over_all_requested_solutions': int((predictions[method] == correct).sum()) / args.task_count,
            'verification_service_mean_seconds': statistics.mean(times),
            'verification_service_sd_seconds': statistics.stdev(times),
            'acceptance_stable_across_repeats': all(np.array_equal(repeat_choices[0], x) for x in repeat_choices),
            'repeats': trials,
        }
    comparisons = []
    sizes = cluster_totals(np.ones(len(ids)), groups, count)[sampled].sum(axis=1)
    for left, right in combinations(args.methods, 2):
        a, b = predictions[left], predictions[right]
        difference = (b == correct).astype(int) - (a == correct).astype(int)
        bootstrap = cluster_totals(difference, groups, count)[sampled].sum(axis=1) / sizes
        matches, decisions = 0, 0
        for first, second in zip(indicators[left], indicators[right]):
            assert first['indicator_ids'] == second['indicator_ids']
            matches += sum(x == y for x, y in zip(first['verdicts'], second['verdicts']))
            decisions += len(first['verdicts'])
        comparisons.append({'reference': left, 'method': right,
                            'accuracy_difference': float(difference.mean()),
                            'accuracy_difference_ci': interval(bootstrap, args.confidence),
                            'acceptance_agreement': float((a == b).mean()),
                            'indicator_agreement': matches / decisions, 'indicator_decisions': decisions})
    retrieval_generations = [g for r in retrieval for g in r['generation']]
    retrieval_itl = [t for generation in retrieval_generations for row in generation['decode']
                     for t in row['inter_token_seconds']]
    metadata = [json.loads(p.read_text()) for p in args.run.glob('retrieve-rank-*/run.json')]
    initial_metadata = [json.loads(p.read_text()) for p in (args.run / 'recovery').glob('initial-rank-*-run.json')]
    report = {
        'requested_solutions': args.task_count, 'requested_problem_clusters': full_cluster_count,
        'available_solutions': len(ids), 'coverage': len(ids) / args.task_count,
        'retrieval_abstentions': unavailable,
        'solutions': len(ids), 'problem_clusters': count, 'first_error_free': int(correct.sum()),
        'flawed_solutions': int((~correct).sum()), 'quality_repeat': args.quality_repeat,
        'bootstrap_replicates': args.bootstrap_replicates, 'bootstrap_seed': args.bootstrap_seed,
        'confidence': args.confidence,
        'target': 'Conditional on valid retrieved indicators: accept iff the original ProcessBench first-error label is -1. Retrieval failures are unavailable decisions, not inferred rejections. This is not full-cohort ADv2 accuracy, individual-indicator correctness or repaired-answer accuracy.',
        'uncertainty': 'Paired percentile bootstrap over distinct original problem texts; retain all solution variants in every sampled problem cluster. Timing repeats do not increase quality N.',
        'timing_scope': 'Sum of isolated synchronized verification batch service times across ranks, excluding retrieval and warmup. This is not an end-to-end multi-agent latency measurement.',
        'model_residency': {'retrieval': 'Frozen Qwen3.5-4B and Qwen3-Embedding-0.6B resident together per GPU.',
                            'verification': 'Fresh process with frozen Qwen3.5-4B only; embedding model unloaded.',
                            'memory_scope': 'Maximum allocated CUDA bytes on any one rank/GPU, not summed across GPUs.'},
        'retrieval': {
            'component_service_seconds': None if any(r['component_elapsed_seconds'] is None for r in retrieval) else sum(r['component_elapsed_seconds'] for r in retrieval),
            'recorded_component_service_seconds': sum(r['component_elapsed_seconds'] for r in retrieval if r['component_elapsed_seconds'] is not None),
            'unrecorded_component_batches': [r['batch'] for r in retrieval if r['component_elapsed_seconds'] is None],
            'embedding_reconstruction_seconds': sum(r['reconstruction_seconds'] for r in retrieval if r['reconstructed_retrieval']),
            'generation_seconds': sum(g['elapsed_seconds'] for g in retrieval_generations),
            'recorded_original_embedding_query_seconds': sum(r['retrieval']['elapsed_seconds'] for r in retrieval if not r['reconstructed_retrieval']),
            'reconstructed_embedding_query_seconds': sum(r['retrieval']['elapsed_seconds'] for r in retrieval if r['reconstructed_retrieval']),
            'continuation_embedding_load_and_index_seconds': sum(r['embedding_load_and_index_seconds'] for r in metadata),
            'initial_embedding_load_and_index_seconds': sum(r['embedding_load_and_index_seconds'] for r in initial_metadata),
            'input_tokens': sum(sum(g['input_tokens']) for g in retrieval_generations),
            'output_tokens': sum(sum(g['output_tokens']) for g in retrieval_generations),
            'itl_p95_ms': float(np.quantile(retrieval_itl, .95) * 1000),
            'itl_max_ms': max(retrieval_itl) * 1000,
        }, 'methods': methods, 'paired_comparisons': comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'solutions': len(ids), 'problem_clusters': count, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
