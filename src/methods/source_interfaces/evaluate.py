import argparse
from collections import Counter
import importlib.util
from pathlib import Path
import statistics

import numpy as np

from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from methods.evidence_qa.evaluate import answer_metrics
from methods.source_interfaces.analysis import categories, hierarchy_statistics, paired_comparisons
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def evaluate(validate_hierarchy, source_ids):
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    tasks = read_jsonl(stage['data']['tasks'])
    identities = [row['task_id'] for row in tasks]
    outcomes = {}
    sessions = {}
    for arm in stage['arms']:
        protocol = read(args.output / arm / 'protocol.json')
        assert protocol['fixed'] == stage['fixed'] and protocol['data'] == stage['data']
        assert protocol['prompts'] == load_prompt(stage['prompts'])
        assert protocol['plan_schema'] == read(stage['plan_schema'])
        assert read(args.output / arm / 'completed.json') == {'task_ids': identities, 'arm': arm}
        outcomes[arm] = [read(args.output / arm / 'measured' / f'{index:03d}' / 'outcome.json')
                         for index in range(len(tasks))]
        assert [row['primary']['task_id'] for row in outcomes[arm]] == identities
        sessions[arm] = []
        for name in sorted({row['session'] for row in outcomes[arm]}):
            session = Path(name)
            warmup = [read(session / 'warmup' / f'{index:03d}' / 'outcome.json')['primary']
                      for index in range(len(tasks))]
            assert [row['task_id'] for row in warmup] == identities
            backend = read(session / 'backend.json')
            sessions[arm].append({'session': name, 'load_seconds': backend['load_seconds'],
                                  'index_seconds': backend['index_seconds'], 'warmup_tasks': len(warmup),
                                  'warmup_seconds': sum(row['elapsed_seconds'] for row in warmup)})
    labels = {row['task_id']: row for row in read_jsonl(stage['data']['labels'])}
    assert set(labels) == set(identities)
    spec = importlib.util.spec_from_file_location('multihoprag_qa', stage['evaluation']['upstream'])
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    methods = {}
    for arm, records in outcomes.items():
        queries = []
        for outcome in records:
            result = outcome['primary']
            gold = labels[result['task_id']]
            calls = read(Path(outcome['attempt']) / 'calls.json')
            units = {unit['id']: unit for unit in result['sources']}
            assert set(result['evidence']) <= set(units)
            selected = set(source_ids(result))
            if arm == 'direct':
                selected = set(units)
            assert selected <= set(units)
            topology = None
            if arm != 'direct' and result['status'] == 'completed':
                topology = validate_hierarchy(result, calls)
            scores = {name: answer_metrics(value, gold['answer'], upstream)
                      for name, value in {'primary': result, **outcome['interventions']}.items()}
            generated = [call for call in calls if 'decode' in call]
            intervals = [value for call in generated for sequence in call['decode'] for value in sequence['inter_token_seconds']]
            evidence = set(gold['evidence_document_ids'])
            retrieved_documents = {unit['doc_id'] for unit in units.values()}
            selected_documents = {units[identity]['doc_id'] for identity in selected}
            cited_documents = {units[identity]['doc_id'] for identity in result['evidence']}
            timings = {kind: sum(call['elapsed_seconds'] for call in calls if call['kind'] == kind)
                       for kind in ['planner', 'worker', 'answer']}
            queries.append({'task_id': result['task_id'], 'question_type': gold['question_type'],
                'status': result['status'], 'failure': result.get('failure'), 'scores': scores,
                'elapsed_seconds': result['elapsed_seconds'], 'component_seconds': timings,
                'peak_allocated_bytes': result['peak_allocated_bytes'], 'peak_reserved_bytes': result['peak_reserved_bytes'],
                'workers': result['worker_invocations'], 'model_invocations': result['model_invocations'],
                'dependency_depth': result['model_dependency_depth'], 'levels': len(result['levels']),
                'topology': topology,
                'generated_tokens': sum(sum(call['result']['output_tokens']) for call in generated),
                'maximum_itl_seconds': max(intervals) if intervals else None,
                'intervals_over_100ms': sum(value >= .1 for value in intervals),
                'retrieved_document_recall': len(evidence & retrieved_documents) / len(evidence) if evidence else None,
                'selected_document_recall': len(evidence & selected_documents) / len(evidence) if evidence else None,
                'cited_document_recall': len(evidence & cited_documents) / len(evidence) if evidence else None})
        methods[arm] = {'queries': queries, 'completed': sum(row['status'] == 'completed' for row in queries),
            'exact_match': statistics.mean(row['scores']['primary']['exact_match'] for row in queries),
            'token_f1': statistics.mean(row['scores']['primary']['token_f1'] for row in queries),
            'elapsed_seconds': sum(row['elapsed_seconds'] for row in queries),
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in queries),
            'mean_workers': statistics.mean(row['workers'] for row in queries),
            'maximum_workers': max(row['workers'] for row in queries),
            'maximum_dependency_depth': max((row['dependency_depth'] for row in queries if row['dependency_depth'] is not None), default=None),
            'intervals_over_100ms': sum(row['intervals_over_100ms'] for row in queries),
            'startup_sessions': sessions[arm],
            'by_question_type': categories(queries),
            'component_seconds': {kind: sum(row['component_seconds'][kind] for row in queries)
                                  for kind in ['planner', 'worker', 'answer']},
            'failures': dict(Counter(row['failure'] for row in queries if row['status'] != 'completed'))}
    ours, direct, serial = [methods[arm] for arm in ['streamed', 'direct', 'json']]
    observed = [row for row in ours['queries'] if 'no_references' in row['scores']]
    utility = np.array([row['scores']['primary']['exact_match'] - row['scores']['no_references']['exact_match']
                        for row in observed])
    delta = np.array([a['scores']['primary']['exact_match'] - b['scores']['primary']['exact_match']
                      for a,b in zip(ours['queries'],direct['queries'])])
    rng = np.random.default_rng(stage['evaluation']['seed'])
    indices = rng.integers(len(tasks), size=(stage['evaluation']['bootstrap_samples'], len(tasks)))
    utility_mean, utility_interval = None, None
    if len(utility):
        utility_indices = np.random.default_rng(stage['evaluation']['seed']).integers(
            len(utility), size=(stage['evaluation']['bootstrap_samples'], len(utility)))
        utility_mean = float(utility.mean())
        utility_interval = np.quantile(utility[utility_indices].mean(axis=1), [.025, .975]).tolist()
    direct_dominates = (direct['exact_match'] >= ours['exact_match']
        and direct['elapsed_seconds'] <= ours['elapsed_seconds']
        and direct['peak_allocated_bytes'] <= ours['peak_allocated_bytes']
        and (direct['exact_match'] > ours['exact_match'] or direct['elapsed_seconds'] < ours['elapsed_seconds']
             or direct['peak_allocated_bytes'] < ours['peak_allocated_bytes']))
    conditions = {'quality_preserved': (ours['exact_match'] >= direct['exact_match']
                                       and ours['token_f1'] >= direct['token_f1']),
                  'faster_than_json': ours['elapsed_seconds'] < serial['elapsed_seconds'],
                  'less_memory_than_json': ours['peak_allocated_bytes'] < serial['peak_allocated_bytes'],
                  'positive_reference_utility': len(observed) == len(tasks) and utility_mean > 0,
                  'not_dominated_by_direct': not direct_dominates}
    report = {'task_count': len(tasks), 'dataset_split': Path(stage['data']['tasks']).parent.name,
              'methods': methods, 'promotion': conditions, 'promoted': all(conditions.values()),
              'decision_scope': 'Predeclared point-estimate screening conditions, not statistical evidence of equivalence or superiority.',
              'paired_comparisons': paired_comparisons(methods, indices),
              'mean_reference_utility': utility_mean,
              'reference_utility_interval95': utility_interval,
              'reference_intervention_tasks': [row['task_id'] for row in observed],
              'reference_utility_scope': 'Paired effect among observed interventions. Missing interventions are unobserved, not zero effects. Promotion requires intervention coverage of all assigned questions.',
              'paired_exact_match_difference': float(delta.mean()),
              'paired_exact_match_interval95': np.quantile(delta[indices].mean(axis=1), [.025,.975]).tolist()}
    return args, stage, report


def write_evaluation(args, report):
    write(args.output / 'evaluation.json', report)
    print({arm: {key: result[key] for key in ['completed', 'exact_match', 'elapsed_seconds', 'mean_workers']}
           for arm, result in report['methods'].items()})
    print(report['promotion'])


if __name__ == '__main__':
    args, _, report = evaluate(hierarchy_statistics, lambda result: [value['source_id'] for value in result['references']
                                                                     if value['source_id'] is not None])
    write_evaluation(args, report)
