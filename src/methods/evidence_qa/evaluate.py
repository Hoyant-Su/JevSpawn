import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import statistics
import string

import numpy as np

from methods.evidence_flow.environment import read_jsonl
from methods.evidence_interfaces.inputs import read, write
from jev_spawn.infra.prompts import resolve_prompts


def normalize(text):
    text = ''.join(character for character in text.lower() if character not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_metrics(result, gold, upstream):
    prediction = result['answer'] if result['status'] == 'completed' else ''
    predicted, target = normalize(prediction), normalize(gold)
    common = sum((Counter(predicted.split()) & Counter(target.split())).values())
    f1 = 2 * common / (len(predicted.split()) + len(target.split())) if common else 0.0
    return {'exact_match': int(result['status'] == 'completed' and predicted == target),
            'token_f1': f1,
            'upstream_token_overlap': int(upstream.has_intersection(upstream.extract_answer(prediction).lower(), gold.lower()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    rows = read_jsonl(stage['data']['tasks'])
    task_ids = [row['task_id'] for row in rows]
    outcomes = {}
    for arm in stage['arms']:
        assert read(args.output / arm / 'completed.json') == {'task_ids': task_ids, 'arm': arm}
        outcomes[arm] = [read(args.output / arm / 'measured' / f'{i:03d}' / 'outcome.json') for i in range(len(rows))]
        assert [item['primary']['task_id'] for item in outcomes[arm]] == task_ids
    specification = importlib.util.spec_from_file_location('multihoprag_upstream_qa', stage['evaluation']['upstream'])
    upstream = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(upstream)
    labels = {row['task_id']: row for row in read_jsonl(stage['data']['labels'])}
    assert set(labels) == set(task_ids)
    units = {row['id']: row for row in read_jsonl(stage['data']['units'])}
    methods = {}
    for arm, records in outcomes.items():
        queries = []
        for row, outcome in zip(rows, records):
            primary = outcome['primary']
            gold = labels[row['task_id']]
            scores = {key: answer_metrics(value, gold['answer'], upstream)
                      for key, value in {'primary': primary, **outcome['interventions']}.items()}
            calls = read(Path(outcome['attempt']) / 'calls.json')
            generated = [call for call in calls if 'decode' in call]
            intervals = [value for call in generated for sequence in call['decode'] for value in sequence['inter_token_seconds']]
            gold_documents = set(gold['evidence_document_ids'])
            retrieved = {unit['doc_id'] for unit in primary['sources']}
            cited = {units[identity]['doc_id'] for identity in primary['evidence']}
            assert set(primary['evidence']) <= {unit['id'] for unit in primary['sources']}
            record = {'task_id': row['task_id'], 'question_type': gold['question_type'],
                      'status': primary['status'], 'scores': scores, 'elapsed_seconds': primary['elapsed_seconds'],
                      'workers': primary['worker_invocations'], 'agents': primary['model_invocations'],
                      'model_dependency_depth': primary['model_dependency_depth'], 'rounds': len(primary['rounds']),
                      'peak_allocated_bytes': primary['peak_allocated_bytes'],
                      'peak_reserved_bytes': primary['peak_reserved_bytes'],
                      'generated_tokens': sum(sum(call['result']['output_tokens']) for call in generated),
                      'max_itl_seconds': max(intervals) if intervals else None,
                      'intervals_over_100ms': sum(value >= .1 for value in intervals),
                      'retrieved_document_recall': len(gold_documents & retrieved) / len(gold_documents) if gold_documents else None,
                      'cited_document_recall': len(gold_documents & cited) / len(gold_documents) if gold_documents else None}
            queries.append(record)
        categories = sorted({row['question_type'] for row in queries})
        methods[arm] = {'queries': queries, 'completed': sum(row['status'] == 'completed' for row in queries),
            'metrics': {metric: statistics.mean(row['scores']['primary'][metric] for row in queries)
                        for metric in ['exact_match', 'token_f1', 'upstream_token_overlap']},
            'per_category': {category: {metric: statistics.mean(row['scores']['primary'][metric] for row in queries if row['question_type'] == category)
                                        for metric in ['exact_match', 'token_f1', 'upstream_token_overlap']} for category in categories},
            'elapsed_seconds': sum(row['elapsed_seconds'] for row in queries),
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in queries),
            'mean_workers': statistics.mean(row['workers'] for row in queries),
            'maximum_workers': max(row['workers'] for row in queries),
            'maximum_model_dependency_depth': max(row['model_dependency_depth'] for row in queries),
            'intervals_over_100ms': sum(row['intervals_over_100ms'] for row in queries)}
    primary = methods['streamed']
    differences = [row['scores']['primary']['exact_match'] - row['scores']['no_values']['exact_match']
                   if 'no_values' in row['scores'] else 0 for row in primary['queries']]
    refinement = [row['task_id'] for row in primary['queries'] if 'first_round_only' in row['scores']
                  and row['scores']['primary']['exact_match'] > row['scores']['first_round_only']['exact_match']]
    conditions = {'quality_preserved': primary['metrics']['exact_match'] >= methods['direct']['metrics']['exact_match'],
                  'faster_than_json': primary['elapsed_seconds'] < methods['json']['elapsed_seconds'],
                  'less_memory_than_json': primary['peak_allocated_bytes'] < methods['json']['peak_allocated_bytes'],
                  'positive_message_utility': statistics.mean(differences) > 0}
    direct = methods['direct']
    dominates = (direct['metrics']['exact_match'] >= primary['metrics']['exact_match']
                 and direct['elapsed_seconds'] <= primary['elapsed_seconds']
                 and direct['peak_allocated_bytes'] <= primary['peak_allocated_bytes']
                 and (direct['metrics']['exact_match'] > primary['metrics']['exact_match']
                      or direct['elapsed_seconds'] < primary['elapsed_seconds']
                      or direct['peak_allocated_bytes'] < primary['peak_allocated_bytes']))
    conditions['not_dominated_by_direct'] = not dominates
    delta = np.array([a['scores']['primary']['exact_match'] - b['scores']['primary']['exact_match']
                      for a,b in zip(primary['queries'], methods['direct']['queries'])])
    rng = np.random.default_rng(stage['evaluation']['seed'])
    samples = delta[rng.integers(len(delta), size=(stage['evaluation']['bootstrap_samples'], len(delta)))].mean(axis=1)
    report = {'methods': methods, 'promotion': conditions, 'promoted': all(conditions.values()),
              'message_utility': differences, 'mean_message_utility': statistics.mean(differences),
              'useful_refinement_cases': refinement,
              'paired_exact_match_difference': float(delta.mean()),
              'paired_exact_match_interval95': np.quantile(samples,[.025,.975]).tolist()}
    write(args.output / 'evaluation.json', report)
    print(json.dumps({'methods': {arm: {key:value for key,value in result.items() if key != 'queries'} for arm,result in methods.items()},
                      'promotion':conditions,'promoted':report['promoted']}))


if __name__ == '__main__':
    main()
