import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from data.evaluate_bright import ranking_metrics
from methods.evidence_interfaces.inputs import read, write
from jev_spawn.infra.prompts import resolve_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = resolve_prompts(read(args.stage))
    fixed = stage['fixed']
    rows = [json.loads(line) for line in Path(stage['inputs']).read_text().splitlines()]
    task_ids = [row['task_id'] for row in rows]
    assert read(args.output / 'completed.json') == {'task_ids': task_ids, 'arms': stage['arms']}
    outcomes = {arm: [read(args.output / 'measured' / arm / f'{i:03d}' / 'outcome.json')
                      for i in range(len(rows))] for arm in stage['arms']}
    gold = {row['task_id']: row['relevant_document_ids'] for row in
            [json.loads(line) for line in Path(stage['relevance']).read_text().splitlines()]}
    report = {}
    for arm, values in outcomes.items():
        queries = []
        for row, value in zip(rows, values):
            assert value['task_id'] == row['task_id']
            identifiers = [document['document_id'] for document in row['documents']]
            assert value['document_ids'] == identifiers
            assert len(value['ranking']) == len(identifiers) and set(value['ranking']) == set(identifiers)
            assert sum(value['batch_sizes']) == value['receiver_calls'] == fixed['documents_per_query']
            assert max(value['batch_sizes']) <= fixed['batch_size']
            assert len(value['input_tokens']) == fixed['documents_per_query']
            queries.append({**value, **ranking_metrics(value['ranking'], gold[row['task_id']], fixed['ranking_cutoff'])})
        report[arm] = {'queries': queries, 'ndcg': statistics.mean(row['ndcg'] for row in queries),
                       'elapsed_seconds': sum(row['elapsed_seconds'] for row in queries),
                       'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in queries),
                       'input_tokens': sum(sum(row['input_tokens']) for row in queries)}
    delta = np.array([b['ndcg'] - a['ndcg'] for a, b in
                      zip(report['winner_only']['queries'], report['full_posterior']['queries'])])
    rng = np.random.default_rng(stage['evaluation']['seed'])
    samples = delta[rng.integers(len(delta), size=(stage['evaluation']['bootstrap_samples'], len(delta)))].mean(axis=1)
    paired = {'ndcg_difference': float(delta.mean()), 'per_query': delta.tolist(),
              'percentile_interval_95': np.quantile(samples, [.025, .975]).tolist(),
              'improved': int((delta > 0).sum()), 'worsened': int((delta < 0).sum()),
              'tied': int((delta == 0).sum())}
    result = {'scope': 'Frozen-worker receiver communication ablation. Costs exclude upstream planning and evidence workers.',
              'methods': report, 'paired': paired, 'supports_complete_workflow_test': bool(delta.mean() > 0)}
    write(args.output / 'evaluation.json', result)
    print(json.dumps({'methods': {arm: {k: v for k, v in entry.items() if k != 'queries'}
                                  for arm, entry in report.items()}, 'paired': paired}))


if __name__ == '__main__':
    main()
