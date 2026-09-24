import argparse
import json
from pathlib import Path
import statistics

from data.evaluate_bright import ranking_metrics
from methods.evidence_interfaces.interfaces import compact, evidence_view
from methods.evidence_interfaces.inputs import inputs, read, write
from jev_spawn.infra.prompts import resolve_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = resolve_prompts(read(args.settings))
    stage, rows = inputs(settings)
    assigned = [row['task_id'] for row in rows]
    outcomes = {}
    for arm in settings['arms']:
        assert read(args.output / arm / 'completed.json')['task_ids'] == assigned
        outcomes[arm] = [read(args.output / arm / 'measured' / f'{i:03d}' / 'outcome.json') for i in range(len(rows))]
        assert [item['primary']['task_id'] for item in outcomes[arm]] == assigned
    gold = {row['task_id']: row['relevant_document_ids']
            for row in [json.loads(line) for line in Path(stage['data']['relevance']).read_text().splitlines()]}
    assert set(gold) == set(assigned)
    retrieval = [{'task_id': row['task_id'], **ranking_metrics(
        [doc['document_id'] for doc in row['candidates']], gold[row['task_id']], stage['fixed']['ranking_cutoff'])}
        for row in rows]
    report = {}
    for arm, values in outcomes.items():
        measured = []
        for row, outcome in zip(rows, values):
            primary = outcome['primary']
            candidates = {doc['document_id'] for doc in row['candidates']}
            metrics = {}
            for name, result in {'primary': primary, **outcome['interventions']}.items():
                ranked = result['ranking']
                assert len(set(ranked)) == len(ranked) and set(ranked) <= candidates
                assert result['status'] in {'completed', 'failed'}
                assert len(ranked) == (settings['ranking_length'] if result['status'] == 'completed' else 0)
                metrics[name] = ranking_metrics(ranked, gold[row['task_id']], stage['fixed']['ranking_cutoff'])
            calls = [json.loads(line) for line in (Path(outcome['attempt']) / 'calls.jsonl').read_text().splitlines()]
            intervals = [value for call in calls if 'decode' in call
                         for sequence in call['decode'] for value in sequence['inter_token_seconds']]
            generated = [call for call in calls if 'decode' in call]
            dependency_depth = sum(index == 0 or call['kind'] != calls[index - 1]['kind']
                                   for index, call in enumerate(calls))
            agent_invocations = primary['logical_workers'] + sum(
                call['result']['batch_size'] for call in calls if call['kind'] != 'worker')
            input_lengths = [length for call in calls for length in
                             ([node['input_tokens'] for group in call['result']['groups'] for node in group]
                              if 'groups' in call['result'] else call['result']['input_tokens'])]
            assert max(input_lengths) <= stage['fixed']['context_tokens']
            assert all(call['result']['batch_size'] <= stage['fixed']['model_batch_capacity'] for call in generated)
            for call in calls:
                if 'peak_field_concurrency' in call['result']:
                    assert call['result']['peak_field_concurrency'] <= stage['fixed']['branch_tile_capacity']
            terminal = primary['terminal']
            message_bytes = (len(compact(evidence_view(terminal['definitions'], terminal['messages'])).encode())
                             if terminal is not None else None)
            measured.append({'task_id': row['task_id'], 'metrics': metrics, 'status': primary['status'],
                             'elapsed_seconds': primary['elapsed_seconds'], 'workers': primary['logical_workers'],
                             'depth': primary['worker_depth'], 'peak_allocated_bytes': primary['peak_allocated_bytes'],
                             'model_dependency_depth': dependency_depth, 'agent_invocations': agent_invocations,
                             'peak_reserved_bytes': primary['peak_reserved_bytes'],
                             'generated_tokens': sum(sum(call['result']['output_tokens']) for call in generated),
                             'minimum_input_tokens': min(input_lengths), 'maximum_input_tokens': max(input_lengths),
                             'terminal_evidence_bytes': message_bytes,
                             'actual_generation_batch_sizes': [call['result']['batch_size'] for call in generated],
                             'max_itl_seconds': max(intervals) if intervals else None,
                             'intervals_over_100ms': sum(value >= 0.1 for value in intervals),
                             'intervention_seconds': sum(result['elapsed_seconds'] for result in outcome['interventions'].values())})
        report[arm] = {'queries': measured, 'ndcg': statistics.mean(item['metrics']['primary']['ndcg'] for item in measured),
                       'valid': sum(item['status'] == 'completed' for item in measured),
                       'elapsed_seconds': sum(item['elapsed_seconds'] for item in measured),
                       'peak_allocated_bytes': max(item['peak_allocated_bytes'] for item in measured),
                       'generated_tokens': sum(item['generated_tokens'] for item in measured),
                       'intervals_over_100ms': sum(item['intervals_over_100ms'] for item in measured)}
    differences = []
    for item in report['streamed']['queries']:
        if item['status'] == 'completed':
            differences.append(item['metrics']['primary']['ndcg'] - item['metrics']['no_values']['ndcg'])
        else:
            differences.append(0)
    primary = report['streamed']
    promotion = {'quality_preserved': primary['ndcg'] >= report['direct']['ndcg'],
                 'faster_than_json': primary['elapsed_seconds'] < report['json']['elapsed_seconds'],
                 'less_memory_than_json': primary['peak_allocated_bytes'] < report['json']['peak_allocated_bytes'],
                 'positive_message_utility': statistics.mean(differences) > 0}
    useful_refinement = [item['task_id'] for item in primary['queries'] if item['status'] == 'completed'
                         and item['depth'] > 1 and item['metrics']['primary']['ndcg'] > item['metrics']['first_round_only']['ndcg']]
    write(args.output / 'evaluation.json', {'methods': report, 'promotion': promotion,
          'promoted': all(promotion.values()), 'paired_message_utility': differences,
          'mean_message_utility': statistics.mean(differences), 'useful_refinement_cases': useful_refinement,
          'hierarchy_supported': bool(useful_refinement),
          'supplied_retrieval_order': {'queries': retrieval, 'ndcg': statistics.mean(row['ndcg'] for row in retrieval),
          'scope': 'Quality of the already supplied BM25 candidate order. This requires no additional model computation and does not measure corpus retrieval latency.'}})
    print(json.dumps({'promotion': promotion, 'promoted': all(promotion.values()),
                      'methods': {arm: {key: value for key, value in entry.items() if key != 'queries'}
                                  for arm, entry in report.items()}}), flush=True)


if __name__ == '__main__':
    main()
