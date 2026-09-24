import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import jsonschema


def read(path):
    return json.loads(path.read_text())


def summarize(run):
    protocol, evaluation = read(run / 'protocol.json'), read(run / 'evaluation.json')
    validator = jsonschema.Draft202012Validator(protocol.get('plan_schema', protocol['schema']))
    report = {}
    for method, scored in evaluation['methods'].items():
        rows = [read(run / method / f'task-{index:04d}' / 'complete.json')
                for index in range(len(protocol['tasks']))]
        assert [row['task']['task_id'] for row in rows] == [row['task_id'] for row in scored['scores']]
        counts, details = [], []
        for row, score in zip(rows, scored['scores']):
            records = row['records']
            requests = sum(record['result'].get('logical_field_count', record['result']['batch_size'])
                           for record in records)
            counts.append(requests)
            intervals = [value * 1000 for record in records for decoding in record.get('decode', [])
                         for value in decoding['inter_token_seconds']]
            item = {'task_id': row['task']['task_id'], 'status': row['status'],
                    'final_contract_valid': score['valid'], 'correct': score['correct'],
                    'completed_model_requests': requests, 'compute_seconds': row['compute_wall_seconds'],
                    'peak_allocated_bytes': row['profile']['peak_allocated_bytes'],
                    'max_itl_ms': max(intervals) if intervals else None,
                    'intervals_at_least_100ms': sum(value >= 100 for value in intervals)}
            if method == 'graph':
                nodes = row['flow']['nodes']
                operations = {event['metrics']['call_index']: event['kind']
                              for event in row['flow']['events'] if event['event'] == 'batch'
                              and 'call_index' in event['metrics']}
                operations[0] = 'expand'
                operation_seconds = Counter()
                for index, record in enumerate(records):
                    operation_seconds[operations.get(index, 'rejected_before_batch_event')] += record['elapsed_seconds']
                item['model_seconds_by_operation'] = dict(operation_seconds)
                item['non_model_compute_seconds'] = row['compute_wall_seconds'] - sum(operation_seconds.values())
                item['instantiated_model_nodes'] = sum(node['kind'] in {'decide', 'generate', 'generate_json', 'expand'}
                                                       for node in nodes.values())
                item['completed_model_lifecycles'] = sum(
                    node['kind'] in {'decide', 'generate', 'generate_json', 'expand'} and node['status'] == 'completed'
                    for node in nodes.values())
                item['host_nodes'] = sum(node['kind'] in {'map', 'reduce', 'collect'} for node in nodes.values())
                item['nodes_by_kind'] = dict(Counter(node['kind'] for node in nodes.values()))
                item['max_spawn_depth'] = max(node['depth'] for node in nodes.values())
                try:
                    initial_program = json.loads(records[0]['result']['texts'][0])
                    if protocol['settings'].get('plan_language') == 'grounded':
                        validator = jsonschema.Draft202012Validator(
                            protocol['plan_schema_by_task'][row['task']['task_id']])
                    validator.validate(initial_program)
                except (ValueError, jsonschema.ValidationError) as error:
                    item.update(initial_program_schema_valid=False,
                                initial_program_error=type(error).__name__ + ': ' + str(error).splitlines()[0])
                else:
                    item['initial_program_schema_valid'] = True
                initial_nodes = 2 if protocol['settings'].get('finalization') == 'generate_json' else 1
                item['initial_program_instantiated'] = len(nodes) > initial_nodes
            item['failure'] = row.get('error', score.get('error'))
            details.append(item)
        report[method] = {'assigned_tasks': len(rows), 'valid': scored['valid'], 'correct': scored['correct'],
                          'mean_completed_model_requests_per_assigned_task': statistics.mean(counts),
                          'maximum_completed_model_requests_per_task': max(counts),
                          'complete_workload_compute_seconds': sum(row['compute_wall_seconds'] for row in rows),
                          'tasks': details}
    result = {'run': str(run), 'settings': protocol['settings'], 'methods': report,
              'interpretation': 'All assigned tasks remain in the denominator. Model requests include the coordinator and backend outputs rejected later by graph validation. Host nodes are counted separately. Timings of failed or invalid answers do not establish task speedup. No timing ratio is computed by selecting only mutually correct tasks.'}
    graph, direct = report['graph'], report['direct']
    result['development_qualification'] = {
        'all_graph_final_contracts_valid': graph['valid'] == graph['assigned_tasks'],
        'graph_correct_count_at_least_direct': graph['correct'] >= direct['correct'],
        'passes_declared_interface_and_quality_checks': (
            graph['valid'] == graph['assigned_tasks'] and graph['correct'] >= direct['correct']),
        'graph_complete_workload_faster': (
            graph['complete_workload_compute_seconds'] < direct['complete_workload_compute_seconds']),
        'graph_all_observed_itl_below_100ms': all(
            row['intervals_at_least_100ms'] == 0 for row in graph['tasks']),
        'scope': 'Eight development tasks cannot establish statistical quality preservation. Passing the interface checks does not establish useful massive fanout or end-to-end acceleration.'}
    if 'no_messages' in report:
        control = report['no_messages']
        result['message_utility'] = {
            'control': protocol['message_control'],
            'graph_correct': graph['correct'], 'control_correct': control['correct'],
            'paired_correctness': [
                {'task_id': full['task_id'], 'graph_correct': full['correct'],
                 'control_correct': empty['correct']}
                for full, empty in zip(graph['tasks'], control['tasks'])],
            'graph_correct_count_at_least_control': graph['correct'] >= control['correct'],
            'graph_complete_workload_faster_than_control': (
                graph['complete_workload_compute_seconds'] < control['complete_workload_compute_seconds'])}
    (run / 'qualification_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    for run in args.runs:
        result = summarize(run)
        print(json.dumps({'run': str(run), 'methods': {
            method: {key: value for key, value in row.items() if key != 'tasks'}
            for method, row in result['methods'].items()}}))


if __name__ == '__main__':
    main()
