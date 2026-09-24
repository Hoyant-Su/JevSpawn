import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median
from urllib.parse import quote


def distribution(values):
    return {'count': len(values), 'mean_seconds': mean(values), 'median_seconds': median(values),
            'minimum_seconds': min(values), 'maximum_seconds': max(values)} if values else {'count': len(values)}


def analyze(run, settings):
    traces = [json.loads(path.read_text()) for path in sorted(Path(settings['traces']).glob(settings['trace_glob']))]
    protocol = json.loads((run / 'protocol.json').read_text())
    assert {trace['task_id'] for trace in traces} == {task['task_id'] for task in protocol['tasks']}
    reports = []
    for trace in traces:
        calls = {call['node']: call for call in trace['calls'] if call['kind'] in settings['decision_kinds']}
        graph = trace['computations']
        finite = {identity: call for identity, call in calls.items() if identity in graph['ready_monotonic']}
        edges = []
        for node in graph['nodes']:
            if node['id'] in finite:
                for parent in node['depends_on']:
                    if parent in finite:
                        edges.append({'parent': parent, 'child': node['id'], 'seconds':
                            finite[node['id']]['decision']['delivered_monotonic']
                            - finite[parent]['decision']['delivered_monotonic']})
        reports.append({'task_id': trace['task_id'], 'finite_calls': len(calls),
            'queue_to_delivery': distribution([call['decision']['delivered_monotonic']
                - call['decision']['submitted_monotonic'] for call in calls.values()]),
            'node_ready_to_delivery': distribution([call['decision']['delivered_monotonic']
                - graph['ready_monotonic'][identity] for identity, call in finite.items()]),
            'finite_dependency_inter_decision': distribution([edge['seconds'] for edge in edges]),
            'finite_dependency_edges': edges,
            'unmeasured_reason': None if edges else 'No executed finite-to-finite dependency edge in this trace; no dependency IDL estimate.'})
    batches = [batch for path in sorted(run.glob(settings['batch_glob'])) for batch in json.loads(path.read_text())]
    groups = defaultdict(list)
    for batch in batches:
        groups[(batch.get('operation', settings['text_operation']), batch['batch_size'])].append(batch)
    measured = []
    for (operation, size), records in groups.items():
        row = {'operation': operation, 'batch_size': size, 'batches': len(records),
               'service_latency': distribution([record['elapsed_seconds'] for record in records])}
        if operation == settings['finite_operation']:
            row['components'] = {name: distribution([record['structured']['timings'][name] for record in records])
                                 for name in settings['finite_components']}
            row['graph_captures'] = sum(record['graph_captures'] for record in records)
        else:
            row['ordinary_itl'] = distribution([interval for record in records
                for output in record['decode'] for interval in output['inter_token_seconds']])
        measured.append(row)
    return {'scope': settings['scope'], 'tasks': reports, 'batches': measured,
            'dependency_idl_definition': 'Child finite result delivery minus each declared finite predecessor delivery. Includes intervening work; edges sharing nodes are not independent samples.',
            'ready_latency_definition': 'Input-ready timestamp in the execution graph to actual finite future delivery.',
            'queue_latency_definition': 'Native request submission to actual finite future delivery, including queue, input processing, GPU work and delivery.',
            'ordinary_itl_definition': 'Per-row adjacent autoregressive output-token intervals; never divide a batch latency by its size.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, json.loads(args.settings.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
