import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def describe(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'maximum': max(values) if values else None, 'sum': sum(values)}


def summarize(spec):
    run = Path(spec['run'])
    protocol = read(run / 'protocol.json')
    completion = read(run / 'completion.json')
    evaluation = read(spec['evaluation'])
    sessions = [read(path) for path in sorted(run.glob('session-*/completion.json'))]
    batches = [batch for path in sorted(run.glob('session-*/batches.json')) for batch in read(path)]
    tasks = [read(path) for path in sorted(run.glob('task-*.json'))]
    assert completion['task_ids'] == [task['task_id'] for task in tasks]
    assert len(tasks) == evaluation['tasks'] == len(protocol['tasks'])
    finite = [batch for batch in batches if batch.get('operation') == 'finite']
    text = [batch for batch in batches if batch.get('operation') != 'finite']
    clean = [batch for batch in text if not any(
        call['started_monotonic'] >= batch['started_monotonic'] and
        call['finished_monotonic'] <= batch['finished_monotonic'] for call in finite)]
    itl = [interval for batch in text for row in batch['decode'] for interval in row['inter_token_seconds']]
    clean_itl = [interval for batch in clean for row in batch['decode'] for interval in row['inter_token_seconds']]
    result = {'run': str(run), 'evaluation': spec['evaluation'],
              'status_counts': evaluation['status_counts'], 'dataset_quality': evaluation['datasets'],
              'session_seconds': sum(session['elapsed_seconds'] for session in sessions),
              'tasks': [{'task_id': task['task_id'], 'status': task['status'],
                         'elapsed_seconds': task['elapsed_seconds'], 'error': task.get('error')}
                        for task in tasks],
              'finite_batches': len(finite), 'finite_decisions': sum(batch['batch_size'] for batch in finite),
              'text_batches': len(text), 'text_batch_sizes': dict(Counter(batch['batch_size'] for batch in text)),
              'text_output_tokens': sum(sum(batch['output_tokens']) for batch in text),
              'text_service_seconds': sum(batch['elapsed_seconds'] for batch in text),
              'text_inter_token_seconds': describe(itl), 'text_inter_token_seconds_without_nested_finite': describe(clean_itl),
              'text_queue_seconds': describe([delay for batch in text for delay in batch['queue_seconds']]),
              'text_graph_capture_seconds': sum(batch['graph_capture_seconds'] for batch in text),
              'peak_allocated_bytes': max(batch['peak_allocated_bytes'] for batch in batches),
              'peak_reserved_bytes': max(batch['peak_reserved_bytes'] for batch in batches)}
    return result, protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    config = read(parser.parse_args().config)
    reports, protocols = {}, {}
    for name, spec in config['runs'].items():
        reports[name], protocols[name] = summarize(spec)
    first, *others = protocols.values()
    checks = {key: all(protocol[key] == first[key] for protocol in others)
              for key in ('tasks', 'shared_config_text', 'tools')}
    assert all(checks.values())
    result = {'runs': reports, 'exact_protocol_equality': checks,
              'scope': 'Original 12 noncoding qualification tasks. Fastest reference is scoped only to measured HiAgent and LatentMAS. No all-baseline or full-dataset claim.',
              'timing': 'Complete task lifecycles including tools, queue waits and captures; model loading excluded. Failed tasks remain in denominators and can terminate early. Distinct trajectories are not matched kernel speedups.',
              'itl': 'Observed per-row text token intervals include dispatch and stopping; nested finite work is separately excluded where possible. Latent-role computation is included in session and service time, not text ITL.',
              'quality': 'Official metrics remain separate by dataset; no pooled score across heterogeneous metrics.'}
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {'seconds': row['session_seconds'], 'status': row['status_counts'],
                            'text_itl': row['text_inter_token_seconds']}
                      for name, row in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
