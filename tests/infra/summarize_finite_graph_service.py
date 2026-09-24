import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    return {'count': len(values), 'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'maximum': max(values) if values else None, 'sum': sum(values)}


def summarize(specification):
    run = Path(specification['run'])
    protocol, completion = read(run / 'protocol.json'), read(run / 'completion.json')
    sessions = [read(path) for path in sorted(run.glob('session-*/completion.json'))]
    evaluation = read(specification['evaluation'])
    tasks = [read(run / f'task-{index:05d}.json') for index in range(completion['tasks'])]
    assert completion['task_ids'] == [task['task_id'] for task in tasks]
    assert len(tasks) == evaluation['tasks'] == len(protocol['tasks'])
    batches = [batch for path in sorted(run.glob('session-*/batches.json')) for batch in read(path)]
    finite = [batch for batch in batches if batch.get('operation') == 'finite']
    text = [batch for batch in batches if batch.get('operation') != 'finite']
    phases, by_size, by_task = defaultdict(list), defaultdict(list), Counter()
    for batch in finite:
        for phase, seconds in batch['structured']['timings'].items():
            phases[phase].append(seconds)
        by_size[batch['batch_size']].append(batch['elapsed_seconds'])
        by_task.update(batch['task_ids'])
    clean_text = [batch for batch in text if not any(
        candidate['started_monotonic'] >= batch['started_monotonic']
        and candidate['finished_monotonic'] <= batch['finished_monotonic'] for candidate in finite)]
    itl = [value for batch in text for row in batch['decode'] for value in row['inter_token_seconds']]
    clean_itl = [value for batch in clean_text for row in batch['decode'] for value in row['inter_token_seconds']]
    task_rows = []
    for task, score in zip(tasks, evaluation['scores'], strict=True):
        assert task['task_id'] == score['task_id']
        calls = task.get('calls')
        task_rows.append({'task_id': task['task_id'], 'dataset': score['dataset'], 'status': task['status'],
                          'elapsed_seconds': task['elapsed_seconds'], 'executed_finite_decisions': by_task[task['task_id']],
                          'recorded_call_kinds': dict(Counter(call['kind'] for call in calls)) if calls is not None else None,
                          'judgments': len(task['judgments']) if 'judgments' in task else None,
                          'expansions': len(task['expansions']) if 'expansions' in task else None,
                          'proposals': len(task['proposals']) if 'proposals' in task else None,
                          'revisions': len(task['revisions']) if 'revisions' in task else None,
                          'failure': task.get('error'), 'quality': score})
    result = {'run': str(run), 'evaluation': specification['evaluation'], 'completion': completion, 'sessions': sessions,
              'session_lifecycle_seconds': sum(session['elapsed_seconds'] for session in sessions),
              'status_counts': evaluation['status_counts'], 'datasets': evaluation['datasets'], 'tasks': task_rows,
              'finite': {'batches': len(finite), 'decisions': sum(batch['batch_size'] for batch in finite),
                         'batch_size_counts': dict(Counter(batch['batch_size'] for batch in finite)),
                         'root_count_counts': dict(Counter(len(set(batch['task_ids'])) for batch in finite)),
                         'service_seconds': sum(batch['elapsed_seconds'] for batch in finite),
                         'per_call': [{'task_ids': batch['task_ids'], 'node_ids': batch['node_ids'],
                                       'batch_size': batch['batch_size'], 'elapsed_seconds': batch['elapsed_seconds'],
                                       'group_sizes': batch['structured']['group_sizes'],
                                       'suffix_tokens': batch['structured']['suffix_tokens'],
                                       'persistent_prefix_hits': batch['structured']['persistent_prefix_hits'],
                                       'timings': batch['structured']['timings']} for batch in finite],
                         'phase_seconds': {name: distribution(values) for name, values in phases.items()},
                         'service_seconds_by_batch_size': {size: distribution(values) for size, values in by_size.items()},
                         'queue_seconds': distribution([value for batch in finite for value in batch['queue_seconds']]),
                         'forward_shape_counts': dict(Counter(str(shape) for batch in finite for shape in batch['forward_input_shapes']))},
              'text': {'batches': len(text), 'output_tokens': sum(sum(batch['output_tokens']) for batch in text),
                       'observed_inter_token_seconds': distribution(itl),
                       'cohorts_without_nested_finite': len(clean_text),
                       'inter_token_seconds_without_nested_finite': distribution(clean_itl),
                       'queue_seconds': distribution([value for batch in text for value in batch['queue_seconds']]),
                       'capture_seconds': sum(batch['graph_capture_seconds'] for batch in text)}}
    return result, protocol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = read(args.config)
    reports, protocols = {}, {}
    for name, specification in settings['runs'].items():
        reports[name], protocols[name] = summarize(specification)
    first, *others = protocols.values()
    result = {'runs': reports, 'same_tasks': all(protocol['tasks'] == first['tasks'] for protocol in others),
              'same_shared_config': all(protocol['shared_config_text'] == first['shared_config_text'] for protocol in others),
              'same_prompts': all(protocol['prompts'] == first['prompts'] for protocol in others),
              'interpretation': ['End-to-end task outcomes may follow different trajectories; timing differences are not same-workload kernel speedups.',
                                 'tiles_seconds contains suffix-prefill, load, capture and replay subphases; do not add nested phase totals.',
                                 'Observed text inter-token intervals can include nested finite work. No-interleaving cohorts are reported separately; these still include stopping/dispatch overhead.',
                                 'Executed finite decisions include controller and value-selection calls; they are not all independent spawned agent lifecycles.',
                                 'Null task trace counts indicate unavailable traces on failed tasks, not zero work.']}
    assert result['same_tasks'] and result['same_shared_config']
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {'status_counts': report['status_counts'],
                           'elapsed_seconds': report['session_lifecycle_seconds'],
                           'finite_batches': report['finite']['batches'], 'finite_decisions': report['finite']['decisions']}
                      for name, report in reports.items()}))


if __name__ == '__main__':
    main()
