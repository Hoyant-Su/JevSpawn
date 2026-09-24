import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def decision_intervals(value):
    if isinstance(value, dict):
        if ({'ready_monotonic', 'submitted_monotonic', 'delivered_monotonic', 'input_tokens'} <= value.keys()
                and value['input_tokens']):
            yield (value['ready_monotonic'], value['submitted_monotonic'], value['delivered_monotonic'])
        for child in value.values():
            yield from decision_intervals(child)
    elif isinstance(value, list):
        for child in value:
            yield from decision_intervals(child)


def summarize(settings):
    lane = json.loads(Path(settings['lane']).read_text())
    arms, protocols, controls = {}, {}, {}
    for name, entry in zip(settings['arms'], lane['entries'], strict=True):
        run = Path(entry['run'])
        evaluation, = json.loads(Path(entry['evaluation']).read_text())['runs']
        protocols[name] = json.loads((run / 'protocol.json').read_text())
        tasks = [json.loads(path.read_text()) for path in sorted(run.glob('task-*.json'))]
        assert len(tasks) == evaluation['summary']['declared_tasks']
        batches = [row for file in sorted(run.glob('session-*/batches.json'))
                   for row in json.loads(file.read_text())]
        finite = [row for row in batches if row.get('operation') == settings['finite_operation']]
        text = [row for row in batches if 'operation' not in row]
        warm = [row for row in finite if all(row['structured']['persistent_prefix_hits'])
                and not row['graph_captures']]
        intervals = [interval for task in tasks for interval in set(decision_intervals(task['trace']))]
        idl = [delivered - ready for ready, submitted, delivered in intervals]
        arms[name] = {'quality': evaluation['summary'],
            'turns': sum(len(task['trace']['rounds']) for task in tasks),
            'spawned_children': sum(len(turn['children']) for task in tasks for turn in task['trace']['rounds']
                                    if 'children' in turn),
            'finite_calls': len(finite), 'finite_seconds': sum(row['elapsed_seconds'] for row in finite),
            'warm_finite_calls': len(warm), 'warm_finite_seconds': sum(row['elapsed_seconds'] for row in warm),
            'warm_finite_median_seconds': median(row['elapsed_seconds'] for row in warm),
            'text_seconds': sum(row['elapsed_seconds'] for row in text),
            'text_output_tokens': sum(sum(row['output_tokens']) for row in text),
            'finite_batch_histogram': dict(Counter(row['batch_size'] for row in finite)),
            'computed_input_tokens': sum(row['structured']['computed_input_tokens'] for row in finite),
            'idl_ready_to_delivery_mean_seconds': mean(idl),
            'idl_ready_to_delivery_median_seconds': median(idl)}
        controls[name] = {task['task_id']: {'answer': task['answer'],
            'choices': [(turn['selected'], turn['selected_operation']) for turn in task['trace']['rounds']]}
                         for task in tasks}
    first, second = settings['arms']
    protocol_equal = {key: protocols[first][key] == protocols[second][key]
                      for key in settings['shared_protocol_fields']}
    assert all(protocol_equal.values())
    result = {'arms': arms, 'shared_protocol_fields_equal': protocol_equal,
        'per_task_equal': {identity: controls[first][identity] == controls[second][identity]
                           for identity in controls[first]}, 'scope': settings['scope']}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    summarize(json.loads(parser.parse_args().config.read_text()))
