import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def decision_events(tasks):
    events = set()
    pending = [task['trace'] for task in tasks]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if {'ready_monotonic', 'submitted_monotonic', 'delivered_monotonic', 'input_tokens'} <= value.keys() and value['input_tokens']:
                events.add((value['ready_monotonic'], value['delivered_monotonic']))
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return [delivered - ready for ready, delivered in events]


def arm_record(entry):
    run = Path(entry['run'])
    evaluation, = json.loads(Path(entry['evaluation']).read_text())['runs']
    tasks = [json.loads(path.read_text()) for path in sorted(run.glob('task-*.json'))]
    batches = [batch for path in sorted(run.glob('session-*/batches.json'))
               for batch in json.loads(path.read_text())]
    finite = [batch for batch in batches if batch.get('operation') == 'finite']
    text = [batch for batch in batches if 'operation' not in batch]
    rounds = [record for task in tasks for record in task['trace']['rounds']]
    latency = decision_events(tasks)
    record = {'quality': evaluation['summary'], 'turns': len(rounds),
        'spawned_children': sum(len(record['children']) for record in rounds if 'children' in record),
        'finite_calls': len(finite), 'finite_seconds': sum(batch['elapsed_seconds'] for batch in finite),
        'finite_median_seconds': median(batch['elapsed_seconds'] for batch in finite),
        'finite_batch_histogram': dict(Counter(batch['batch_size'] for batch in finite)),
        'finite_engines': dict(Counter(batch['decode_engine'] for batch in finite)),
        'computed_input_tokens': sum(batch['structured']['computed_input_tokens'] for batch in finite),
        'text_seconds': sum(batch['elapsed_seconds'] for batch in text),
        'text_output_tokens': sum(sum(batch['output_tokens']) for batch in text),
        'graph_capture_seconds': sum(batch['graph_capture_seconds'] for batch in finite),
        'graph_replays': sum(batch['graph_replays'] for batch in finite),
        'recorded_model_decision_events': len(latency),
        'idl_ready_to_delivery_mean_seconds': mean(latency),
        'idl_ready_to_delivery_median_seconds': median(latency)}
    history = [batch['structured'] for batch in finite
               if 'history_prefix_hits' in batch['structured']]
    if history:
        record['history_cache'] = {
            'measured_finite_calls': len(history),
            'request_groups': sum(len(item['history_prefix_hits']) for item in history),
            'hit_groups': sum(sum(item['history_prefix_hits']) for item in history),
            'reused_history_tokens': sum(item['reused_state_tokens'] for item in history),
            'reused_root_tokens': sum(item['reused_root_tokens'] for item in history)}
    return record, json.loads((run / 'protocol.json').read_text()), {
        score['task_id']: score for score in evaluation['scores']}


def compare(settings):
    arms, protocols, scores = {}, {}, {}
    for name, entry in settings['arms'].items():
        arms[name], protocols[name], scores[name] = arm_record(entry)
    reference, candidate = settings['reference'], settings['candidate']
    fields_equal = {key: protocols[reference][key] == protocols[candidate][key]
                    for key in protocols[reference]}
    assert all(fields_equal.values())
    assert scores[reference].keys() == scores[candidate].keys()
    paired = [{'task_id': identity, 'reference_correct': item['correct'],
        'candidate_correct': scores[candidate][identity]['correct'],
        'answer_equal': item['answer'] == scores[candidate][identity]['answer'],
        'reference_seconds': item['elapsed_seconds'],
        'candidate_seconds': scores[candidate][identity]['elapsed_seconds']}
        for identity, item in scores[reference].items()]
    report = {'scope': settings['scope'], 'arms': arms,
        'shared_protocol_fields_equal': fields_equal, 'paired_tasks': paired,
        'mean_task_speedup': arms[reference]['quality']['mean_sample_seconds'] /
                             arms[candidate]['quality']['mean_sample_seconds'],
        'idl_definition': settings['idl_definition']}
    Path(settings['output']).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': settings['output'], 'speedup': report['mean_task_speedup'],
                      'quality': {name: record['quality'] for name, record in arms.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    compare(json.loads(parser.parse_args().config.read_text()))
