import argparse
import json

from jev_spawn.infra.configuration import load_resource
from jev_spawn.runtime.reference_transport import unpack_references
from project_paths import ROOT


def read(path):
    return json.loads(path.read_text())


def evidence_records(value):
    if isinstance(value, dict):
        if 'observations' in value:
            yield from value['observations']
        for key, child in value.items():
            if key != 'observations':
                yield from evidence_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from evidence_records(child)


def collect(directory, settings):
    completion = read(directory / 'completion.json')
    protocol = read(directory / 'protocol.json')
    tasks = {record['task_id']: record for record in map(read, sorted(directory.glob(settings['record_glob'])))}
    assert set(tasks) == set(completion['task_ids']) == {task['task_id'] for task in protocol['tasks']}
    transport = load_resource(protocol['method']['settings']['reference_transport']['resource'])
    prefix = protocol['prompts']['encoded_state'].format(value='')
    delivery = {task_id: [] for task_id in tasks}
    for cohort in map(read, sorted(directory.glob(settings['finite_glob']))):
        for request in cohort['requests']:
            field = request['field']
            assert field['state'] in request['rendered']
            assert any(field['state'] in message['content'] for message in request['messages'])
            context, marker, encoded = field['state'].partition(prefix)
            assert marker and field['context'] in context
            semantic = unpack_references(json.loads(encoded), transport)
            for observed in evidence_records(semantic['input']):
                stored = next(item for item in tasks[request['task_id']]['observations']
                              if item['tool'] == observed['tool'] and item['arguments'] == observed['arguments'])
                descriptor = observed['result']
                delivery[request['task_id']].append({
                    'cohort': cohort['cohort'], 'node': field['id'], 'tool': observed['tool'],
                    'arguments': observed['arguments'], 'stored_result': stored['result'],
                    'delivered_result_descriptor': descriptor,
                    'complete_result_delivered': 'value' in descriptor and descriptor['value'] == stored['result']})
    summary = {}
    for task_id, record in tasks.items():
        item = {'status': record['status'], 'answer': record['answer'],
                'elapsed_seconds': record['elapsed_seconds'], 'actual_state_evidence': delivery[task_id]}
        if record['status'] == settings['completed_status']:
            item.update(actions=record['actions'], finite_calls=[
                {'node': call['node'], 'choice': call['result']['choice']}
                for call in record['calls'] if call['kind'] == settings['finite_kind']],
                judgments=len(record['judgments']), expansions=record['expansions'],
                revisions=len(record['revisions']))
        else:
            item['error'] = record['error']
        summary[task_id] = item
    batches = read(directory / settings['batches'])
    finite = [batch for batch in batches if 'operation' in batch]
    text = [batch for batch in batches if 'messages' in batch]
    assert len(finite) + len(text) == len(batches)
    return {'run': str(directory.relative_to(ROOT)), 'completion': completion, 'tasks': summary,
            'whole_session_seconds': read(directory / settings['elapsed'])['seconds'],
            'finite_batches': len(finite), 'finite_decisions': sum(batch['batch_size'] for batch in finite),
            'text_batches': len(text), 'text_requests': sum(batch['batch_size'] for batch in text),
            'text_output_tokens': sum(sum(batch['output_tokens']) for batch in text)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    settings = read(ROOT / arguments.config)
    before = collect(ROOT / settings['before'], settings)
    after = collect(ROOT / settings['after'], settings)
    assert set(before['tasks']) == set(after['tasks'])
    result = {'before': before, 'after': after, 'comparison': {
        task_id: {'answer_changed': before['tasks'][task_id]['answer'] != item['answer'],
                  'before_status': before['tasks'][task_id]['status'], 'after_status': item['status']}
        for task_id, item in after['tasks'].items()}}
    (ROOT / settings['output']).write_text(json.dumps(result, **settings['serialization']) + '\n')


if __name__ == '__main__':
    main()
