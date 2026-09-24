import argparse
import json
from pathlib import Path

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.query_execution import QueryExecution
from tests.methods.observed_fields.candidate import prepare


def replay(settings):
    method = json.loads(Path(settings['method']).read_text())
    execution_settings = method['settings']['rollout']['execution']
    records = []
    for path in settings['tasks']:
        task = json.loads(Path(path).read_text())
        rounds = task['trace']['rounds']
        events = {event['id']: event for turn in rounds
                  for child in turn.get('children', {}).values()
                  for computation in child
                  for event in computation.get('observations', [])}
        for turn in rounds:
            for computations in turn.get('parent_computations', {}).values():
                for computation in computations:
                    for request in computation.get('requests', []):
                        state = json.loads(request['state'])
                        execution = QueryExecution(request['context'], None, task['task_id'], execution_settings)
                        execution.prompts = load_prompt(settings['prompt'])
                        execution.history = request['history']
                        execution.history_events = tuple(events[event['id']]
                            for event in map(json.loads, request['history'].splitlines()))
                        execution.tool_observations = [events[identity] for identity in state['current_path']]
                        execution.latest_feedback = [events[identity] for identity in state['current_feedback']]
                        execution.bound_fields = state['bound_fields']
                        execution.active_declaration = state['active_declaration']
                        execution.declaration_feedback = [state['declaration_feedback_records'][index]
                            for index in state['declaration_feedback_ids']]
                        fields = state['pending_fields']
                        _, requests, evidence = prepare(execution, fields)
                        new = next(item for item in requests if item['id'] == request['id'])
                        rendered = json.loads(new['state'])
                        assert new['context'] == request['context'] and new['history'] == request['history']
                        assert new['options'] == request['options']
                        assert rendered['latest_action_observations'] == execution.latest_feedback
                        assert rendered['bound_fields'] == state['bound_fields']
                        assert rendered['active_declaration'] == state['active_declaration']
                        assert evidence == [event['id'] for event in execution.history_events]
                        records.append({'task': task['task_id'], 'turn': turn['turn'], 'field': request['id'],
                            'old_state_characters': len(request['state']), 'new_state_characters': len(new['state']),
                            'actual_observation_count': len(execution.latest_feedback)})
    result = {'scope': 'Saved-request input preservation only; no model accuracy or speed claim.',
              'sources': settings['tasks'], 'requests_checked': len(records), 'requests': records}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': settings['output'], 'requests_checked': len(records)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    replay(json.loads(parser.parse_args().config.read_text()))
