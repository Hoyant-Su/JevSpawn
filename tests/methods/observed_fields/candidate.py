from copy import deepcopy
import json

from jev_spawn.runtime.query_execution import QueryExecution


def observed_state(execution, fields):
    return {
        'current_path': [event['id'] for event in execution.tool_observations],
        'latest_action_observations': execution.latest_feedback,
        'bound_fields': execution.bound_fields,
        'pending_fields': fields,
        'active_declaration': execution.active_declaration,
        'declaration_feedback': execution.declaration_feedback,
    }


def prepare(self, fields):
    domains = {field['id']: {
        self.settings['candidate_id'].format(index=index): deepcopy(value)
        for index, value in enumerate(field['values'])} for field in fields}
    evidence = [event['id'] for event in self.history_events]
    state = json.dumps(observed_state(self, fields), **self.settings['serialization'])
    requests = [{'id': field['id'], 'context': self.query, 'history': self.history,
        'state': state, 'question': self.prompts['select_field'].format(identity=field['id']),
        'options': [{'id': identity, 'description': json.dumps(value, **self.settings['serialization'])}
                    for identity, value in domains[field['id']].items()]} for field in fields]
    return domains, requests, evidence


def install():
    QueryExecution.prepare = prepare
