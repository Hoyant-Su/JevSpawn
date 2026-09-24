from copy import deepcopy

from jev_spawn.runtime.state import intern


def branch_view(state, description):
    payload = deepcopy(state)
    observations = payload.pop('observations')
    events = payload.pop('execution_events')
    actual = {event['id']: {**{key: value for key, value in event.items() if key != 'observation_id'},
                           'value': observations[event['observation_id']]} for event in events}
    payload['execution_order'] = [event['id'] for event in events]
    payload['current_branch_events'] = [actual[key] for key in payload['current_path']]
    selected = set(payload['current_path'])
    alternatives, values, indexes = [], [], {}
    for identity, event in actual.items():
        if identity not in selected:
            event['observation_id'] = intern(event.pop('value'), values, indexes)
            alternatives.append(event)
    payload['alternative_executions'] = {'execution_events': alternatives, 'observations': values}
    payload['format'] = description
    return payload
