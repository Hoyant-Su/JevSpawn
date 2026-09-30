from copy import deepcopy
from itertools import takewhile
import json

from jev_spawn.infra.prompts import load_prompt


def string_leaves(value, path):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from string_leaves(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from string_leaves(child, (*path, index))


def compact_observations(events):
    originals = {event['id']: dict(string_leaves(event['value'], ())) for event in events}
    for event in events:
        prefixes = []
        for path, text in originals[event['id']].items():
            matches = [(len(prefix), identity) for identity in event['parent_feedback']
                       for location, prefix in originals[identity].items()
                       if location == path and prefix and text.startswith(prefix)]
            if matches:
                length, identity = max(matches)
                target = event
                destination = ('value', *path)
                for key in destination[:-1]:
                    target = target[key]
                target[destination[-1]] = text[length:]
                prefixes.append({'path': list(path), 'event': identity})
        if prefixes:
            event['text_prefixes'] = prefixes


def intern(value, table, indexes):
    key = json.dumps(value, sort_keys=True, separators=(',', ':'))
    if key not in indexes:
        indexes[key] = len(table)
        table.append(value)
    return indexes[key]


def pack(payload, branch_states):
    observations, observation_ids, definitions, definition_ids = [], {}, [], {}
    feedback_records, feedback_ids = [], {}
    compact_observations(payload['execution_events'])
    for event in payload['execution_events']:
        event['observation_id'] = intern(event.pop('value'), observations, observation_ids)
    for state in branch_states:
        if 'declaration_feedback' in state:
            state['declaration_feedback_ids'] = [intern(record, feedback_records, feedback_ids)
                                                for record in state.pop('declaration_feedback')]
        values = []
        for field in state.pop('computed_fields'):
            value = field.pop('value')
            values.append([intern(field, definitions, definition_ids), value])
        state['computed_values'] = values
    payload['observations'] = observations
    payload['field_definitions'] = definitions
    payload['declaration_feedback_records'] = feedback_records
    payload['format'] = load_prompt('jevspawn.state')['format']
    return payload


def pack_state(state):
    payload = deepcopy(state)
    return pack(payload, [payload])


def pack_frontier(state):
    payload = deepcopy(state)
    return pack(payload, [branch['state'] for branch in payload['branches'].values()])


def frontier_view(state):
    payload = pack_frontier(state)
    branches = payload.pop('branches')
    observations = payload.pop('observations')
    events = payload.pop('execution_events')
    actual = {event['id']: {**{key: value for key, value in event.items() if key != 'observation_id'},
                           'value': observations[event['observation_id']]} for event in events}
    current = [identity for branch in branches.values() for identity in branch['state']['current_feedback']]
    assert len(current) == len(set(current)), 'Frontier branches must own distinct current events.'
    current = set(current)
    payload['execution_order'] = [event['id'] for event in events]
    payload['execution_events'], payload['observations'], indexes = [], [], {}
    for event in actual.values():
        if event['id'] not in current:
            event['observation_id'] = intern(event.pop('value'), payload['observations'], indexes)
            payload['execution_events'].append(event)
    for branch in branches.values():
        branch['execution_events'] = [actual[identity] for identity in branch['state']['current_feedback']]
    payload['format'] = load_prompt('jevspawn.state')['frontier_format']
    return payload, branches


def extend_event_history(events, preceding):
    protocol = load_prompt('jevspawn.state')
    serialization = protocol['history_serialization']
    sources = [(event['id'], dict(string_leaves(event['value'], ()))) for event in preceding]
    records = []
    for original in events:
        event = deepcopy(original)
        references = []
        for path, text in string_leaves(original['value'], ()):
            lines = text.splitlines(keepends=True)
            best = None
            count = 0
            for identity, leaves in sources:
                if path not in leaves:
                    continue
                shared = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1],
                    zip(lines, leaves[path].splitlines(keepends=True))))
                if shared > count:
                    best, count = identity, shared
            if not count:
                continue
            reference = {'path': list(path), 'event': best, 'lines': count}
            candidate = deepcopy(event)
            target = candidate
            destination = ('value', *path)
            for key in destination[:-1]:
                target = target[key]
            target[destination[-1]] = ''.join(lines[count:])
            candidate['text_prefixes'] = [*references, reference]
            if len(json.dumps(candidate, **serialization)) < len(json.dumps(event, **serialization)):
                event = candidate
                references.append(reference)
        records.append(protocol['history_record'].format(event=json.dumps(event, **serialization)))
        sources.append((original['id'], dict(string_leaves(original['value'], ()))))
    return ''.join(records)


def history_state(state):
    payload = deepcopy({key: value for key, value in state.items() if key != 'execution_events'})
    payload['execution_events'] = []
    payload = pack(payload, [payload])
    payload.pop('execution_events')
    payload.pop('observations')
    payload['execution_order'] = [event['id'] for event in state['execution_events']]
    payload['format'] = load_prompt('jevspawn.state')['history_format']
    return payload


def history_frontier(state):
    payload = deepcopy({key: value for key, value in state.items() if key != 'execution_events'})
    payload['execution_events'] = []
    branches = payload['branches']
    payload = pack(payload, [branch['state'] for branch in branches.values()])
    payload.pop('branches')
    payload.pop('execution_events')
    payload.pop('observations')
    payload['execution_order'] = [event['id'] for event in state['execution_events']]
    payload['format'] = load_prompt('jevspawn.state')['history_format']
    return payload, branches
