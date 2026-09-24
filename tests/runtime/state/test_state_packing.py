from copy import deepcopy
from functools import reduce
from operator import getitem
import unittest

from jev_spawn.runtime.state import frontier_view, pack_frontier, pack_state


def restore_text(events):
    originals = {}
    for event in events:
        event['value'] = deepcopy(event['value'])
        for prefix in event.pop('text_prefixes', []):
            path = ('value', *prefix['path'])
            target = reduce(getitem, path[:-1], event)
            target[path[-1]] = reduce(getitem, prefix['path'], originals[prefix['event']]) + target[path[-1]]
        originals[event['id']] = event['value']


def unpack(payload, frontier):
    restored = deepcopy(payload)
    restored.pop('format')
    observations = restored.pop('observations')
    definitions = restored.pop('field_definitions')
    feedback = restored.pop('declaration_feedback_records')
    for event in restored['execution_events']:
        event['value'] = observations[event.pop('observation_id')]
    restore_text(restored['execution_events'])
    states = [branch['state'] for branch in restored['branches'].values()] if frontier else [restored]
    for state in states:
        if 'declaration_feedback_ids' in state:
            state['declaration_feedback'] = [deepcopy(feedback[index])
                                             for index in state.pop('declaration_feedback_ids')]
        state['computed_fields'] = [{**definitions[index], 'value': value}
                                    for index, value in state.pop('computed_values')]
    return restored


def unpack_view(payload, branches):
    restored = deepcopy(payload)
    definitions = restored.pop('field_definitions')
    feedback = restored.pop('declaration_feedback_records')
    observations = restored.pop('observations')
    order = restored.pop('execution_order')
    restored.pop('format')
    events = {}
    for event in restored.pop('execution_events'):
        event['value'] = observations[event.pop('observation_id')]
        events[event['id']] = event
    restored['branches'] = deepcopy(branches)
    for branch in restored['branches'].values():
        events.update({event['id']: event for event in branch.pop('execution_events')})
        state = branch['state']
        if 'declaration_feedback_ids' in state:
            state['declaration_feedback'] = [deepcopy(feedback[index])
                                             for index in state.pop('declaration_feedback_ids')]
        state['computed_fields'] = [{**definitions[index], 'value': value}
                                    for index, value in state.pop('computed_values')]
    restored['execution_events'] = [events[identity] for identity in order]
    restore_text(restored['execution_events'])
    return restored


class StatePackingTests(unittest.TestCase):
    def setUp(self):
        observation = {'observation': 'An unchanged room.', 'done': False}
        self.events = [
            {'id': 'a', 'parent_feedback': [], 'action': {'arguments': {'action': 0}}, 'value': observation},
            {'id': 'b', 'parent_feedback': ['a'], 'action': {'arguments': {'action': 1}}, 'value': deepcopy(observation)},
            {'id': 'c', 'parent_feedback': ['a'], 'action': {'arguments': {'action': 2}},
             'value': {'observation': 'A different room.', 'done': False}}]
        self.branch = {'current_path': ['a', 'b'], 'current_feedback': ['b'],
            'computed_fields': [{'id': 'f0', 'question': 'Choose the old action.', 'domain': [0, 1], 'value': 1}],
            'field_evidence': {'f0': ['a']}, 'bound_fields': {'f0': 1},
            'declaration_feedback': [{'accepted': False, 'error': 'Actual compiler diagnostic.'}]}

    def test_frontier_roundtrip_preserves_revised_definitions_and_duplicate_observations(self):
        revised = deepcopy(self.branch)
        revised['current_path'] = ['a', 'c']
        revised['computed_fields'][0].update(question='Choose the revised action.', domain=[1, 2], value=2)
        frontier = {'execution_events': deepcopy(self.events), 'declarations': {'new': {'fields': []}},
                    'branches': {'b': {'state': deepcopy(self.branch), 'declaration_id': 'new', 'done': False},
                                 'c': {'state': revised, 'declaration_id': 'new', 'done': False},
                                 'd': {'state': deepcopy(self.branch), 'declaration_id': 'new', 'done': False}}}
        before = deepcopy(frontier)
        packed = pack_frontier(frontier)
        self.assertEqual(frontier, before)
        self.assertEqual(unpack(packed, frontier=True), before)
        self.assertEqual(len(packed['observations']), 3)
        self.assertEqual(len(packed['field_definitions']), 2)
        self.assertEqual(packed['execution_events'][1]['text_prefixes'], [{'path': ['observation'], 'event': 'a'}])
        self.assertNotEqual(packed['branches']['b']['state']['computed_values'][0][0],
                            packed['branches']['c']['state']['computed_values'][0][0])

    def test_individual_state_roundtrip_preserves_all_evidence(self):
        state = {**deepcopy(self.branch), 'execution_events': deepcopy(self.events),
                 'pending_fields': [{'id': 'next', 'values': ['x', 'y']}],
                 'action_template': {'tool': 'execute', 'arguments': {'action': '${next}'}}}
        before = deepcopy(state)
        packed = pack_state(state)
        self.assertEqual(state, before)
        self.assertEqual(unpack(packed, frontier=False), before)
        self.assertEqual(packed['current_path'], before['current_path'])
        self.assertEqual(packed['declaration_feedback_ids'], [0])
        self.assertEqual(packed['declaration_feedback_records'], before['declaration_feedback'])
        self.assertEqual(packed['field_evidence'], before['field_evidence'])
        self.assertEqual(packed['bound_fields'], before['bound_fields'])

    def test_empty_state_roundtrip(self):
        state = {'execution_events': [], 'current_path': [], 'computed_fields': [], 'declaration_feedback': []}
        self.assertEqual(unpack(pack_state(state), frontier=False), state)

    def test_frontier_choices_preserve_histories_order_and_every_event_once(self):
        revised = deepcopy(self.branch)
        revised.update(current_path=['a', 'c'], current_feedback=['c'])
        revised['computed_fields'][0].update(question='Revised action.', domain=[1, 2], value=2)
        state = {'execution_events': deepcopy(self.events), 'declarations': {'new': {'fields': []}},
                 'branches': {'b': {'state': deepcopy(self.branch), 'declaration_id': 'new', 'done': False},
                              'c': {'state': revised, 'declaration_id': 'new', 'done': False}}}
        before = deepcopy(state)
        shared, choices = frontier_view(state)
        self.assertEqual(state, before)
        self.assertEqual(unpack_view(shared, choices), before)
        self.assertEqual(list(choices), list(state['branches']))
        self.assertNotIn('branches', shared)
        self.assertEqual([event['id'] for event in shared['execution_events']], ['a'])
        serialized = shared['execution_events'] + [event for choice in choices.values()
                                                    for event in choice['execution_events']]
        self.assertCountEqual([event['id'] for event in serialized], ['a', 'b', 'c'])
        restored_events = {event['id']: event for event in unpack_view(shared, choices)['execution_events']}
        for identity in choices:
            self.assertEqual(restored_events[identity],
                             next(event for event in self.events if event['id'] == identity))

    def test_nested_cumulative_observations_roundtrip_across_branches(self):
        events = [
            {'id': 'root', 'parent_feedback': [], 'value': {'nested': ['Initial state.'], 'flag': False}},
            {'id': 'left', 'parent_feedback': ['root'], 'value': {'nested': ['Initial state. Left result.'], 'flag': True}},
            {'id': 'right', 'parent_feedback': ['root'], 'value': {'nested': ['Initial state. Right result.'], 'flag': False}},
            {'id': 'next', 'parent_feedback': ['left'], 'value': {'nested': ['Initial state. Left result. Final result.'], 'flag': True}}]
        state = {'execution_events': events, 'computed_fields': []}
        packed = pack_state(state)
        self.assertEqual(unpack(packed, frontier=False), state)
        suffixes = [packed['observations'][event['observation_id']]['nested'][0]
                    for event in packed['execution_events']]
        self.assertEqual(suffixes, ['Initial state.', ' Left result.', ' Right result.', ' Final result.'])

    def test_scalar_observations_roundtrip(self):
        state = {'execution_events': [
            {'id': 'a', 'parent_feedback': [], 'value': 'Previous'},
            {'id': 'b', 'parent_feedback': ['a'], 'value': 'Previous plus new'}], 'computed_fields': []}
        self.assertEqual(unpack(pack_state(state), frontier=False), state)

    def test_empty_frontier_choice_requires_no_invented_observation(self):
        branch = deepcopy(self.branch)
        branch.update(current_path=[], current_feedback=[], computed_fields=[])
        state = {'execution_events': [], 'declarations': {},
                 'branches': {'root': {'state': branch, 'declaration_id': None, 'done': False}}}
        shared, choices = frontier_view(state)
        self.assertEqual(choices['root']['execution_events'], [])
        self.assertEqual(shared['observations'], [])
        self.assertEqual(unpack_view(shared, choices), state)


if __name__ == '__main__':
    unittest.main()
