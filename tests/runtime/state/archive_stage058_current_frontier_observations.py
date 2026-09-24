from copy import deepcopy
import json
from unittest import TestCase, main

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.state import event_history, history_frontier as original_history_frontier, pack_frontier


def history_frontier(state):
    # Archived rejected stage058 behavior; production retains feedback IDs only.
    payload, branches = original_history_frontier(state)
    events = {event['id']: deepcopy(event) for event in state['execution_events']}
    for branch in branches.values():
        branch['current_action_observations'] = [events[identity]
            for identity in branch['state']['current_feedback']]
    return payload, branches


class CurrentFrontierObservationsTests(TestCase):
    def test_exact_events_preserve_types_order_ancestry_and_original_payload(self):
        values = ['Exact\r\n雪\n ', False, 0, 0.0, None, {'nested': ['line\n', 3, True]}]
        events = [{'id': f'n{i}', 'source': 'tool', 'parent_feedback': [f'n{i-1}'] if i else [],
                   'argument_bindings': {'action': '${f0}'},
                   'action': {'id': f'n{i}', 'tool': 'execute', 'arguments': {'action': f'move-{i}'}},
                   'value': value} for i, value in enumerate(values)]
        branches = {event['id']: {'state': {
            'current_path': [item['id'] for item in events[:i+1]], 'current_feedback': [event['id']],
            'computed_fields': [{'id': 'f0', 'question': 'Action', 'value': f'move-{i}'}],
            'declaration_feedback': [{'accepted': True}]}, 'declaration_id': 'd0', 'done': False}
            for i, event in enumerate(events)}
        branches['multiple'] = {'state': {'current_path': ['n0', 'n1'], 'current_feedback': ['n1', 'n0'],
                                         'computed_fields': []}, 'declaration_id': None, 'done': False}
        branches['root'] = {'state': {'current_path': [], 'current_feedback': [], 'computed_fields': []},
                            'declaration_id': None, 'done': False}
        original = {'execution_events': events, 'branches': branches, 'declarations': {'d0': {'fields': []}}}
        before = deepcopy(original)
        original_history = event_history(events)
        shared, actual = history_frontier(original)
        self.assertEqual(original, before)
        self.assertEqual(event_history(events), original_history)

        packed_input = deepcopy(original)
        packed_input['execution_events'] = []
        expected = pack_frontier(packed_input)
        expected_branches = expected.pop('branches')
        expected.pop('execution_events')
        expected.pop('observations')
        expected['execution_order'] = [event['id'] for event in events]
        expected['format'] = load_prompt('jevspawn.state')['history_format']
        self.assertEqual(shared, expected)
        for identity, branch in actual.items():
            expected_events = [events[int(key[1:])] for key in branch['state']['current_feedback']]
            self.assertEqual(json.dumps(branch['current_action_observations'], sort_keys=True),
                             json.dumps(expected_events, sort_keys=True))
            self.assertEqual({k: v for k, v in branch.items() if k != 'current_action_observations'},
                             expected_branches[identity])

        actual['n5']['current_action_observations'][0]['value']['nested'][0] = 'changed'
        self.assertEqual(original, before)
        self.assertEqual(shared, expected)


if __name__ == '__main__':
    main()
