from copy import deepcopy
import json
import unittest

from jev_spawn.runtime.state import history_frontier, history_state, pack_frontier


def restored_feedback(shared, states):
    restored = deepcopy(states)
    for state in restored:
        if 'declaration_feedback_ids' in state:
            state['declaration_feedback'] = [deepcopy(shared['declaration_feedback_records'][index])
                for index in state.pop('declaration_feedback_ids')]
    return restored


class FeedbackInterningTests(unittest.TestCase):
    def test_frontier_preserves_order_duplicates_types_and_field_presence(self):
        record = {'accepted': False, 'error': 'Exact\r\n雪 diagnostic.',
                  'source_signatures': ['{x:enum("a", "b")}\n'],
                  'proposed_declaration': {'typed': [False, 0, 0.0, None]}}
        alternate = {**record, 'accepted': True}
        states = [{'computed_fields': [], 'declaration_feedback': [record, alternate, record]},
                  {'computed_fields': [], 'declaration_feedback': [deepcopy(record)]},
                  {'computed_fields': [], 'declaration_feedback': []}, {'computed_fields': []}]
        original = {'execution_events': [], 'declarations': {},
                    'branches': {str(i): {'state': state} for i, state in enumerate(states)}}
        before = deepcopy(original)
        packed = pack_frontier(original)
        self.assertEqual(original, before)
        self.assertEqual(packed['declaration_feedback_records'], [record, alternate])
        branches = [b['state'] for b in packed['branches'].values()]
        self.assertEqual([b.get('declaration_feedback_ids') for b in branches], [[0, 1, 0], [0], [], None])
        restored = restored_feedback(packed, branches)
        for state in restored:
            state['computed_fields'] = state.pop('computed_values')
        self.assertEqual(json.dumps(restored, sort_keys=True), json.dumps(states, sort_keys=True))

    def test_equal_records_ignore_dictionary_order_without_collapsing_scalar_types(self):
        records = [{'value': False, 'accepted': True}, {'accepted': True, 'value': False},
                   {'value': 0, 'accepted': True}, {'value': 0.0, 'accepted': True}]
        packed = history_state({'execution_events': [], 'computed_fields': [],
                                'declaration_feedback': records})
        self.assertEqual(packed['declaration_feedback_ids'], [0, 0, 1, 2])
        self.assertEqual(len(packed['declaration_feedback_records']), 3)
        restored = restored_feedback(packed, [packed])[0]
        self.assertEqual(json.dumps(restored['declaration_feedback'], sort_keys=True),
                         json.dumps(records, sort_keys=True))

    def test_history_frontier_shares_records_but_keeps_branch_associations(self):
        rejected = {'accepted': False, 'error': 'Invalid slot.', 'source_signatures': ['{slot}']}
        accepted = {'accepted': True, 'source_signatures': ['move']}
        original = {'execution_events': [], 'declarations': {}, 'branches': {
            'left': {'state': {'computed_fields': [], 'declaration_feedback': [rejected, accepted]}},
            'right': {'state': {'computed_fields': [], 'declaration_feedback': [accepted, rejected]}}}}
        before = deepcopy(original)
        shared, branches = history_frontier(original)
        self.assertEqual(original, before)
        self.assertEqual(branches['left']['state']['declaration_feedback_ids'], [0, 1])
        self.assertEqual(branches['right']['state']['declaration_feedback_ids'], [1, 0])
        restored = restored_feedback(shared, [b['state'] for b in branches.values()])
        self.assertEqual([s['declaration_feedback'] for s in restored], [[rejected, accepted], [accepted, rejected]])


if __name__ == '__main__':
    unittest.main()
