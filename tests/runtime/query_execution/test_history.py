import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from jev_spawn.runtime.query_execution import QueryExecution


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())


class HistoryProjectionTests(unittest.TestCase):
    def execution(self):
        return QueryExecution('Generic event transport test.', Mock(), 'history-test', SETTINGS)

    def execute(self, execution, identity, value):
        execution.execute_actions([{'id': identity, 'tool': 'record',
            'arguments': {'payload': {'value': value}}}],
            lambda calls: {call['id']: {'observation': 'Recorded.'} for call in calls})

    def test_history_projects_actual_branch_calls_with_native_types(self):
        parent = self.execution()
        self.execute(parent, 'first', [2, False])
        left, right = parent.fork(), parent.fork()
        self.execute(left, 'left', {'literal': '$$unchanged'})
        self.execute(right, 'right', 'unselected sibling')
        reference = {SETTINGS['history_reference']: ['arguments', 'payload', 'value']}
        result = left.materialize(reference)
        self.assertEqual(result, [[2, False], {'literal': '$unchanged'}])
        result[0].append('must not mutate trace')
        self.assertEqual(parent.materialize(reference), [[2, False]])
        self.assertEqual(right.materialize(reference), [[2, False], 'unselected sibling'])
        parent.service.decide.assert_not_called()

    def test_history_composes_with_fields_lists_and_constant_structure(self):
        execution = self.execution()
        execution.values['confirmed'] = True
        self.execute(execution, 'observed', {'cell': [3, 4]})
        template = {'arbitrary_output_key': {SETTINGS['history_reference']: [
                        'arguments', 'payload', 'value', 'cell', 1]},
                    'metadata': ['${confirmed}', None, 'constant']}
        self.assertEqual(execution.materialize(template),
                         {'arbitrary_output_key': [4], 'metadata': [True, None, 'constant']})

    def test_empty_history_contains_no_guessed_actions(self):
        execution = self.execution()
        execution.values['proposed'] = 'not executed'
        self.assertEqual(execution.materialize({SETTINGS['history_reference']: ['arguments']}), [])

    def test_join_serializes_only_executed_calls_on_selected_branch(self):
        parent = self.execution()
        self.execute(parent, 'first', '<pulse>')
        selected, sibling = parent.fork(), parent.fork()
        self.execute(selected, 'second', '<hold>')
        self.execute(sibling, 'alternative', '<release>')
        template = {'commands': {SETTINGS['join_reference']: [' ', {
            SETTINGS['history_reference']: ['arguments', 'payload', 'value']}]}}
        self.assertEqual(selected.materialize(template), {'commands': '<pulse> <hold>'})
        self.assertEqual(parent.materialize(template), {'commands': '<pulse>'})
        parent.service.decide.assert_not_called()

    def test_missing_projection_path_fails_without_substitute(self):
        execution = self.execution()
        self.execute(execution, 'observed', 'value')
        with self.assertRaises(KeyError):
            execution.materialize({SETTINGS['history_reference']: ['arguments', 'absent']})


if __name__ == '__main__':
    unittest.main()
