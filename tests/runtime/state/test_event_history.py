from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.runtime.state import event_history, extend_event_history, history_frontier, history_state


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())


def restore(history):
    records = {}
    for line in history.splitlines():
        event = json.loads(line)
        for reference in event.pop('text_prefixes', []):
            original = records[reference['event']]['value']
            for key in reference['path']:
                original = original[key]
            target = event
            path = ['value', *reference['path']]
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = ''.join(original.splitlines(keepends=True)[:reference['lines']]) + target[path[-1]]
        records[event['id']] = event
    return list(records.values())


class EventHistoryTests(unittest.TestCase):
    def setUp(self):
        self.execution = QueryExecution('Original task.', Mock(), 'task', SETTINGS)
        self.field = {'id': 'choice', 'question': 'Which record?', 'values': ['left', 'right']}

    def execute(self, execution, identity, value):
        execution.execute_actions([{'id': identity, 'tool': 'record',
            'arguments': {'value': value}}],
            lambda calls: {call['id']: {'observed': call['arguments']['value']} for call in calls})

    def test_shared_rules_and_cumulative_text_are_lossless_and_append_stable(self):
        rules = 'One shared instruction with unicode 雪 and exact spacing.\r\n' * 8
        self.execute(self.execution, 'first', {'text': rules + 'board one\n', 'scalar': [None, False, 7]})
        prefix = self.execution.history
        left, right = self.execution.fork(), self.execution.fork()
        self.execute(left, 'left', {'text': rules + 'board one\nAction A\nboard two\n', 'scalar': [None, False, 7]})
        self.execute(right, 'right', {'text': rules + 'board three\n', 'scalar': [None, False, 7]})
        events = [*self.execution.tool_observations, *left.latest_feedback, *right.latest_feedback]
        history = event_history(events)
        self.assertTrue(history.startswith(prefix))
        self.assertEqual(restore(history), events)
        self.assertEqual(history, event_history(events[:2]) + extend_event_history(events[2:], events[:2]))
        self.assertEqual(left.history, event_history(left.tool_observations))
        self.assertLess(len(history), len(''.join(json.dumps(event) for event in events)))
        encoded = [json.loads(line) for line in history.splitlines()]
        self.assertIn('text_prefixes', encoded[-1])
        self.assertEqual(encoded[-1]['parent_feedback'], ['first'])
        self.assertEqual(events[-1]['value']['observed']['text'], rules + 'board three\n')

    def test_original_records_roundtrip_and_append_without_rewriting(self):
        self.execute(self.execution, 'first', {'unicode': '雪', 'lines': 'a\nb', 'typed': [None, False, 7]})
        prefix = self.execution.history
        first = deepcopy(self.execution.tool_observations)
        self.execute(self.execution, 'second', 'a\nb\nc')
        self.assertTrue(self.execution.history.startswith(prefix))
        self.assertEqual([json.loads(line) for line in self.execution.history.splitlines()],
                         self.execution.tool_observations)
        self.assertEqual(self.execution.tool_observations[:1], first)
        self.assertEqual(self.execution.history, event_history(self.execution.tool_observations))

    def test_global_observations_do_not_change_branch_ancestry_or_projection(self):
        self.execute(self.execution, 'first', 'common')
        left, right = self.execution.fork(), self.execution.fork()
        self.execute(left, 'left', 'selected')
        self.execute(right, 'right', 'alternative')
        events = [*self.execution.tool_observations, left.latest_feedback[0], right.latest_feedback[0]]
        left.history = right.history = event_history(events)
        left.history_events = right.history_events = tuple(events)
        left.active_declaration = {'fields': [self.field], 'action': {'tool': 'record'}}
        _, requests, evidence = left.prepare([self.field])
        request, = requests
        state = json.loads(request['state'])
        self.assertEqual(request['context'], 'Original task.')
        self.assertEqual(request['history'], right.history)
        self.assertEqual(evidence, ['first', 'left', 'right'])
        self.assertEqual(state['current_path'], ['first', 'left'])
        self.assertEqual(state['current_feedback'], ['left'])
        self.assertEqual(state['active_declaration'], left.active_declaration)
        self.assertNotIn('execution_events', state)
        self.assertNotIn('observations', state)
        self.assertEqual(left.materialize({SETTINGS['history_reference']: ['arguments', 'value']}),
                         ['common', 'selected'])
        child = left.fork()
        self.assertIs(child.history, left.history)
        self.assertEqual(child.active_declaration, left.active_declaration)
        self.execute(child, 'next', 'child')
        self.assertTrue(child.history.startswith(left.history))
        self.assertEqual(child.latest_feedback[0]['parent_feedback'], ['left'])
        self.assertEqual(left.latest_feedback[0]['id'], 'left')

    def test_mutable_state_preserves_definitions_values_and_event_references(self):
        self.execute(self.execution, 'first', 'exact observation')
        self.execution.values['choice'] = 'left'
        self.execution.fields['choice'] = self.field
        original = self.execution.state()
        before = deepcopy(original)
        packed = history_state(original)
        self.assertEqual(original, before)
        self.assertEqual(packed['execution_order'], ['first'])
        definition, value = packed['computed_values'][0]
        self.assertEqual({**packed['field_definitions'][definition], 'value': value},
                         original['computed_fields'][0])
        self.assertEqual(packed['current_path'], original['current_path'])
        self.assertEqual(packed['field_evidence'], original['field_evidence'])

    def test_frontier_changes_do_not_relocate_or_duplicate_event_records(self):
        self.execute(self.execution, 'first', 'common')
        child = self.execution.fork()
        self.execute(child, 'child', 'complete child observation')
        branch = child.state()
        branch.pop('execution_events')
        state = {'execution_events': child.tool_observations,
                 'declarations': {'d0': {'fields': [self.field]}},
                 'branches': {'child': {'state': branch, 'declaration_id': 'd0', 'done': False}}}
        before = deepcopy(state)
        shared, choices = history_frontier(state)
        self.assertEqual(state, before)
        self.assertEqual(shared['execution_order'], ['first', 'child'])
        self.assertEqual(shared['declarations'], state['declarations'])
        self.assertEqual(choices['child']['state']['current_path'], ['first', 'child'])
        self.assertNotIn('execution_events', choices['child'])
        self.assertNotIn('execution_events', shared)
        self.assertNotIn('observations', shared)
        self.assertEqual([json.loads(line) for line in event_history(state['execution_events']).splitlines()],
                         state['execution_events'])

    def test_controller_prefix_ends_at_history_without_mutable_trailer(self):
        self.execute(self.execution, 'first', 'full observation')
        template = load_prompt('jevspawn.controller')['prefix_template']
        prefix = template.format(context=self.execution.query, history=self.execution.history)
        self.execute(self.execution, 'second', 'next observation')
        extended = template.format(context=self.execution.query, history=self.execution.history)
        self.assertTrue(extended.startswith(prefix))
        self.assertTrue(prefix.endswith(event_history(self.execution.tool_observations[:1])))


if __name__ == '__main__':
    unittest.main()
