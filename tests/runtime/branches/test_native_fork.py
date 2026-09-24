import json
from pathlib import Path
import unittest

import torch
from unittest.mock import Mock

from environments.maze import MazeEnvironment
from jev_spawn.runtime.branches import Branch, spawn, spawn_scored
from jev_spawn.runtime.query_execution import QueryExecution


ROOT = Path(__file__).resolve().parents[3]


class NativeForkTest(unittest.TestCase):
    def test_equivalent_actions_execute_once_per_parent(self):
        settings = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())
        native = Mock(tool_timings=[])
        native.fork.return_value = native
        native.observe.return_value = {'received': '$literal'}
        execution = QueryExecution('Action identity test.', Mock(), 'test', settings)
        parents = {name: Branch(execution, native, None) for name in ['first', 'second']}
        proposals = [
            {'id': 'a', 'parent': 'first', 'values': {'text': '$$literal'},
             'action': {'tool': 'echo', 'arguments': {'text': '${text}', 'other': 'x'}}},
            {'id': 'b', 'parent': 'first', 'values': {},
             'action': {'tool': 'echo', 'arguments': {'other': 'x', 'text': '$$literal'}}},
            {'id': 'c', 'parent': 'second', 'values': {},
             'action': {'tool': 'echo', 'arguments': {'text': '$$literal', 'other': 'x'}}},
            {'id': 'd', 'parent': 'first', 'values': {},
             'action': {'tool': 'echo', 'arguments': {'text': 'different', 'other': 'x'}}}]
        children = spawn(parents, proposals, 2)
        self.assertEqual(list(children), ['a', 'c', 'd'])
        self.assertEqual(native.fork.call_count, 3)
        self.assertEqual(native.observe.call_count, 3)
        self.assertEqual([event['id'] for child in children.values()
                          for event in child.execution.latest_feedback], ['a', 'c', 'd'])
        self.assertEqual(execution.tool_observations, [])

    def test_action_template_is_materialized_once(self):
        settings = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())
        native = Mock(tool_timings=[])
        native.fork.return_value = native
        native.observe.return_value = {'received': '$literal'}
        execution = QueryExecution('Literal transport test.', Mock(), 'test', settings)
        proposal = {'id': 'child', 'parent': 'root', 'values': {},
                    'action': {'tool': 'echo', 'arguments': {'text': '$$literal'}}}
        children = spawn({'root': Branch(execution, native, None)}, [proposal], 1)
        native.observe.assert_called_once_with('echo', {'text': '$literal'})
        event, = children['child'].execution.latest_feedback
        self.assertEqual(event['action']['arguments'], {'text': '$literal'})

    def test_actions_execute_from_same_parent(self):
        contract = json.loads((ROOT / 'configs/jevspawn/interactive.json').read_text())['environment']
        factory = json.loads((ROOT / contract['factory']).read_text())
        task = json.loads((ROOT / contract['tasks']).read_text().splitlines()[0])
        settings = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())
        native = MazeEnvironment(task, {}, contract['directory'], deadline=Mock(), **factory['parameters'])
        query = native.context(False, native.configuration['serialization'])
        parent = Branch(QueryExecution(query, Mock(), task['task_id'], settings), native, None)
        parent.execution.fields['direction'] = {'id': 'direction', 'question': 'Movement direction',
                                                 'values': list(native.native_actions)}
        parent.execution.values['direction'] = next(iter(native.native_actions))
        initial = native.observation
        proposals = [{'id': action, 'parent': task['task_id'], 'values': {'direction': action},
                      'action': {'id': action, 'tool': contract['tool'],
                                 'arguments': {contract['argument']: '${direction}'}}}
                     for action in native.native_actions]
        children = spawn({task['task_id']: parent}, proposals, len(proposals))
        self.assertEqual(native.observation, initial)
        self.assertEqual(native.actions, [])
        self.assertEqual(parent.execution.latest_feedback, [])
        self.assertGreater(len({child.environment.observation for child in children.values()}), 1)
        for proposal in proposals:
            child = children[proposal['id']]
            reference = native.fork()
            resolved = child.execution.materialize(proposal['action'])
            actual = reference.observe(resolved['tool'], resolved['arguments'])
            event, = child.execution.latest_feedback
            self.assertEqual(event['value'], actual)
            self.assertEqual(event['action'], resolved)
            self.assertEqual(resolved['arguments'][contract['argument']], proposal['id'])
            self.assertEqual(event['parent_feedback'], [])
            self.assertEqual(child.execution.trace[-1]['tool_timings'], child.environment.tool_timings)

    def test_scored_spawn_uses_ranked_calls_and_real_feedback(self):
        contract = json.loads((ROOT / 'configs/jevspawn/interactive.json').read_text())['environment']
        factory = json.loads((ROOT / contract['factory']).read_text())
        task = json.loads((ROOT / contract['tasks']).read_text().splitlines()[0])
        settings = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())
        native = MazeEnvironment(task, {}, contract['directory'], deadline=Mock(), **factory['parameters'])
        query = native.context(False, native.configuration['serialization'])
        service = Mock()
        execution = QueryExecution(query, service, task['task_id'], settings)
        calls = list(native.native_actions)
        field = {'id': 'action', 'question': 'Which action advances this task?', 'values': calls}
        options = [settings['candidate_id'].format(index=index) for index in range(len(calls))]
        ranked = list(reversed(options))
        service.decide.return_value = [{'id': json.dumps([task['task_id'], field['id']]),
            'choice': ranked[0], 'option_ids': options, 'ranked_option_ids': ranked}]
        service.extend.return_value = ([0] * len(calls), [options.index(option) for option in ranked],
                                       torch.full((len(calls),), -1.0))
        children = spawn_scored({task['task_id']: Branch(execution, native,
                                {'fields': [field], 'action': {'tool': contract['tool'],
                                     'arguments': {contract['argument']: '${action}'}}})},
                                len(calls), len(calls), iter(options))
        service.decide.assert_called_once()
        self.assertEqual(native.actions, [])
        for option, child in zip(ranked, children.values(), strict=True):
            expected = calls[options.index(option)]
            event, = child.execution.latest_feedback
            self.assertEqual(event['action']['arguments'], {contract['argument']: expected})
            self.assertEqual(child.execution.values[field['id']], expected)
            self.assertEqual(event['value']['observation'], child.environment.observation)
            self.assertEqual(child.execution.trace[-2]['tool_timings'], child.environment.tool_timings)


if __name__ == '__main__':
    unittest.main()
