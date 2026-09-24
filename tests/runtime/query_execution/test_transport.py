import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.rollout.branching import branch_state
from jev_spawn.runtime.branches import Branch
from jev_spawn.runtime.state import frontier_view

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/jevspawn/query_execution.json').read_text())
FIXTURE = json.loads(Path(__file__).with_name('transport.json').read_text())

def unpack_request(request):
    state = json.loads(request['state'])
    state['computed_fields'] = [{**state['field_definitions'][index], 'value': value}
                               for index, value in state['computed_values']]
    state['execution_events'] = [json.loads(line) for line in request['history'].splitlines()]
    return state


class QueryTransportTests(unittest.TestCase):
    """Transport tests with scripted service responses, not model-quality tests."""

    def setUp(self):
        self.service = Mock()
        self.execution = QueryExecution(FIXTURE['query'], self.service, FIXTURE['task_id'], SETTINGS)

    def test_batch_identity_and_next_turn_meanings(self):
        self.service.decide.side_effect = FIXTURE['decisions']
        for batch in FIXTURE['batches']:
            self.execution.execute(batch)
        for call, batch in zip(self.service.decide.call_args_list, FIXTURE['batches'], strict=True):
            requests, = call.args
            self.assertEqual([field['id'] for field in requests], [field['id'] for field in batch])
            self.assertTrue(all(field['context'] == FIXTURE['query'] for field in requests))
            self.assertEqual(call.kwargs, {'task_id': FIXTURE['task_id']})
        requests, = self.service.decide.call_args.args
        for field in requests:
            state = unpack_request(field)
            self.assertEqual(state['computed_fields'], FIXTURE['second_turn_observation'])
            self.assertEqual(state['pending_fields'], FIXTURE['batches'][-1])
        self.assertEqual(self.execution.materialize(FIXTURE['answer_template']), FIXTURE['answer'])

    def test_missing_outputs_cannot_commit_partial_batch(self):
        batch, *_ = FIXTURE['batches']
        self.service.decide.return_value = []
        with self.assertRaises(ValueError):
            self.execution.execute(batch)
        self.assertEqual(self.execution.observation(), [])
        self.assertEqual(self.execution.trace, [])

    def test_reordered_outputs_preserve_field_identity(self):
        batch, *_ = FIXTURE['batches']
        decisions, *_ = FIXTURE['decisions']
        self.service.decide.return_value = list(reversed(decisions))
        self.execution.execute(batch)
        self.assertEqual(self.execution.observation(), FIXTURE['second_turn_observation'])

    def test_actual_tool_feedback_reaches_next_spawn(self):
        first, second = FIXTURE['batches']
        self.service.decide.side_effect = FIXTURE['decisions']
        self.execution.execute(first)
        tool_batch = Mock(return_value=FIXTURE['tool_results'])
        expected_calls = self.execution.materialize(FIXTURE['tool_calls'])
        self.execution.execute_actions(FIXTURE['tool_calls'], tool_batch)
        tool_batch.assert_called_once_with(expected_calls)
        before_spawn = self.execution.observation()
        self.execution.execute(second)
        requests, = self.service.decide.call_args.args
        for request in requests:
            state = unpack_request(request)
            self.assertEqual(state['computed_fields'] + state['execution_events'], before_spawn)
        event, = self.execution.tool_observations
        self.assertEqual(event['value'], FIXTURE['tool_results'][event['id']])
        self.assertEqual(self.execution.materialize(FIXTURE['tool_reference']), event['value'])

    def test_forks_keep_feedback_separate_in_one_model_batch(self):
        first, second = FIXTURE['batches']
        initial, subsequent = FIXTURE['decisions']
        self.service.decide.return_value = initial
        self.execution.execute(first)
        left, right = self.execution.fork(), self.execution.fork()
        left.execute_actions(FIXTURE['tool_calls'], Mock(return_value=FIXTURE['tool_results']))
        self.assertEqual(right.latest_feedback, [])
        self.assertEqual(self.execution.latest_feedback, [])
        branches = {'left': left, 'right': right}
        self.service.reset_mock()
        self.service.decide.return_value = [
            {**decision, 'id': json.dumps([identity, decision['id']])}
            for identity in reversed(branches) for decision in subsequent]
        values = QueryExecution.evaluate_branches(branches, {identity: second for identity in branches})
        self.service.decide.assert_called_once()
        requests, = self.service.decide.call_args.args
        self.assertEqual(len(requests), len(branches) * len(second))
        for request in requests:
            identity, field = json.loads(request['id'])
            state = unpack_request(request)
            self.assertEqual(state['current_feedback'],
                             [event['id'] for event in branches[identity].latest_feedback])
            self.assertIn(field, values[identity])
        self.assertEqual(left.trace[-1]['based_on_feedback'], list(FIXTURE['tool_results']))
        self.assertEqual(right.trace[-1]['based_on_feedback'], [])
        self.assertEqual(self.execution.observation(), FIXTURE['second_turn_observation'])

    def test_local_branches_keep_real_feedback_while_frontier_sees_both(self):
        first, second = FIXTURE['batches']
        self.service.decide.return_value = FIXTURE['decisions'][0]
        self.execution.execute(first)
        self.execution.execute_actions(FIXTURE['tool_calls'], Mock(return_value=FIXTURE['tool_results']))
        branches = {'left': self.execution.fork(), 'right': self.execution.fork()}
        for identity, execution in branches.items():
            execution.execute_actions([{'id': identity, 'tool': 'record',
                'arguments': {'value': identity}}],
                lambda calls: {call['id']: {'observation': call['arguments']['value']} for call in calls})
        self.service.decide.return_value = [
            {**decision, 'id': json.dumps([identity, decision['id']])}
            for identity in branches for decision in FIXTURE['decisions'][1]]
        QueryExecution.evaluate_branches(branches, {identity: second for identity in branches})
        requests, = self.service.decide.call_args.args
        for request in requests:
            identity, field = json.loads(request['id'])
            state = unpack_request(request)
            expected = list(FIXTURE['tool_results']) + [identity]
            self.assertEqual([event['id'] for event in state['execution_events']], expected)
            self.assertEqual(state['current_path'], expected)
            self.assertEqual(state['execution_events'][-1]['value'], {'observation': identity})
            self.assertEqual(state['computed_fields'], FIXTURE['second_turn_observation'])
            self.assertEqual(branches[identity].field_evidence[field], expected)
            self.assertEqual(branches[identity].fork().state(), branches[identity].state())
        events = {event['id']: event for execution in branches.values()
                  for event in execution.tool_observations}
        shared, choices = frontier_view({'execution_events': list(events.values()), 'declarations': {},
            'branches': {identity: {'state': branch_state(Branch(execution, None, None)),
                         'declaration_id': None, 'done': False}
                         for identity, execution in branches.items()}})
        all_events = shared['execution_events'] + [event for choice in choices.values()
                                                   for event in choice['execution_events']]
        self.assertCountEqual([event['id'] for event in all_events], list(events))
        for identity in branches:
            self.assertEqual(choices[identity]['execution_events'][-1]['value'], {'observation': identity})
        self.assertEqual([event['id'] for event in self.execution.state()['execution_events']],
                         list(FIXTURE['tool_results']))

    def test_unknown_reference_cannot_become_an_answer(self):
        with self.assertRaises(KeyError):
            self.execution.materialize(FIXTURE['answer_template'])

if __name__ == '__main__':
    unittest.main()
