from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from baselines.common.errors import TaskLimitError
from baselines.common.transition_budget import TransitionBudget
from baselines.tool_agents.tools import ActionError


class NativeEnvironment:
    def __init__(self):
        self.state = []
        self.answer = None
        self.done = False
        self.tool_timings = []
        self.actions = []
        self.tools = ['act', 'finish']
        self.input_schemas = {'act': {'type': 'string'},
                              'finish': {'type': 'object', 'required': ['result'],
                                         'properties': {'result': {'type': 'string'}}}}

    def fork(self):
        return deepcopy(self)

    def execute(self, name, arguments):
        if name == 'finish':
            self.answer, self.done = arguments, True
        else:
            self.state.append(arguments)
        return deepcopy(self.state), self.done

    def observe(self, name, arguments):
        output, done = self.execute(name, arguments)
        return {'output': output, 'done': done}


class TransitionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.root = TransitionBudget(NativeEnvironment(), 2, 'finish', 0, [])

    def test_parallel_siblings_share_depth_not_summed_credits(self):
        left, right = self.root.fork(), self.root.fork()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda branch: branch.observe('act', 'step'), [left, right]))
        self.assertEqual([left.transition_depth, right.transition_depth], [1, 1])
        self.assertEqual(self.root.transition_depth, 0)
        self.assertEqual(self.root.transitions, [{'depth': 1, 'tool': 'act'}] * 2)
        self.assertEqual(results, [{'output': ['step'], 'done': False}] * 2)

    def test_serial_transition_inherits_depth_and_isolates_state(self):
        parent = self.root.fork()
        parent.execute('act', 'first')
        child = parent.fork()
        child.execute('act', 'second')
        self.assertEqual(child.transition_depth, 2)
        self.assertEqual(child.state, ['first', 'second'])
        self.assertEqual(parent.state, ['first'])
        self.assertEqual(self.root.state, [])

    def test_model_scoring_and_summary_access_do_not_consume_turns(self):
        for _ in range(8):
            self.assertEqual(self.root.state, [])
        self.assertEqual(self.root.transition_depth, 0)
        self.assertEqual(self.root.transitions, [])

    def test_exhausted_branch_can_still_submit(self):
        self.root.execute('act', 'first')
        self.root.observe('act', 'second')
        with self.assertRaises(TaskLimitError):
            self.root.execute('act', 'third')
        self.assertEqual(self.root.state, ['first', 'second'])
        self.root.observe('finish', {'result': 'submitted'})
        self.assertTrue(self.root.done)
        self.assertEqual(self.root.answer, {'result': 'submitted'})
        self.assertEqual(self.root.transition_depth, 2)

    def test_parser_failure_without_native_observation_is_uncharged(self):
        with patch.object(self.root._environment, 'execute', side_effect=ValueError('Invalid argument format.')):
            with self.assertRaises(ValueError):
                self.root.execute('act', 'invalid')
        self.assertEqual(self.root.transition_depth, 0)
        self.assertEqual(self.root.transitions, [])

    def test_invalid_arguments_return_feedback_without_native_mutation(self):
        observation, done = self.root.execute('act', {'wrong': 'shape'})
        self.assertFalse(done)
        self.assertEqual(json.loads(observation)['error']['type'], 'ValidationError')
        self.assertEqual(self.root.state, [])
        self.assertEqual(self.root.transition_depth, 1)
        self.root.execute('act', 'corrected')
        self.assertEqual(self.root.state, ['corrected'])

    def test_unknown_tool_feedback_is_shared_by_observe_and_forks(self):
        child = self.root.fork()
        result = child.observe('missing_tool', {})
        self.assertEqual(result['error']['type'], 'ActionError')
        self.assertFalse(result['done'])
        self.assertEqual(child.transition_depth, 1)
        self.assertEqual(self.root.transition_depth, 0)
        self.assertEqual(self.root.state, [])
        self.assertEqual(child.state, [])

    def test_explicit_native_action_error_is_feedback_and_charged(self):
        with patch.object(self.root._environment, 'execute', side_effect=ActionError('Illegal move.')):
            observation, done = self.root.execute('act', 'invalid')
        self.assertEqual(json.loads(observation)['error']['message'], 'Illegal move.')
        self.assertFalse(done)
        self.assertEqual(self.root.transition_depth, 1)
        self.assertEqual(self.root.state, [])

    def test_invalid_submission_and_parser_feedback_cannot_retry_forever(self):
        self.root.observe('finish', {})
        self.root.reject(None, 'broken frame', ActionError('Missing action frame.'))
        self.assertEqual(self.root.transition_depth, self.root.max_turns)
        with self.assertRaises(TaskLimitError):
            self.root.observe('finish', {})
        self.root.execute('finish', {'result': 'corrected'})
        self.assertTrue(self.root.done)

    def test_infrastructure_failures_are_not_model_observations(self):
        for error in (RuntimeError('GPU failure'), OSError('Storage failure'), KeyError('implementation bug'),
                      json.JSONDecodeError('Corrupt internal artifact', 'invalid', 0)):
            with patch.object(self.root._environment, 'execute', side_effect=error):
                with self.assertRaises(type(error)):
                    self.root.execute('act', 'valid')
        self.assertEqual(self.root.transition_depth, 0)
        self.assertEqual(self.root.actions, [])

    def test_same_branch_native_mutations_keep_serial_depth(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda value: self.root.execute('act', value), ['first', 'second']))
        self.assertEqual(self.root.transition_depth, 2)
        self.assertEqual([item['depth'] for item in self.root.transitions], [1, 2])
        self.assertEqual(set(self.root.state), {'first', 'second'})

    def test_result_assignments_reach_native_environment(self):
        native = self.root._environment
        self.root.answer = {'result': 'selected'}
        self.root.done = True
        self.root.tool_timings = [{'tool': 'act'}]
        self.assertEqual(native.answer, self.root.answer)
        self.assertEqual(native.done, self.root.done)
        self.assertEqual(native.tool_timings, self.root.tool_timings)


if __name__ == '__main__':
    unittest.main()
