import json
import unittest
from unittest.mock import Mock

from baselines.common.environment import TaskEnvironment
from baselines.common.errors import InvalidOutputError
from project_paths import ROOT


FIXTURE = json.loads((ROOT / 'tests/algo/finite_control/fixtures.json').read_text())['tool_feedback']


class ToolFeedbackTests(unittest.TestCase):
    def setUp(self):
        tasks = [json.loads(line) for line in (ROOT / FIXTURE['qualification_tasks']).read_text().splitlines()]
        self.task = next(task for task in tasks if task['task_id'] == FIXTURE['task_id'])
        settings = json.loads((ROOT / FIXTURE['environment']).read_text())
        self.environment = TaskEnvironment(self.task, settings, ROOT, deadline=lambda: None)

    def test_actual_rejected_expression_becomes_observed_error(self):
        expression = self.task
        for key in FIXTURE['invalid_input_path']:
            expression = expression[key]
        arguments = {'expression': expression}
        observation = self.environment.observe(FIXTURE['tool'], arguments)
        self.assertEqual(observation['error']['type'], 'ActionError')
        self.assertIn('Invalid calculator expression', observation['error']['message'])
        self.assertEqual(self.environment.actions, [
            {'tool': FIXTURE['tool'], 'arguments': arguments, 'result': observation}])
        self.assertIsNone(self.environment.answer)

    def test_real_success_preserves_the_calculation(self):
        result = self.environment.observe(FIXTURE['tool'], {'expression': FIXTURE['valid_expression']})
        self.assertEqual(result, {'value': FIXTURE['expected_value']})

    def test_runtime_failures_are_not_tool_observations(self):
        for error in (TimeoutError, RuntimeError):
            with self.subTest(error=error):
                self.environment.execute = Mock(side_effect=error)
                with self.assertRaises(error):
                    self.environment.observe(FIXTURE['tool'], {'expression': FIXTURE['valid_expression']})
        self.assertEqual(self.environment.actions, [])

    def test_unlocatable_span_is_invalid_output_without_submission(self):
        case = FIXTURE['invalid_submission']
        task = next(json.loads(line) for line in (ROOT / FIXTURE['qualification_tasks']).read_text().splitlines()
                    if json.loads(line)['task_id'] == case['task_id'])
        environment = TaskEnvironment(task, self.environment.settings, ROOT, deadline=lambda: None)
        with self.assertRaises(InvalidOutputError):
            environment.execute('finish', case['answer'])
        self.assertIsNone(environment.answer)
        self.assertEqual(environment.actions, [])


if __name__ == '__main__':
    unittest.main()
