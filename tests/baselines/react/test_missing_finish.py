import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from baselines.common.errors import TaskLimitError
from baselines.common.react import solve


FIXTURE = json.loads(Path('tests/fixtures/baselines/common/failure_semantics.json').read_text())['react']


class MissingFinishTests(unittest.TestCase):
    def run_return(self, answer, deadline):
        environment = SimpleNamespace(answer=answer, deadline=deadline, actions=[])
        core = Mock(return_value=(FIXTURE['upstream_reward'], FIXTURE['upstream_result']))
        complete = Mock(side_effect=AssertionError('This CPU boundary fixture must not run inference.'))
        with patch('baselines.common.react.original_function', return_value=core):
            result = solve(FIXTURE['task'], environment, complete, FIXTURE['settings'], FIXTURE['prompts'])
        complete.assert_not_called()
        return result

    def test_missing_finish_is_limit_failure(self):
        deadline = Mock(return_value=FIXTURE['remaining_seconds'])
        with self.assertRaisesRegex(TaskLimitError, FIXTURE['missing_finish_message']):
            self.run_return(None, deadline)
        deadline.assert_called_once_with()

    def test_expired_missing_finish_remains_timeout(self):
        with self.assertRaisesRegex(TimeoutError, FIXTURE['deadline_message']):
            self.run_return(None, Mock(side_effect=TimeoutError(FIXTURE['deadline_message'])))

    def test_accepted_finish_preserves_upstream_return(self):
        result = self.run_return(FIXTURE['accepted_answer'], Mock(return_value=FIXTURE['remaining_seconds']))
        self.assertIs(result['answer'], FIXTURE['accepted_answer'])
        self.assertEqual(result['trajectory'], FIXTURE['upstream_result']['traj'])
        self.assertEqual(result['core_calls'], FIXTURE['upstream_result']['n_calls'])
        self.assertEqual(result['core_format_retries'], FIXTURE['upstream_result']['n_badcalls'])


if __name__ == '__main__':
    unittest.main()
