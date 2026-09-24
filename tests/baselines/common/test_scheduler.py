from threading import Event
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from baselines.common.deadlines import SampleDeadlines
from baselines.common.errors import FlowInputError, InputLimitError, InvalidOutputError, TaskLimitError
from baselines.common.jevspawn import solve as solve_jevspawn
from baselines.common.scheduler import run_tasks


class RollingAdmissionTest(unittest.TestCase):
    def test_next_root_starts_before_slow_root_finishes(self):
        third_started = Event()
        committed = []

        def solve(task):
            if task['task_id'] == 'slow':
                self.assertTrue(third_started.wait(timeout=2))
            if task['task_id'] == 'third':
                third_started.set()
            return {'task_id': task['task_id'], 'answer': task['task_id']}

        tasks = [{'task_id': identity} for identity in ['slow', 'fast', 'third']]
        rows, elapsed = run_tasks(tasks, solve, 2, SampleDeadlines(5),
                                  lambda index, row: committed.append(row['task_id']))
        self.assertEqual([row['task_id'] for row in rows], ['slow', 'fast', 'third'])
        self.assertEqual(len(committed), 3)
        self.assertEqual(len(set(committed)), 3)
        self.assertTrue(all(row['status'] == 'completed' for row in rows))
        self.assertGreater(elapsed, 0)

    def test_timeout_retains_identity_and_failure(self):
        def timeout(task):
            raise TimeoutError('Task exceeded its deadline')

        rows, _ = run_tasks([{'task_id': 'expired'}], timeout, 1,
                            SampleDeadlines(1), lambda index, row: None)
        self.assertEqual(rows[0]['task_id'], 'expired')
        self.assertEqual(rows[0]['status'], 'timeout')
        self.assertIsNone(rows[0]['answer'])

    def test_jevspawn_failure_preserves_partial_trace_and_failure_status(self):
        task = {'task_id': 'failed', 'context': 'Official context'}
        settings = {'rollout': {'prompts': 'unused'}, 'terminal_answer': {},
                    'max_turns': 3, 'max_new_tokens': 20, 'temperature': 0.0}
        complete = SimpleNamespace(func=SimpleNamespace(__self__=object()))
        actual_round = {'turn': 0, 'children': {'branch': [
            {'action': 'attempted action', 'observation': {'accepted': False}}]}}
        cases = [(TimeoutError, 'timeout'), (InvalidOutputError, 'invalid_output'),
                 (FlowInputError, 'invalid_output'), (InputLimitError, 'limit_exceeded'),
                 (TaskLimitError, 'limit_exceeded')]
        for error_type, status in cases:
            with self.subTest(error_type=error_type):
                failure = error_type('Actual failure')

                def fail(query, **kwargs):
                    kwargs['trace']['rounds'] = [actual_round]
                    raise failure

                def solve(task):
                    return solve_jevspawn(task, object(), complete, settings, {})

                committed = []
                with patch('baselines.common.jevspawn.solve_branching', side_effect=fail), \
                        patch('baselines.common.jevspawn.load_prompt', return_value={}):
                    rows, _ = run_tasks([task], solve, 1, SampleDeadlines(5),
                                        lambda index, row: committed.append(row))
                self.assertEqual(rows[0]['status'], status)
                self.assertIsNone(rows[0]['answer'])
                self.assertEqual(rows[0]['error'], 'Actual failure')
                self.assertIs(rows[0]['trace'], failure.trace)
                self.assertEqual(rows[0]['trace'], {'query': task['context'], 'rounds': [actual_round]})
                self.assertIs(committed[0], rows[0])

    def test_runtime_errors_abort(self):
        def fail(task):
            raise RuntimeError('Required inference failed')

        with self.assertRaisesRegex(RuntimeError, 'Required inference failed'):
            run_tasks([{'task_id': 'broken'}], fail, 1,
                      SampleDeadlines(1), lambda index, row: None)


if __name__ == '__main__':
    unittest.main()
