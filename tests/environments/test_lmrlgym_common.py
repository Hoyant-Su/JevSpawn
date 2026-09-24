from copy import deepcopy
import json
import time
import unittest

import jsonschema

from baselines.common.evaluate import score_task
from baselines.common.schema_environment import environment_definition
from environments.maze_evaluation import MazeScorer
from jev_spawn.infra.configuration import resolve_symbol
from project_paths import ROOT


class LMRLCommonTest(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((ROOT / 'tests/environments/lmrlgym_common_cases.json').read_text())
        self.spec = json.loads((ROOT / self.fixture['primary_specification']).read_text())
        self.factory, _ = environment_definition(self.spec)
        self.runtime = json.loads((ROOT / self.fixture['runtime']).read_text())
        self.scorer = MazeScorer(self.fixture['scorer_settings'])

    def environment(self, task):
        return self.factory(task, {}, ROOT / self.fixture['output'], deadline=lambda: None)

    def test_all_official_source_ids_and_native_transitions(self):
        tasks = [json.loads(line) for line in (ROOT / self.fixture['source_tasks']).read_text().splitlines()]
        identities = json.loads((ROOT / self.fixture['source_identities']).read_text())
        self.assertEqual([task['task_id'] for task in tasks], [item['item_id'] for item in identities])
        for task in tasks:
            task['answer_schema'] = self.runtime['answer_schema']
            env = self.environment(task)
            direct, history = deepcopy(env.native), deepcopy(env.history)
            for action in self.fixture['comparison_actions']:
                actual = env.observe(self.fixture['action_tool'], {self.fixture['action_field']: action})
                history, reward, done = direct.step(history + (env.text_type(env.native_actions[action], True),))
                self.assertEqual(actual, {'observation': history[self.runtime['latest_history_index']].text,
                                         'reward': reward, 'done': done})
                self.assertEqual(tuple(env.native.position), tuple(direct.position))

    def test_blocked_moves_native_limit_and_replay_failures(self):
        task = self.scorer.tasks[self.fixture['blocked_task_id']]
        env = self.environment(task)
        position = tuple(env.native.position)
        arguments = {self.fixture['action_field']: self.fixture['blocked_action']}
        for _ in range(self.runtime['environment']['max_steps']):
            env.observe(self.fixture['action_tool'], arguments)
            self.assertEqual(tuple(env.native.position), position)
            self.assertFalse(env.done)
            self.assertIsNone(env.answer)
        env.observe(self.fixture['action_tool'], arguments)
        self.assertTrue(env.done)
        self.assertFalse(env.evaluate(env.answer))
        result = {'task_id': task['task_id'], 'status': self.fixture['status'],
                  'answer': env.answer, 'actions': env.actions}
        detail = self.scorer.score(task, result, task)
        self.assertTrue(detail['replayed'])
        self.assertFalse(detail['native_success'])
        self.assertEqual(detail['action_count'], len(env.actions))
        self.assertEqual(detail['total_reward'], sum((row['result']['reward'] for row in env.actions),
                                                    self.fixture['reward_sum_start']))
        changed = deepcopy(result)
        changed['answer'] = {self.fixture['answer_field']: self.fixture['empty_actions']}
        with self.assertRaisesRegex(AssertionError, 'contradicts native terminal'):
            self.scorer.score(task, changed, task)
        changed = deepcopy(result)
        changed['actions'][self.runtime['latest_history_index']]['result']['observation'] = env.initial_observation
        with self.assertRaisesRegex(AssertionError, 'differs from native replay'):
            self.scorer.score(task, changed, task)

    def test_goal_step_and_invalid_action(self):
        tasks = {row['task_id']: row for row in map(json.loads,
            (ROOT / self.fixture['source_tasks']).read_text().splitlines())}
        task = tasks[self.fixture['goal_task_id']]
        task['answer_schema'] = self.runtime['answer_schema']
        env = self.environment(task)
        with self.assertRaises(jsonschema.ValidationError):
            env.observe(self.fixture['action_tool'], {self.fixture['action_field']: self.fixture['invalid_action']})
        self.assertEqual(env.actions, [])
        action = self.fixture['goal_action']
        result = env.observe(self.fixture['action_tool'], {self.fixture['action_field']: action})
        self.assertTrue(result['done'])
        self.assertEqual(tuple(env.native.position), tuple(env.native.goal))
        self.assertEqual(env.answer, {self.fixture['answer_field']: [action]})
        self.assertTrue(env.evaluate(env.answer))

    def test_noncompleted_statuses_remain_in_denominator(self):
        task = self.scorer.tasks[self.fixture['blocked_task_id']]
        for status in self.fixture['scorer_settings']['noncompleted_statuses']:
            result = {'task_id': task['task_id'], 'status': status, 'answer': None}
            detail = self.scorer.score(task, result, task)
            self.assertEqual(detail['score'], self.fixture['scorer_settings']['noncompleted_score'])
            self.assertFalse(detail['replayed'])
            self.assertIsNone(detail['native_success'])

    def test_common_context_native_steps_fork_and_replay(self):
        fixture = json.loads((ROOT / 'tests/environments/lmrlgym_common_cases.json').read_text())
        controls = {row['task_id']: row['actions'] for row in fixture['controls']}
        scorer = MazeScorer(fixture['scorer_settings'])
        reports = []
        for specification in fixture['specifications']:
            spec = json.loads((ROOT / specification).read_text())
            factory, execution = environment_definition(spec)
            native_class = resolve_symbol(execution['class'])
            tools = json.loads((ROOT / spec['environment']).read_text())
            tasks = [json.loads(line) for line in (ROOT / spec['tasks']).read_text().splitlines()]
            self.assertEqual(len(tasks), spec['task_count'])
            for task in tasks:
                directory = ROOT / fixture['output'] / task['task_id']
                directory.mkdir(parents=True, exist_ok=True)
                env = factory(task, tools, directory, deadline=lambda: None)
                self.assertEqual(env.reset(), task['context'])
                self.assertEqual(env.context(True, {}), env.context(False, {}))
                self.assertIn(task['instruction'], env.reset())
                self.assertFalse(hasattr(env, 'decision_frame'))
                branch, direct = env.fork(), deepcopy(env.native)
                history = deepcopy(env.history)
                initial_position = tuple(env.native.position)
                trace = []
                started = time.perf_counter()
                for action in controls[task['task_id']]:
                    actual = branch.observe(fixture['action_tool'], {fixture['action_field']: action})
                    history, reward, done = direct.step(history + (env.text_type(env.native_actions[action], True),))
                    expected = {'observation': history[env.configuration['latest_history_index']].text,
                                'reward': reward, 'done': done}
                    self.assertEqual(actual, expected)
                    self.assertEqual(tuple(branch.native.position), tuple(direct.position))
                    trace.append({'action': action, **actual})
                self.assertEqual(tuple(env.native.position), initial_position)
                self.assertEqual(env.actions, [])
                self.assertTrue(branch.done)
                answer = {fixture['answer_field']: controls[task['task_id']]}
                self.assertEqual(branch.answer, answer)
                self.assertTrue(branch.evaluate(answer))
                artifact = directory / fixture['artifact_name']
                result = {'task_id': task['task_id'], 'status': fixture['status'], 'answer': answer,
                          'actions': branch.actions, 'tool_timings': branch.tool_timings,
                          'elapsed_seconds': time.perf_counter() - started}
                artifact.write_text(json.dumps(result))
                correct = score_task(task, artifact, native_class, execution['parameters'], tools, directory)
                replay_score = scorer.score(task, result, task)
                self.assertTrue(correct['correct'])
                self.assertTrue(replay_score['native_success'])
                submitted = env.fork()
                submitted.observe(fixture['submission_tool'], answer)
                self.assertTrue(submitted.evaluate(submitted.answer))
                env.observe(fixture['submission_tool'], {fixture['answer_field']: fixture['empty_actions']})
                incorrect_artifact = directory / fixture['incorrect_artifact_name']
                incorrect_artifact.write_text(json.dumps({'task_id': task['task_id'], 'status': fixture['status'],
                    'answer': env.answer, 'actions': env.actions, 'tool_timings': env.tool_timings,
                    'elapsed_seconds': time.perf_counter() - started}))
                incorrect = score_task(task, incorrect_artifact, native_class, execution['parameters'], tools, directory)
                self.assertFalse(incorrect['correct'])
                reports.append({'specification': specification, 'task_id': task['task_id'],
                    'model_inference': False, 'control_policy': 'Test-only shortest path on native maze; never given to the model.',
                    'native_steps': len(trace), 'context_matches': True, 'fork_isolation': True,
                    'correct_control_score': correct['score'], 'incorrect_control_score': incorrect['score'],
                    'maze_scorer_replayed': replay_score['replayed'], 'trace': trace})
        (ROOT / fixture['report']).write_text(json.dumps({'model_inference': False,
            'protocol_origin': fixture['protocol_origin'], 'checks': reports}, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
