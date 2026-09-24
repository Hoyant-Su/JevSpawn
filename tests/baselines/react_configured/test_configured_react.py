from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import unittest

from baselines.common.configured_react import solve
from baselines.common.errors import TaskLimitError
from baselines.common.react import EnvironmentAdapter, solve as original_solve
from baselines.official_react.adapter import original_function as original_extraction
from baselines.official_react.configured_adapter import original_function
from project_paths import ROOT


class FixtureEnvironment:
    def __init__(self, fixture, terminal_after):
        self.fixture, self.terminal_after = fixture, terminal_after
        self.actions = []
        self.answer = None

    def reset(self):
        return self.fixture['initial_observation']

    def deadline(self):
        pass

    def execute(self, name, arguments):
        assert name == 'move' and arguments == {}
        self.actions.append({'tool': name, 'arguments': arguments})
        done = len(self.actions) == self.terminal_after
        if done:
            self.answer = {'completion': self.fixture['completion']}
        observation = self.fixture['terminal_observation'] if done else self.fixture['ordinary_observation']
        return observation, done


class ConfiguredReactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((Path(__file__).parent / 'config.json').read_text())
        cls.fixture = config['fixture']
        cls.settings = {**config['method']['settings'],
                        'notebook': str(ROOT / config['method']['settings']['notebook']),
                        'max_new_tokens': cls.fixture['max_new_tokens'],
                        'temperature': cls.fixture['temperature']}
        cls.prompts = {key: cls.fixture[key] for key in ['instruction', 'invalid_action']}
        cls.task = {'task_id': cls.fixture['task_id']}

    def completion(self, first_output=None):
        calls = []

        def complete(messages, max_tokens, temperature, stop):
            prompt = messages[0]['content']
            calls.append(prompt)
            if first_output is not None and len(calls) == 1:
                return [first_output]
            if re.search(r'Action \d+:$', prompt):
                return [self.fixture['action']]
            index = re.search(r'Thought (\d+):$', prompt).group(1)
            return [self.fixture['thought'] + '\nAction ' + index + ': ' + self.fixture['action']]

        return complete

    def namespace(self, environment):
        adapter = EnvironmentAdapter(environment, self.prompts)
        complete = self.completion()
        return {'env': adapter,
                'llm': lambda prompt, stop: complete([{'content': prompt}],
                    self.settings['max_new_tokens'], self.settings['temperature'], stop)[0],
                'step': lambda env, action: env.step(action),
                'webthink_prompt': self.prompts['instruction']}

    def test_default_extraction_is_unchanged(self):
        first = original_extraction(self.settings['notebook'], self.namespace(
            FixtureEnvironment(self.fixture, self.fixture['short_terminal_after_actions'])))
        second = original_function(self.settings['notebook'], self.namespace(
            FixtureEnvironment(self.fixture, self.fixture['short_terminal_after_actions'])))
        self.assertEqual(first.__code__.co_code, second.__code__.co_code)
        self.assertEqual(first.__code__.co_consts, second.__code__.co_consts)
        self.assertEqual(first(idx=self.task['task_id'], to_print=False),
                         second(idx=self.task['task_id'], to_print=False))

    def test_only_declared_loop_bound_changes(self):
        original = original_extraction(self.settings['notebook'], self.namespace(
            FixtureEnvironment(self.fixture, self.fixture['short_terminal_after_actions'])))
        configured = original_function(self.settings['notebook'], self.namespace(
            FixtureEnvironment(self.fixture, self.fixture['short_terminal_after_actions'])),
            episode_action_limit=self.settings['episode_action_limit'],
            original_episode_range=self.settings['original_episode_range'])
        start, stop = self.settings['original_episode_range']
        expected = tuple(start + self.settings['episode_action_limit'] if value == stop else value
                         for value in original.__code__.co_consts)
        self.assertEqual(configured.__code__.co_consts, expected)
        self.assertEqual(configured.__code__.co_code, original.__code__.co_code)
        self.assertEqual(configured.__code__.co_names, original.__code__.co_names)

    def test_configured_episode_reaches_terminal_after_original_limit(self):
        environment = FixtureEnvironment(self.fixture, self.fixture['terminal_after_actions'])
        result = solve(self.task, environment, self.completion(), self.settings, self.prompts)
        self.assertEqual(len(result['actions']), self.fixture['terminal_after_actions'])
        self.assertEqual(result['core_calls'], self.fixture['terminal_after_actions'])
        self.assertEqual(result['core_format_retries'], 0)
        self.assertEqual(result['answer'], {'completion': self.fixture['completion']})
        self.assertIn(self.fixture['terminal_observation'], result['trajectory'])

    def test_format_retry_remains_original(self):
        limit = self.fixture['short_terminal_after_actions']
        old_env, new_env = FixtureEnvironment(self.fixture, limit), FixtureEnvironment(self.fixture, limit)
        with redirect_stdout(io.StringIO()):
            old = original_solve(self.task, old_env, self.completion(self.fixture['malformed_thought']),
                                 self.settings, self.prompts)
            new = solve(self.task, new_env, self.completion(self.fixture['malformed_thought']),
                        self.settings, self.prompts)
        self.assertEqual(new, old)
        self.assertEqual(new['core_format_retries'], 1)

    def test_invalid_action_behavior_remains_original(self):
        invalid = self.fixture['thought'] + '\nAction 1: ' + self.fixture['invalid_label_action']
        limit = self.fixture['short_terminal_after_actions']
        old_env, new_env = FixtureEnvironment(self.fixture, limit), FixtureEnvironment(self.fixture, limit)
        old = original_solve(self.task, old_env, self.completion(invalid), self.settings, self.prompts)
        new = solve(self.task, new_env, self.completion(invalid), self.settings, self.prompts)
        self.assertEqual(new, old)
        self.assertIn(self.fixture['invalid_action'], new['trajectory'])

    def test_model_budget_error_remains_original(self):
        environment = FixtureEnvironment(self.fixture, self.settings['episode_action_limit'])
        settings = {**self.settings, 'max_model_calls': self.fixture['short_terminal_after_actions']}
        with self.assertRaises(TaskLimitError):
            solve(self.task, environment, self.completion(), settings, self.prompts)


if __name__ == '__main__':
    unittest.main()
