from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

from baselines.common.config import SharedConfig
from baselines.common.react import solve
from jev_spawn.infra.prompts import load_prompt


class SharedAuthorityTests(unittest.TestCase):
    def setUp(self):
        self.shared = SharedConfig.load('configs/shared_config_tp4_v2.yaml')

    def test_all_active_method_configs_have_one_shared_authority(self):
        for path in Path('configs/baselines/common/methods').rglob('*.json'):
            method = json.loads(path.read_text())['settings']
            with self.subTest(path=str(path)):
                settings = self.shared.method_settings(method)
                for group in (self.shared.runtime, self.shared.generation, self.shared.model):
                    for key, value in asdict(group).items():
                        self.assertEqual(settings[key], value)
                self.assertNotIn('max_model_calls', method)
                self.assertNotIn('topology_seed', method)

    def test_shared_parameter_redefinition_is_rejected(self):
        for group in (self.shared.runtime, self.shared.generation, self.shared.model):
            for key, value in asdict(group).items():
                with self.subTest(parameter=key), self.assertRaisesRegex(ValueError, key):
                    self.shared.method_settings({key: value})
        with self.assertRaisesRegex(ValueError, 'context_length'):
            self.shared.method_settings({'context_length': self.shared.model.max_input_tokens})

    def test_config_changes_reach_method_without_secondary_defaults(self):
        changed = replace(self.shared, runtime=replace(self.shared.runtime,
            max_turns=self.shared.runtime.max_turns + self.shared.runtime.batch_size,
            seed=self.shared.runtime.seed + self.shared.runtime.batch_size))
        settings = changed.method_settings({})
        self.assertEqual(settings['max_turns'], changed.runtime.max_turns)
        self.assertEqual(settings['seed'], changed.runtime.seed)

    def test_real_react_core_reaches_shared_turn_limit(self):
        method = json.loads(Path('configs/baselines/common/methods/react.json').read_text())
        settings = self.shared.method_settings(method['settings'])
        prompts = load_prompt(method['prompts'])
        calls = []
        shared = self.shared

        class Environment:
            def __init__(self):
                self.answer = None
                self.actions = []

            def reset(self):
                return 'Synthetic budget-propagation test: execute tick until the environment terminates.'

            def deadline(self):
                return shared.runtime.sample_timeout_seconds

            def execute(self, name, arguments):
                self.actions.append((name, arguments))
                done = len(self.actions) == shared.runtime.max_turns
                if done:
                    self.answer = {'ticks': len(self.actions)}
                return 'Synthetic tick observed.', done

        def complete(messages, tokens, temperature, stop):
            calls.append((tokens, temperature))
            return [f'Observe the next tick.\nAction {len(calls)}: tick[{{}}]']

        environment = Environment()
        result = solve({'task_id': 'synthetic-shared-turn-propagation'}, environment,
                       complete, settings, prompts)
        self.assertEqual(len(result['actions']), self.shared.runtime.max_turns)
        self.assertEqual(len(calls), self.shared.runtime.max_turns)
        self.assertTrue(all(tokens == self.shared.generation.max_new_tokens and
                            temperature == self.shared.generation.temperature
                            for tokens, temperature in calls))


if __name__ == '__main__':
    unittest.main()
