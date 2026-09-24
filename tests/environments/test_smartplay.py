from copy import deepcopy
import json
import unittest

from environments.smartplay import SmartPlayEnvironment
from project_paths import ROOT


class SmartPlayNativeTest(unittest.TestCase):
    def test_native_transitions_fork_and_final_evaluation(self):
        fixture = json.loads((ROOT / 'tests/environments/smartplay_cases.json').read_text())
        tasks = [json.loads(line) for line in (ROOT / fixture['tasks']).read_text().splitlines()]
        for task, plan in zip(tasks, fixture['solutions'], strict=True):
            env = SmartPlayEnvironment(task, {}, ROOT / fixture['output'],
                                      deadline=lambda: None, configuration=fixture['configuration'])
            self.assertEqual(env.reset(), task['context'])
            self.assertIn(env.native.desc, task['context'])
            self.assertEqual(env.context(True, {}), env.context(False, {}))
            initial = env.native.current_state
            direct, branch = deepcopy(env.native), env.fork()
            for action in plan:
                result = branch.observe('execute', {'action': action})
                _, _, done, info = direct.step(action - env.configuration['action_offset'])
                self.assertEqual(result['observation'], info['obs'])
                self.assertEqual(result['done'], done)
                self.assertEqual(branch.native.current_state, direct.current_state)
            self.assertTrue(branch.done)
            self.assertTrue(branch.evaluate(branch.answer))
            self.assertFalse(env.evaluate({'actions': fixture['empty_actions']}))
            self.assertEqual(env.native.current_state, initial)
            self.assertFalse(env.done)
            env.observe('finish', {'actions': fixture['empty_actions']})
            self.assertFalse(env.evaluate(env.answer))


if __name__ == '__main__':
    unittest.main()
