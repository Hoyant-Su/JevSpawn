import json
from pathlib import Path
import unittest
from unittest.mock import patch

from baselines.common.lats import solve
from environments.ppnl import PPNLEnvironment
from jev_spawn.infra.prompts import load_prompt


class NativeBranchStateTest(unittest.TestCase):
    def test_siblings_execute_from_their_parent_state(self):
        task = json.loads(Path('data/native_context/ppnl/ICL_test_set.jsonl').read_text().splitlines()[0])
        specification = json.loads(Path('configs/experiments/native_context/ppnl_jevspawn.json').read_text())
        environment = PPNLEnvironment(task, {}, Path('runs/cpu-lats-branch-test'),
            deadline=lambda: 60, **specification['environment_execution']['parameters'])
        settings = json.loads(Path('configs/baselines/common/methods/lats.json').read_text())['settings']
        settings.update(max_turns=2, max_new_tokens=128, temperature=0, expansion_samples=2,
                        rollout_samples=1)
        starts = []
        execute = PPNLEnvironment.execute

        def recorded_execute(branch, name, arguments):
            starts.append((branch.position, arguments))
            return execute(branch, name, arguments)

        def complete(messages, tokens, temperature, n=1, stop=None):
            text = messages[-1]['content']
            if messages[0]['role'] == 'system':
                value = 1 if '"right"' in text else 10
                return [f'Thus the correctness score is {value}']
            if 'Observation 1:' not in text:
                return ['Move left.\nAction 1: execute[{"actions":"left"}]',
                        'Move right.\nAction 1: execute[{"actions":"right"}]']
            return ['Reach the goal.\nAction 2: execute[{"actions":"down down"}]']

        initial = environment.position
        with patch.object(PPNLEnvironment, 'execute', recorded_execute):
            result = solve({'task_id': task['task_id']}, environment, complete, settings,
                           load_prompt('configs/baselines/common/schema/lats.json'))
        self.assertEqual([row[0] for row in starts[:2]], [initial, initial])
        self.assertEqual(starts[2][0], (initial[0], initial[1] - 1))
        self.assertEqual(environment.position, initial)
        self.assertEqual(result['answer'], {'actions': 'left down down'})
        self.assertTrue(environment.evaluate(result['answer']))
        self.assertEqual(len(environment.tool_timings), len(starts))


if __name__ == '__main__':
    unittest.main()
