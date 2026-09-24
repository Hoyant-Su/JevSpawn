from copy import deepcopy
from functools import partial
import json
import time
import unittest

from baselines.common.evaluate import score_task
from jev_spawn.infra.configuration import resolve_symbol
from project_paths import ROOT


class LLFNativeTest(unittest.TestCase):
    def test_common_factory_native_observation_fork_and_scoring(self):
        fixture = json.loads((ROOT / 'tests/environments/llfbench_cases.json').read_text())
        reports = []
        for specification in fixture['specifications']:
            spec = json.loads((ROOT / specification).read_text())
            execution = spec['environment_execution']
            native_class = resolve_symbol(execution['class'])
            factory = partial(native_class, **execution['parameters'])
            settings = json.loads((ROOT / spec['environment']).read_text())
            tasks = [json.loads(line) for line in (ROOT / spec['tasks']).read_text().splitlines()]
            self.assertEqual(len(tasks), spec['task_count'])
            for task in tasks:
                directory = ROOT / fixture['output'] / task['task_id']
                directory.mkdir(parents=True, exist_ok=True)
                env = factory(task, settings, directory, deadline=lambda: None, evidence=None)
                self.assertEqual(env.reset(), task['context'])
                self.assertEqual(env.context(True, {}), env.context(False, {}))
                self.assertNotIn(fixture['expert_key'], env.reset())
                branch, direct = env.fork(), deepcopy(env.native)
                initial_room = env.native.unwrapped.current_room.get_name()
                initial_timestep = env.native.unwrapped.current_timestep
                control_info = deepcopy(env.info)
                selected_actions = []
                trace = []
                started = time.perf_counter()
                while not control_info[fixture['success_key']]:
                    action = control_info[fixture['expert_key']]
                    text, done = branch.execute(fixture['action_tool'], {fixture['action_field']: action})
                    observation, reward, terminated, truncated, control_info = direct.step(action)
                    self.assertEqual(json.loads(text), observation)
                    self.assertEqual(done, terminated or truncated)
                    self.assertEqual(branch.native.unwrapped.current_room.get_name(), direct.unwrapped.current_room.get_name())
                    selected_actions.append(action)
                    trace.append({'action': action, 'observation': observation, 'reward': reward, 'done': done})
                self.assertEqual(env.native.unwrapped.current_room.get_name(), initial_room)
                self.assertEqual(env.native.unwrapped.current_timestep, initial_timestep)
                correct_answer = {fixture['answer_field']: selected_actions}
                _, done = branch.execute(fixture['submission_tool'], correct_answer)
                self.assertTrue(done)
                self.assertEqual(branch.answer, correct_answer)
                artifact = directory / fixture['artifact_name']
                artifact.write_text(json.dumps({'task_id': task['task_id'], 'status': fixture['status'],
                    'answer': branch.answer, 'elapsed_seconds': time.perf_counter() - started,
                    'tool_timings': branch.tool_timings}))
                correct = score_task(task, artifact, native_class, execution['parameters'], settings, directory)
                self.assertTrue(correct['correct'])
                self.assertFalse(env.evaluate({fixture['answer_field']: fixture['empty_actions']}))
                env.execute(fixture['submission_tool'], {fixture['answer_field']: fixture['empty_actions']})
                self.assertFalse(env.evaluate(env.answer))
                incorrect_artifact = directory / fixture['incorrect_artifact_name']
                incorrect_artifact.write_text(json.dumps({'task_id': task['task_id'], 'status': fixture['status'],
                    'answer': env.answer, 'elapsed_seconds': time.perf_counter() - started,
                    'tool_timings': env.tool_timings}))
                incorrect = score_task(task, incorrect_artifact, native_class, execution['parameters'], settings, directory)
                self.assertFalse(incorrect['correct'])
                reports.append({'specification': specification, 'task_id': task['task_id'],
                    'model_inference': False, 'control_policy': 'Private official expert, evaluator verification only.',
                    'native_steps': len(selected_actions), 'context_matches': True, 'fork_isolation': True,
                    'correct_control_score': correct['score'], 'empty_answer_correct': incorrect['correct'],
                    'incorrect_control_score': incorrect['score'], 'trace': trace})
        (ROOT / fixture['report']).write_text(json.dumps({'model_inference': False, 'checks': reports}, indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
