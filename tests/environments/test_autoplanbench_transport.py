import json
from pathlib import Path
import unittest

import yaml

from baselines.common.transition_budget import TransitionBudget
from environments.autoplanbench import AutoPlanBenchEnvironment
from project_paths import ROOT


class AutoPlanBenchTransportTest(unittest.TestCase):
    def test_rejected_syntax_preserves_native_worker_and_state(self):
        case = json.loads(Path(__file__).with_name('autoplanbench_transport_cases.json').read_text())
        settings = json.loads((ROOT / case['configuration']).read_text())
        shared = yaml.safe_load((ROOT / case['shared_config']).read_text())
        tasks = map(json.loads, (ROOT / case['tasks']).read_text().splitlines())
        task = next(task for task in tasks if task['task_id'] == case['task_id'])
        native = AutoPlanBenchEnvironment(task, {}, ROOT / case['directory'],
            deadline=lambda: None, configuration=case['configuration'])
        session = json.loads((ROOT / settings['session']).read_text())
        environment = TransitionBudget(native, shared['runtime']['max_turns'],
            session['submission_tool'], case['initial_depth'], [])
        initial = native.facts.copy()
        errors = []
        for action in case['rejected_actions']:
            observed = environment.observe(settings['tool_name'], {'action': action})
            serialized, done = environment.execute(settings['tool_name'], {'action': action})
            self.assertEqual(observed, json.loads(serialized))
            self.assertEqual(observed['error']['type'], 'ValidationError')
            self.assertFalse(done)
            self.assertEqual(native.facts, initial)
            self.assertIsNone(native.worker.poll())
            self.assertNotIn(str(ROOT), observed['error']['message'])
            errors.append(observed)
        responses = []
        plan = (ROOT / case['gold_plan']).read_text().splitlines()
        for action in (line for line in plan if line.startswith('(')):
            observation = environment.observe(settings['tool_name'], {'action': action})
            self.assertTrue(observation['valid'])
            responses.append(observation)
        self.assertTrue(native.done)
        self.assertTrue(native.evaluate(native.answer))
        record = {'task_id': task['task_id'], 'rejected_actions': case['rejected_actions'],
            'errors': errors, 'valid_actions': len(responses), 'native_goal_reached': native.done,
            'native_submission_correct': True,
            'action_pattern': settings['action_schema']['properties']['action']['pattern']}
        (ROOT / case['output']).write_text(json.dumps(record, indent=2) + '\n')
        native.close()


if __name__ == '__main__':
    unittest.main()
