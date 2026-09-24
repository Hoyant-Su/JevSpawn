import json
from pathlib import Path
import tempfile
import unittest

from environments.autoplanbench import AutoPlanBenchEnvironment
from project_paths import ROOT


class AutoPlanBenchNativeTest(unittest.TestCase):
    def test_public_domain_matches_native_executable_casing(self):
        configuration = 'configs/environments/autoplanbench.json'
        tasks = [json.loads(line) for line in
                 (ROOT / 'data/qualification/native_screen/autoplanbench/tasks.jsonl').read_text().splitlines()]
        task = next(task for task in tasks if task['task_id'] == 'autoplanbench_depot_instance_1')
        source = ROOT / task['source']['domain_file']
        original_domain = source.read_text()
        run_root = ROOT / 'results/tests/autoplanbench'
        run_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(dir=run_root))
        environment = AutoPlanBenchEnvironment(task, {}, directory,
            deadline=lambda: None, configuration=configuration)
        canonical_domain = (directory / 'domain_tmp.pddl').read_text()
        self.assertNotEqual(original_domain, canonical_domain)
        self.assertEqual(canonical_domain, original_domain.lower())
        self.assertIn(canonical_domain, environment.context(True, {}))
        self.assertEqual(environment.context(True, {}), environment.context(False, {}))
        plan = (source.parent / 'gold_plans/instance-1_gold_plan.txt').read_text().splitlines()
        action = next(line for line in plan if line.startswith('('))
        name = action[1:].split()[0]
        self.assertIn(f'(:action {name}', environment.context(False, {}))
        valid = environment.fork().observe('execute', {'action': action})
        self.assertTrue(valid['valid'])
        original_name = next(line.split()[1] for line in original_domain.splitlines()
                             if line.startswith('(:action ') and line.split()[1].lower() == name)
        unchanged_action = action.replace(name, original_name, 1)
        rejected = environment.fork().observe('execute', {'action': unchanged_action})
        self.assertFalse(rejected['valid'])
        self.assertEqual(rejected['observation'], f'{original_name} does not match any possible actions. ')
        self.assertEqual(source.read_text(), original_domain)
        environment.close()

    def test_official_feedback_and_isolated_fork(self):
        configuration = 'configs/environments/autoplanbench.json'
        settings = json.loads((ROOT / configuration).read_text())
        source = ROOT / 'data/context_datasets/autoplanbench/extracted/apb1.0_dataset/data_main/ferry'
        task = {'source': {'domain_file': str(source / 'domain.pddl'),
                           'instance_file': str(source / 'adapted_instances/instance-1.pddl')},
                'answer_schema': settings['answer_schema']}
        run_root = ROOT / 'results/tests/autoplanbench'
        run_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(dir=run_root))
        environment = AutoPlanBenchEnvironment(task, {}, directory,
            deadline=lambda: None, configuration=configuration)
        context = environment.context(False, settings['serialization'])
        self.assertIn('Here is the definition of the domain in PDDL', context)
        self.assertIn('(:objects', context)
        self.assertIn('(:goal', context)
        self.assertNotIn('gold_plan', context)
        original = environment.facts.copy()
        child = environment.fork()
        plan = (source / 'gold_plans/instance-1_gold_plan.txt').read_text().splitlines()
        for action in (line for line in plan if line.startswith('(')):
            response = child.observe('execute', {'action': action})
            self.assertTrue(response['valid'])
        self.assertTrue(child.done)
        self.assertEqual(environment.facts, original)
        self.assertFalse(environment.done)
        invalid = environment.observe('execute', {'action': '(board object_2 object_0)'})
        self.assertFalse(invalid['valid'])
        self.assertEqual(invalid['observation'], 'The action is not applicable in the current state.')
        environment.close()


    def test_non_few_shot_records_and_submission_score(self):
        configuration = 'configs/environments/autoplanbench.json'
        tasks = [json.loads(line) for line in
                 (ROOT / 'data/native_context/autoplanbench/tasks.jsonl').read_text().splitlines()]
        labels = {row['task_id']: row['actions'] for row in
                  map(json.loads, (ROOT / 'data/native_context/autoplanbench/labels.jsonl').read_text().splitlines())}
        for task in tasks:
            environment = AutoPlanBenchEnvironment(task, {},
                ROOT / 'results/tests/autoplanbench' / task['task_id'],
                deadline=lambda: None, configuration=configuration)
            self.assertEqual(environment.context(True, {}), environment.context(False, {}))
            self.assertIn('finish', environment.reset())
            self.assertTrue(environment.evaluate({'actions': labels[task['task_id']]}))
            self.assertFalse(environment.evaluate({'actions': []}))
            environment.observe('finish', {'actions': []})
            self.assertTrue(environment.done)
            self.assertFalse(environment.evaluate(environment.answer))
            environment.close()


if __name__ == '__main__':
    unittest.main()
