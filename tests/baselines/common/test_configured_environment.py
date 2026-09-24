import ast
import json
from pathlib import Path
import unittest

from baselines.common.schema_environment import environment_definition
from baselines.common.environment import TaskEnvironment
from baselines.common.tasks import rows
from methods.evidence_flow.environment import EvidenceEnvironment


DIRECTORY = Path('configs/baselines/common/qualification/search_read_id_v1')


class ConfiguredEnvironmentTests(unittest.TestCase):
    def test_all_method_configs_use_identical_real_environment_protocol(self):
        specifications = [json.loads(path.read_text()) for path in sorted(DIRECTORY.glob('*.json'))]
        first = specifications[0]
        settings = json.loads(Path(first['environment']).read_text())
        corpus = EvidenceEnvironment(**settings['evidence'])
        task = next(task for task in rows(first['tasks']) if task['kind'] == 'answer')
        original = TaskEnvironment(task, settings, DIRECTORY, deadline=lambda: None, evidence=corpus)
        arguments = {'query': task['input']['query'], 'k': settings['evidence']['search_limit']}
        expected, done = original.execute('search', arguments)
        self.assertFalse(done)
        reference = None
        for specification in specifications:
            with self.subTest(method=specification['method']):
                factory, definition = environment_definition(specification)
                environment = factory(task, settings, DIRECTORY, deadline=lambda: None, evidence=corpus)
                if reference is None:
                    reference = definition
                self.assertEqual(definition, reference)
                self.assertEqual(specification['shared_config'], first['shared_config'])
                self.assertEqual(specification['tasks'], first['tasks'])
                self.assertNotIn('enum', environment.input_schemas['read']['properties']['ids']['items'])
                observed, done = environment.execute('search', arguments)
                self.assertFalse(done)
                self.assertEqual(observed, expected)
                ids = sorted(environment.observed_source_ids)
                self.assertEqual(environment.observe('read', {'ids': ids}),
                                 original.observe('read', {'ids': ids}))
                json.dumps({'environment_execution': definition})

    def test_explicit_selector_and_legacy_wrapper_source(self):
        specification = json.loads(Path('configs/baselines/common/qualification/tp4_v1/react.json').read_text())
        with self.assertRaises(KeyError):
            environment_definition(specification)
        tree = ast.parse(Path('src/baselines/common/run.py').read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        call = next(node for node in ast.walk(wrapper) if isinstance(node, ast.Call))
        self.assertEqual(call.func.id, 'run_with_environment')
        arguments = {item.arg: item.value for item in call.keywords}
        self.assertEqual(arguments['environment_factory'].id, 'TaskEnvironment')
        self.assertEqual(ast.literal_eval(arguments['environment_contract']), {})
        guard = next(node for node in wrapper.body if isinstance(node, ast.Assert))
        self.assertEqual(guard.test.left.value, 'environment_execution')
        self.assertIsInstance(guard.test.ops[0], ast.NotIn)

    def test_all_task_types_construct_and_nonretrieval_prompts_remain_exact(self):
        specification = json.loads((DIRECTORY / 'latentmas.json').read_text())
        settings = json.loads(Path(specification['environment']).read_text())
        corpus = EvidenceEnvironment(**settings['evidence'])
        factory, _ = environment_definition(specification)
        for task in rows(specification['tasks']):
            with self.subTest(task_id=task['task_id']):
                options = dict(deadline=lambda: None, evidence=corpus)
                original = TaskEnvironment(task, settings, DIRECTORY, **options)
                candidate = factory(task, settings, DIRECTORY, **options)
                self.assertEqual(candidate.tools, original.tools)
                if task['kind'] != 'answer':
                    self.assertEqual(candidate.reset(), original.reset())


if __name__ == '__main__':
    unittest.main()
