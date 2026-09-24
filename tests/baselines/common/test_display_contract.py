from copy import deepcopy
import json
from pathlib import Path
import unittest

from baselines.common.display_contract import compile_contract, expand_contract_value
from baselines.common.environment import TaskEnvironment
from baselines.common.resources import snapshot
from baselines.common.schema_environment import environment_definition
from baselines.common.tasks import rows
from baselines.tool_agents.tools import ActionError
from methods.evidence_flow.environment import EvidenceEnvironment
from jev_spawn.infra.prompts import load_prompt


DIRECTORY = Path('configs/baselines/common/qualification/display_contract_v1')
FIXTURE = Path('tests/baselines/common/fixtures/display_contract_legacy.json')


class DisplayContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specifications = [json.loads(path.read_text()) for path in sorted(DIRECTORY.glob('*.json'))]
        cls.settings = json.loads(Path(cls.specifications[0]['environment']).read_text())
        cls.tasks = list(rows(cls.specifications[0]['tasks']))
        cls.corpus = EvidenceEnvironment(**cls.settings['evidence'])
        cls.fixtures = {row['task_id']: row for row in json.loads(FIXTURE.read_text())['contexts']}

    def environment(self, factory, task):
        return factory(task, self.settings, DIRECTORY, deadline=lambda: None, evidence=self.corpus)

    def test_legacy_rendering_is_byte_identical_for_every_real_task(self):
        for task in self.tasks:
            with self.subTest(task_id=task['task_id']):
                environment = self.environment(TaskEnvironment, task)
                saved = self.fixtures[task['task_id']]
                self.assertEqual(environment.reset(), saved['reset'])
                self.assertEqual(environment.context(False, {}), saved['without_finish'])
                self.assertEqual(environment.context(False, {'ensure_ascii': False}), saved['without_finish_unicode'])
                self.assertEqual(environment.display_answer_schema(), task['answer_schema'])
                for name in environment.tools:
                    self.assertEqual(environment.display_tool_schema(name), environment.input_schemas[name])

    def test_original_eight_formal_protocols_serialize_identically(self):
        fixture = json.loads(Path('tests/baselines/common/fixtures/display_contract_reference_schema.json').read_text())
        for filename in fixture['legacy_protocols']:
            with self.subTest(protocol=filename):
                saved = json.loads(Path(filename).read_text())
                specification = saved['specification']
                method = json.loads(Path(specification['method']).read_text())
                current = {'specification': specification, 'method': method, 'prompts': load_prompt(method['prompts']),
                    'tools': json.loads(Path(specification['environment']).read_text()),
                    'tasks': list(rows(specification['tasks'])),
                    'shared_config_text': Path(specification['shared_config']).read_text(), 'resources': snapshot()}
                self.assertEqual(json.dumps(saved), json.dumps(current))

    def test_original_schema_refs_and_literal_reference_keys_are_not_reinterpreted(self):
        fixture = json.loads(Path('tests/baselines/common/fixtures/display_contract_reference_schema.json').read_text())
        _, definition = environment_definition(self.specifications[0])
        policy = definition['parameters']['display_policy']
        contract = compile_contract(fixture['task'], fixture['tools'], policy)
        restored = expand_contract_value(contract, contract, policy, [])
        self.assertEqual(restored['input'], fixture['task']['input'])
        self.assertEqual(restored['answer_schema'], fixture['task']['answer_schema'])
        self.assertEqual(restored['tools'], fixture['tools'])
        self.assertEqual(contract['$defs']['answer']['properties']['flag']['enum'], fixture['task']['answer_schema']['properties']['flag']['enum'])

    def test_all_method_factories_preserve_complete_contracts_for_every_task(self):
        for specification in self.specifications:
            factory, definition = environment_definition(specification)
            for task in self.tasks:
                with self.subTest(method=specification['method'], task_id=task['task_id']):
                    before = deepcopy(task)
                    environment = self.environment(factory, task)
                    private = deepcopy(environment.input_schemas)
                    for include_finish in (False, True):
                        contract = environment.contract(include_finish)
                        restored = expand_contract_value(contract, contract, environment.display_policy, [])
                        self.assertEqual(restored['input'], task['input'])
                        self.assertEqual(restored['instruction'], task['instruction'])
                        self.assertEqual(restored['answer_schema'], task['answer_schema'])
                        self.assertEqual(restored['tools'], TaskEnvironment.display_tool_interface(environment, include_finish))
                        self.assertEqual(expand_contract_value(environment.display_tool_interface(include_finish),
                            contract, environment.display_policy, ['tools']), restored['tools'])
                        for name in restored['tools']:
                            self.assertEqual(environment.display_tool_schema(name), private[name])
                    self.assertEqual(environment.display_answer_schema(), task['answer_schema'])
                    self.assertEqual(environment.input_schemas, private)
                    self.assertEqual(task, before)
                    self.assertEqual(environment.reset(), environment.context(True, {}))
                    json.dumps(definition)

    def test_real_search_read_observations_and_private_validation_are_unchanged(self):
        factory, _ = environment_definition(self.specifications[0])
        task = next(task for task in self.tasks if task['kind'] == 'answer')
        original = self.environment(TaskEnvironment, task)
        candidate = self.environment(factory, task)
        arguments = {'query': task['input']['query'], 'k': self.settings['evidence']['search_limit']}
        expected = original.execute('search', arguments)
        observed = candidate.execute('search', arguments)
        self.assertEqual(observed, expected)
        ids = sorted(candidate.observed_source_ids)
        self.assertEqual(candidate.execute('read', {'ids': ids}), original.execute('read', {'ids': ids}))
        self.assertEqual(candidate.read_validation_schema['properties']['ids']['items']['enum'], ids)
        self.assertNotIn('enum', candidate.input_schemas['read']['properties']['ids']['items'])
        unseen = next(row['task_id'] for row in self.tasks if row['task_id'] not in ids)
        with self.assertRaises(ActionError):
            candidate.execute('read', {'ids': [unseen]})

    def test_real_catalog_ids_and_document_observations_are_unchanged(self):
        factory, _ = environment_definition(self.specifications[0])
        task = next(task for task in self.tasks if task['kind'] == 'ranking')
        original, candidate = self.environment(TaskEnvironment, task), self.environment(factory, task)
        schema = candidate.input_schemas['read_documents']['properties']['ids']
        ids = schema['items']['enum']
        self.assertEqual(len(ids), len(task['input']['catalog']))
        for offset in range(0, len(ids), schema['maxItems']):
            arguments = {'ids': ids[offset:offset + schema['maxItems']]}
            self.assertEqual(candidate.execute('read_documents', arguments), original.execute('read_documents', arguments))
        contract = candidate.contract(True)
        answer_ids = contract['$defs']['answer']['properties']['ranking']['items']['enum']['x-enum-source']
        self.assertEqual(answer_ids, {'path': ['catalog'], 'projection': ['id']})


if __name__ == '__main__':
    unittest.main()
