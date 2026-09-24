from copy import deepcopy
from graphlib import CycleError
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import jsonschema

from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration, declaration_contract


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/baselines/common/methods/jevspawn.json').read_text())['settings']['rollout']
RECORDED = json.loads((ROOT / 'tests/fixtures/schema/plancraft_declaration_012.json').read_text())


class DeclarationCompilerTests(unittest.TestCase):
    def compile(self, declaration):
        contract = declaration_contract(SETTINGS['declaration_schema'], 256)
        jsonschema.validate(declaration, contract)
        return compile_declaration(declaration, SETTINGS['execution'])

    def test_recorded_selector_precedes_parameters_and_unused_operation_is_removed(self):
        original = deepcopy(RECORDED['declaration'])
        compiled = self.compile(original)
        self.assertEqual([field['id'] for field in compiled['fields']],
                         ['op_template', 'src', 'dst', 'qty', 'reason'])
        fields = {field['id']: field for field in original['fields']}
        self.assertEqual(compiled['fields'], [fields[name] for name in
                         ['op_template', 'src', 'dst', 'qty', 'reason']])
        self.assertEqual(original, RECORDED['declaration'])
        self.assertEqual(compiled['action'], original['action'])
        self.assertEqual(compiled['answer'], original['answer'])
        execution = QueryExecution('Recorded template compilation.', Mock(), 'recorded', SETTINGS['execution'])
        execution.values.update(op_template=fields['op_template']['values'][1],
                                src='I1', dst='I2', qty='2', reason='missing material')
        self.assertEqual(execution.materialize(compiled['action']),
                         {'tool': 'execute', 'arguments': {'action': 'smelt: from [I1] to [I2] with quantity 2'}})

    def test_nested_templates_preserve_native_types_and_answer_only_fields(self):
        declaration = {
            'fields': [
                {'id': 'quantity', 'question': 'Amount', 'values': [2, 3]},
                {'id': 'enabled', 'question': 'Flag', 'values': [True, False]},
                {'id': 'unused', 'question': 'Unreferenced', 'values': ['a', 'b']},
                {'id': 'payload', 'question': 'Payload', 'values': [
                    {'items': ['${quantity}', '${enabled}'], 'label': '$$literal'},
                    {'items': ['${enabled}', '${quantity}'], 'label': '$$other'}]},
                {'id': 'report', 'question': 'Report', 'values': ['complete', 'pending']}],
            'action': {'tool': 'execute', 'arguments': {'payload': '${payload}'}},
            'answer': {'status': '${report}'}}
        compiled = self.compile(declaration)
        self.assertEqual([field['id'] for field in compiled['fields']],
                         ['payload', 'quantity', 'enabled', 'report'])
        execution = QueryExecution('Nested typed templates.', Mock(), 'nested', SETTINGS['execution'])
        execution.values.update(payload=declaration['fields'][3]['values'][0],
                                quantity=2, enabled=False, report='complete')
        self.assertEqual(execution.materialize(compiled['action']),
                         {'tool': 'execute', 'arguments': {'payload': {'items': [2, False], 'label': '$literal'}}})
        self.assertEqual(execution.materialize(compiled['answer']), {'status': 'complete'})
        execution.service.decide.assert_not_called()

    def test_history_paths_and_escaped_dollars_are_literal(self):
        declaration = deepcopy(RECORDED['declaration'])
        declaration['answer'] = {'history': {'$history': ['arguments', '$literal']},
                                 'literal': '$$unbound'}
        self.compile(declaration)

    def test_undefined_references_fail_in_roots_and_nested_candidate_values(self):
        for location in ['action', 'candidate']:
            declaration = deepcopy(RECORDED['declaration'])
            if location == 'action':
                declaration['action']['arguments']['action'] = '${missing}'
            else:
                declaration['fields'][0]['values'][0] = {'nested': ['${missing}']}
            with self.assertRaisesRegex(ValueError, 'Undefined declaration fields: missing'):
                self.compile(declaration)

    def test_self_and_mutual_reference_cycles_fail_before_execution(self):
        for target in ['op', 'src']:
            declaration = deepcopy(RECORDED['declaration'])
            declaration['fields'][0]['values'][0] = '${' + target + '}'
            declaration['fields'][1]['values'][0] = '${op}'
            with self.assertRaises(CycleError):
                self.compile(declaration)

    def test_invalid_template_syntax_fails(self):
        declaration = deepcopy(RECORDED['declaration'])
        declaration['action']['arguments']['action'] = '${unfinished'
        with self.assertRaisesRegex(ValueError, 'Invalid template expression'):
            self.compile(declaration)

    def test_compilation_is_idempotent(self):
        compiled = self.compile(RECORDED['declaration'])
        self.assertEqual(compile_declaration(compiled, SETTINGS['execution']), compiled)


if __name__ == '__main__':
    unittest.main()
