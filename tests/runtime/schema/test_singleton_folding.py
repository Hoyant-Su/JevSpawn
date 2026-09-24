from itertools import count
import json
import unittest
from unittest.mock import Mock

import jsonschema
import torch

from environments.maze import MazeEnvironment
from jev_spawn.runtime.branches import Branch, spawn_scored
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration, template_references
from declaration_builder_stage052 import DeclarationBuilder as PreviousBuilder
from test_declaration_builder import builder, EXECUTION, ROOT


def previous_builder(**settings):
    instance = builder(**settings)
    instance.__class__ = PreviousBuilder
    return instance


def action_family(instance, template):
    fields = {field['id']: field for field in instance.program['fields']}
    execution = QueryExecution('Exact compiler replay.', None, 'replay', EXECUTION)
    outputs = set()

    def enumerate_values(values):
        references = template_references([template, *values.values()], EXECUTION) - values.keys()
        if references:
            identity = next(identity for identity in fields if identity in references)
            for value in fields[identity]['values']:
                enumerate_values({**values, identity: value})
        else:
            execution.values = values
            outputs.add(json.dumps(execution.materialize(template), ensure_ascii=False, sort_keys=True))

    enumerate_values({})
    return outputs


class SingletonFoldingTests(unittest.TestCase):
    def assert_family(self, signature, fields):
        old, new = previous_builder(max_fields=64), builder()
        before = old.compile_signature(signature, ['action'])
        after = new.compile_signature(signature, ['action'])
        self.assertEqual(action_family(old, before), action_family(new, after))
        self.assertEqual(len(new.program['fields']), fields)

    def test_singleton_types_in_whole_and_embedded_references(self):
        for expression in ['enum(1)', 'enum(1.5)', 'enum(True)', 'enum(False)', 'enum(None)',
                           'enum("")', 'enum("$x ${f0} $$")', 'range(2,3)']:
            for signature in ['{x:' + expression + '}', 'value={x:' + expression + '}!',
                              '{x:' + expression + '}{x}', '{empty:enum("")}{x:' + expression + '}']:
                with self.subTest(signature=signature):
                    self.assert_family(signature, 0)

    def test_repeat_preserves_scalar_or_interpolated_type_and_dollars(self):
        for expression in ['enum(3)', 'enum(None)', 'enum(False)', 'enum("$")']:
            for size in [1, 2, 9]:
                self.assert_family('{x:repeat(' + str(size) + ',' + expression + ')}', 0)
        self.assert_family('x{x:repeat(2,enum("$"))}={v:range(1,3)}', 1)

    def test_only_variable_fields_consume_capacity(self):
        instance = builder(max_fields=1)
        signature = '{a:enum(None)} {b:enum(4)} {c:enum("$")} {d:range(0,2)}'
        template = instance.compile_signature(signature, ['action'])
        self.assertEqual(len(instance.program['fields']), 1)
        self.assertEqual(action_family(instance, template), {'"None 4 $ 0"', '"None 4 $ 1"'})
        with self.assertRaisesRegex(AssertionError, 'field capacity'):
            instance.compile_signature('{e:range(0,2)}', ['second'])

    def test_empty_constant_keeps_interpolated_variable_string_typed(self):
        for signature in ['{empty:enum("")}{x:range(0,2)}',
                          '{x:enum(None,True)}{empty:enum("")}',
                          '{a:enum("")}{b:enum("")}{x:range(0,2)}',
                          '{empty:enum("")}{x:range(0,2)}\n{x}',
                          '{empty:enum("")}{x:range(0,2)}\nwrapped:{empty}{x}']:
            with self.subTest(signature=signature):
                self.assert_family(signature, 2 if '\n' in signature else 1)

    def test_constant_rebinding_preserves_prior_values_and_exact_types(self):
        for first, second in [('enum(False)', 'enum(0)'), ('enum(1)', 'enum(1.0)'),
                              ('enum("a")', 'enum("b")')]:
            instance = builder()
            before = instance.compile_signature('{x:' + first + '}', ['action'])
            after = instance.compile_signature('{x:' + second + '}', ['second'])
            self.assertNotEqual(json.dumps(before), json.dumps(after))
            self.assertEqual(json.dumps(instance.compile_signature('{x}', ['latest'])), json.dumps(after))
            restored = instance.compile_signature('{x:' + first + '}', ['restored'])
            self.assertEqual(json.dumps(restored), json.dumps(before))
            self.assertEqual(instance.program['fields'], [])

    def test_folded_duplicate_alternatives_preserve_set_and_source_validation(self):
        self.assert_family('{a:enum("same")}\n{b:enum("same")}', 0)
        self.assert_family('{a:enum(1)}\n1', 1)
        self.assert_family('{a:enum("$")}\n$', 0)
        for signature in ['same\nsame', '{a:enum("same")}\n{a}']:
            with self.assertRaisesRegex(AssertionError, 'Duplicate native signature'):
                builder().compile_signature(signature, ['action'])

    def test_public_constants_preserve_json_types(self):
        for value in [1, 1.5, True, False, None, '$literal']:
            instance = builder()
            template = instance.allocate('value', instance.public_domain({'const': value}))
            execution = QueryExecution('Typed public constant.', None, 'test', EXECUTION)
            actual = execution.materialize(template)
            self.assertEqual(type(actual), type(value))
            self.assertEqual(actual, value)
            self.assertEqual(instance.program['fields'], [])

    def test_zero_fields_execute_each_declared_constant_in_real_native_environment(self):
        contract = json.loads((ROOT / 'configs/jevspawn/interactive.json').read_text())['environment']
        factory = json.loads((ROOT / contract['factory']).read_text())
        task = json.loads((ROOT / contract['tasks']).read_text().splitlines()[0])
        native = MazeEnvironment(task, {}, contract['directory'], deadline=Mock(), **factory['parameters'])
        method = json.loads((ROOT / 'configs/baselines/common/methods/jevspawn.json').read_text())['settings']['rollout']
        for action in native.native_actions:
            instance = builder()
            template = instance.compile_signature('{direction:enum(' + repr(action) + ')}', ['action'])
            declaration = {'fields': instance.program['fields'], 'action': {'tool': contract['tool'],
                'arguments': {contract['argument']: template}}, 'answer': None}
            jsonschema.validate(declaration, method['declaration_schema'])
            declaration = compile_declaration(declaration, EXECUTION)
            service = Mock()
            service.initial_scores.return_value = torch.zeros(1)
            execution = QueryExecution('Recorded native constant action.', service, task['task_id'], EXECUTION)
            children = spawn_scored({'root': Branch(execution, native, declaration)}, 4, 1,
                                    (f'n{i}' for i in count()))
            child, = children.values()
            event, = child.execution.latest_feedback
            expected = native.fork().observe(contract['tool'], {contract['argument']: action})
            self.assertEqual(event['value'], expected)
            self.assertEqual(event['action']['arguments'], {contract['argument']: action})
            self.assertEqual(child.execution.trace[-1]['selected_values'], {})
            service.decide.assert_not_called()
            service.extend.assert_not_called()
        self.assertEqual(native.actions, [])


if __name__ == '__main__':
    unittest.main()
