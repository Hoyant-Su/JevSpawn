from collections import deque
import json
from pathlib import Path
import unittest

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration
from jev_spawn.schema.declaration_builder import DeclarationBuilder


ROOT = Path(__file__).resolve().parents[3]
SETTINGS = json.loads((ROOT / 'configs/baselines/common/declaration_builder.json').read_text())
PROMPTS = load_prompt(SETTINGS['prompts'])
EXECUTION = json.loads((ROOT / 'configs/baselines/common/methods/jevspawn.json').read_text())['settings']['rollout']['execution']
TASK = json.loads((ROOT / 'data/native_context/llfbench_gridworld/tasks.jsonl').read_text().splitlines()[0])
TOOLS = json.loads((ROOT / 'configs/evaluation/native_context/llfbench_gridworld.json').read_text())['public_contract']['tools']


class ScriptedService:
    def __init__(self, choices=(), signatures=()):
        self.choices, self.signatures = deque(choices), deque(signatures)
        self.messages, self.caps, self.stops = [], [], []

    def decide(self, requests, *, task_id):
        result = []
        for request in requests:
            options = [json.loads(option['description']) for option in request['options']]
            selected = self.choices.popleft()
            result.append({'id': request['id'], 'choice': str(options.index(selected))})
        return result

    def complete_batch(self, messages, max_tokens, temperature, stop, *, task_id, return_tokens):
        self.messages.extend(messages)
        self.caps.append(max_tokens)
        self.stops.append(stop)
        return [{'text': self.signatures.popleft(), 'token_ids': [42], 'finish_reason': 'stop'}
                for _ in messages]


def builder(service=None, **updates):
    return DeclarationBuilder(TASK['context'], TOOLS, TASK['answer_schema'], service, TASK['task_id'],
        {'max_new_tokens': 2048, 'temperature': 0.0}, {**SETTINGS, **updates}, PROMPTS, EXECUTION, {}, feedback={})


def materialize(instance, template, values):
    execution = QueryExecution('context', None, 'task', EXECUTION)
    execution.values.update(values)
    return execution.materialize(template)


class NativeSignatureTests(unittest.TestCase):
    def test_repeat_allocates_independent_fields_without_enumerating_products(self):
        instance = builder()
        result = instance.compile_signature('emit {code:repeat(3,range(0,2))}', ['action'])
        self.assertEqual(result, 'emit ${f0}${f1}${f2}')
        self.assertEqual(len(instance.program['fields']), 3)
        self.assertEqual([field['values'] for field in instance.program['fields']], [[0, 1]] * 3)
        self.assertEqual(materialize(instance, result, {'f0': 1, 'f1': 0, 'f2': 1}), 'emit 101')

    def test_role_is_bound_once_and_reuse_inserts_same_binding(self):
        instance = builder()
        signature = "from {origin:enum('alpha','beta')} to {target:enum('alpha','beta')} via {origin}"
        result = instance.compile_signature(signature, ['action'])
        self.assertEqual(result, 'from ${f0} to ${f1} via ${f0}')
        self.assertEqual(len(instance.program['fields']), 2)
        self.assertEqual(materialize(instance, result, {'f0': 'alpha', 'f1': 'beta'}), 'from alpha to beta via alpha')

    def test_native_variants_preserve_arity_and_share_identical_named_domains(self):
        instance = builder()
        signature = "(take {item:enum('alpha','beta')})\n(put {item} {target:enum('left','right')})"
        template = instance.compile_signature(signature, ['action'])
        self.assertEqual(template, '${f2}')
        selector = instance.program['fields'][2]
        self.assertEqual(selector['values'], ['(take ${f0})', '(put ${f0} ${f1})'])
        declaration = {'fields': instance.program['fields'], 'action': {'tool': 'execute', 'arguments': {'action': template}}, 'answer': {}}
        compiled = compile_declaration(declaration, EXECUTION)
        self.assertEqual(compiled['fields'][0]['id'], 'f2')
        self.assertEqual(materialize(instance, template, {'f0': 'alpha', 'f1': 'right', 'f2': selector['values'][0]}), '(take alpha)')
        self.assertEqual(materialize(instance, template, {'f0': 'alpha', 'f1': 'right', 'f2': selector['values'][1]}), '(put alpha right)')

    def test_literal_whitespace_dollars_and_braces_are_preserved(self):
        instance = builder()
        result = instance.compile_signature('  {{$}} {value:enum("$one","$two")}  ', ['action'])
        self.assertEqual(materialize(instance, result, {'f0': '$$one'}), '  {$} $one  ')
        self.assertEqual(instance.compile_signature('', ['constant']), '')
        self.assertEqual(instance.compile_signature(' ', ['constant']), ' ')

    def test_redeclaration_preserves_prior_templates_and_reuses_interned_domains(self):
        instance = builder()
        boolean = instance.compile_signature('{flag:enum(False,True)}', ['boolean'])
        integer = instance.compile_signature('{flag:enum(0,1)}', ['integer'])
        self.assertEqual((boolean, integer), ('${f0}', '${f1}'))
        self.assertEqual(instance.compile_signature('{flag}', ['latest']), integer)
        values = {'f0': False, 'f1': 0}
        self.assertIs(materialize(instance, boolean, values), False)
        self.assertIs(type(materialize(instance, integer, values)), int)
        restored = instance.compile_signature('{flag:enum(False,True)}', ['restored'])
        self.assertEqual(restored, boolean)
        self.assertEqual(instance.compile_signature('{flag}', ['latest']), boolean)
        self.assertEqual(len(instance.program['fields']), 2)

    def test_redeclaration_takes_effect_at_its_exact_position(self):
        instance = builder()
        template = instance.compile_signature(
            '{x:range(0,2)} {x} {x:range(2,4)} {x} {x:range(0,2)} {x}', ['action'])
        self.assertEqual(template, '${f0} ${f0} ${f1} ${f1} ${f0} ${f0}')
        self.assertEqual(materialize(instance, template, {'f0': 1, 'f1': 3}), '1 1 3 3 1 1')

    def test_redeclaration_interning_preserves_order_role_identity_and_limits(self):
        instance = builder(max_fields=3)
        self.assertEqual(instance.bind('x', 'range(0,2)'), '${f0}')
        self.assertEqual(instance.bind('x', 'enum(0,1)'), '${f0}')
        self.assertEqual(instance.bind('x', 'enum(1,0)'), '${f1}')
        self.assertEqual(instance.bind('y', 'enum(0,1)'), '${f2}')
        self.assertEqual(instance.bind('x', 'range(0,2)'), '${f0}')
        with self.assertRaisesRegex(AssertionError, 'field capacity'):
            instance.bind('x', 'range(2,4)')
        with self.assertRaisesRegex(AssertionError, 'Undefined signature role'):
            instance.bind('undefined', '')

    def test_repeated_domain_rebinding_reuses_all_position_fields(self):
        instance = builder()
        first = instance.bind('code', 'repeat(2,range(0,2))')
        second = instance.bind('code', 'repeat(2,range(2,4))')
        self.assertEqual(first, '${f0}${f1}')
        self.assertEqual(second, '${f2}${f3}')
        self.assertEqual(instance.bind('code', 'repeat(2,range(0,2))'), first)
        self.assertEqual(instance.bind('code', ''), first)
        self.assertEqual(len(instance.program['fields']), 4)

    def test_variant_identity_uses_the_domain_bound_at_each_occurrence(self):
        instance = builder()
        template = instance.compile_signature('emit {x:range(0,2)}\nemit {x:range(2,4)}', ['action'])
        self.assertEqual(template, '${f2}')
        self.assertEqual(instance.program['fields'][2]['values'], ['emit ${f0}', 'emit ${f1}'])
        self.assertEqual(instance.bind('x', ''), '${f1}')
        for signature in ['emit {x:range(0,2)}\nemit {x}',
                          'emit {x:range(0,2)}\nemit {x:range(2,4)}\nemit {x:range(0,2)}',
                          '```\nemit {x:range(0,2)}\n```']:
            with self.subTest(signature=signature), self.assertRaisesRegex(AssertionError, 'Duplicate native signature'):
                builder().compile_signature(signature, ['action'])

    def test_unknown_roles_domain_operators_and_malformed_signatures_fail(self):
        for value in ('{unknown}', '{slot:unknown(1)}', '{slot:range(0,2)', '{slot!r:range(0,2)}'):
            with self.subTest(value=value), self.assertRaises((AssertionError, ValueError)):
                builder().compile_signature(value, ['action'])

    def test_domain_ast_rejects_nonliteral_arguments_and_duplicates(self):
        for expression in ("enum(__import__('os'))", 'range(0,2,1)', 'enum(1,1)', 'range(4,1)'):
            with self.subTest(expression=expression), self.assertRaises((AssertionError, ValueError)):
                builder().domain(expression)

    def test_repeat_fields_and_variant_capacities_are_explicit(self):
        with self.assertRaisesRegex(AssertionError, 'field capacity'):
            builder(max_fields=2).compile_signature('{x:repeat(3,range(0,2))}', ['action'])
        with self.assertRaisesRegex(AssertionError, 'variant capacity'):
            builder(max_variants=1).compile_signature('first\nsecond', ['action'])

    def test_real_public_integer_contract_needs_no_signature_generation(self):
        service = ScriptedService(['execute'])
        instance = builder(service)
        declaration = instance.build_action()
        self.assertEqual(declaration['fields'][0]['values'], [0, 1, 2, 3])
        self.assertEqual(declaration['action'], {'tool': 'execute', 'arguments': {'action': '${f0}'}})
        self.assertIsNone(declaration['answer'])
        self.assertEqual(instance.tokens, 0)
        self.assertEqual(service.messages, [])

    def test_signature_generation_shares_budget_and_preserves_entire_context(self):
        service = ScriptedService(signatures=['emit {bit:range(0,2)}'])
        instance = builder(service)
        instance.budget['max_new_tokens'] = 1
        self.assertEqual(instance.signature({'type': 'string'}, ['action']), 'emit ${f0}')
        self.assertEqual(service.caps, [1])
        self.assertIn(TASK['context'], service.messages[0][1]['content'])
        self.assertEqual(service.messages[0][-1]['role'], 'user')
        self.assertEqual(service.stops, [None])
        with self.assertRaisesRegex(AssertionError, 'generation budget'):
            instance.signature({'type': 'string'}, ['action'])

    def test_signature_uses_actual_remaining_shared_budget(self):
        service = ScriptedService(signatures=['emit {bit:range(0,2)}'])
        instance = builder(service)
        instance.tokens = 200
        instance.signature({'type': 'string'}, ['action'])
        self.assertEqual(service.caps, [1848])
        self.assertEqual(instance.calls[-2]['max_new_tokens'], 1848)

    def test_length_stopped_signature_is_recorded_but_not_compiled(self):
        service = ScriptedService()
        output = {'text': 'emit {bit:range(', 'token_ids': [42] * 2048, 'finish_reason': 'length'}
        service.complete_batch = lambda *args, **kwargs: [output]
        instance = builder(service)
        with self.assertRaisesRegex(ValueError, 'remaining 2048-token budget'):
            instance.signature({'type': 'string'}, ['action'])
        self.assertEqual(instance.program['fields'], [])
        self.assertEqual(instance.tokens, 2048)
        self.assertEqual(instance.calls[-1]['outputs'], [output])

    def test_completed_malformed_signature_keeps_parser_error(self):
        service = ScriptedService(signatures=['emit {bit:range('])
        with self.assertRaisesRegex(ValueError, "unmatched|expected"):
            builder(service).signature({'type': 'string'}, ['action'])

    def test_generation_timeout_is_not_compiler_feedback(self):
        service = ScriptedService()
        service.complete_batch = lambda *args, **kwargs: [
            {'text': 'emit {bit:', 'token_ids': [42], 'finish_reason': 'timeout'}]
        instance = builder(service)
        with self.assertRaisesRegex(TimeoutError, 'task deadline'):
            instance.signature({'type': 'string'}, ['action'])
        self.assertEqual(instance.program['fields'], [])

    def test_variant_selector_consumes_the_declared_field_budget(self):
        with self.assertRaisesRegex(AssertionError, 'field capacity'):
            builder(max_fields=2).compile_signature('{x:range(0,2)} {y:range(0,2)}\n{x}', ['action'])

    def test_word_delimiters_remain_literal_and_repeat_does_not_insert_them(self):
        instance = builder()
        result = instance.compile_signature('{a:enum("up","down")} {b:enum("up","down")}', ['action'])
        self.assertEqual(materialize(instance, result, {'f0': 'up', 'f1': 'down'}), 'up down')

    def test_current_feedback_reaches_signature_without_replacing_original_context(self):
        service = ScriptedService(signatures=['emit {bit:range(0,2)}'])
        instance = builder(service)
        instance.feedback = {'operation': 'revise', 'active_declaration': {'action': 'emit ${f0}'},
            'current_action_observations': [{'action': 'emit 1', 'value': 'Previous action was rejected.'}],
            'execution': {'current_feedback': ['actual-event']}}
        instance.signature({'type': 'string'}, ['action'])
        user = service.messages[0][1]['content']
        self.assertIn(TASK['context'], user)
        self.assertIn('Previous action was rejected.', user)
        control = {key: value for key, value in instance.feedback.items() if key != 'execution'}
        self.assertIn(json.dumps(control, **EXECUTION['serialization']), user)
        self.assertEqual(json.loads(instance.state({}))['feedback'], instance.feedback)

    def test_selected_tool_is_visible_before_generating_its_argument(self):
        service = ScriptedService(['deliver'], ['message {tag:enum("alpha","beta")}'])
        instance = builder(service)
        instance.tools = {'deliver': {'input_schema': {'type': 'object',
            'properties': {'payload': {'type': 'string'}}, 'required': ['payload']}}}
        declaration = instance.build_action()
        user = service.messages[0][1]['content']
        self.assertIn('"partial_declaration":{"fields":[],"action":{"tool":"deliver","arguments":{}}', user)
        self.assertIn('"destination":["action","arguments","payload"]', user)
        self.assertEqual(declaration['action'], {'tool': 'deliver', 'arguments': {'payload': 'message ${f0}'}})



if __name__ == '__main__':
    unittest.main()
