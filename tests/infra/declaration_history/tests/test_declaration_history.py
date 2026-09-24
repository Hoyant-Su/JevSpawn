from collections import deque
import json
from pathlib import Path
import unittest

from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.schema.declaration import compile_declaration
from declaration_history.src.declaration_builder import DeclarationBuilder


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((ROOT / 'config/builder.json').read_text())
PROMPTS = json.loads((ROOT / 'config/prompts.json').read_text())
FIXTURE = json.loads((ROOT / 'config/fixture.json').read_text())
EXECUTION, TASK, TOOLS = FIXTURE['execution'], FIXTURE['task'], FIXTURE['tools']


class ScriptedService:
    def __init__(self, choices=(), signatures=()):
        self.choices, self.signatures = deque(choices), deque(signatures)
        self.messages, self.caps = [], []

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
        return [{'text': self.signatures.popleft(), 'token_ids': [42]} for _ in messages]


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

    def test_reused_role_cannot_change_domain_or_type(self):
        instance = builder()
        instance.compile_signature('{flag:enum(False,True)}', ['action'])
        with self.assertRaisesRegex(AssertionError, 'different domain'):
            instance.compile_signature('{flag:enum(0,1)}', ['answer'])

    def test_unknown_roles_domain_operators_and_malformed_signatures_fail(self):
        for value in ('{unknown}', '{slot:unknown(1)}', '{slot:range(0,2)', '{slot!r:range(0,2)}'):
            with self.subTest(value=value), self.assertRaises((AssertionError, ValueError)):
                builder().compile_signature(value, ['action'])

    def test_domain_ast_rejects_nonliteral_arguments_and_duplicates(self):
        for expression in ("enum(__import__('os'))", 'range(0,2,1)', 'enum(1,1)', 'range(4,1)'):
            with self.subTest(expression=expression), self.assertRaises((AssertionError, ValueError)):
                builder().domain(expression)

    def test_repeat_fields_and_variant_capacities_are_explicit(self):
        with self.assertRaisesRegex(AssertionError, 'repeat count'):
            builder(max_fields=2).compile_signature('{x:repeat(3,range(0,2))}', ['action'])
        with self.assertRaisesRegex(AssertionError, 'variant capacity'):
            builder(max_variants=1).compile_signature('first\nsecond', ['action'])

    def test_real_public_integer_contract_needs_no_signature_generation(self):
        service = ScriptedService(['execute', {'kind': 'history', 'path': ['arguments', 'action']}])
        instance = builder(service)
        declaration = instance.build()
        self.assertEqual(declaration['fields'][0]['values'], [0, 1, 2, 3])
        self.assertEqual(declaration['action'], {'tool': 'execute', 'arguments': {'action': '${f0}'}})
        self.assertEqual(declaration['answer'], {'actions': {'$history': ['arguments', 'action']}})
        self.assertEqual(instance.tokens, 0)
        self.assertEqual(service.messages, [])

    def test_signature_generation_shares_budget_and_preserves_entire_context(self):
        service = ScriptedService(signatures=['emit {bit:range(0,2)}'])
        instance = builder(service)
        instance.budget['max_new_tokens'] = 1
        self.assertEqual(instance.signature({'type': 'string'}, ['action'], 'action'), 'emit ${f0}')
        self.assertEqual(service.caps, [1])
        self.assertIn(TASK['context'], service.messages[0][1]['content'])
        with self.assertRaisesRegex(AssertionError, 'generation budget'):
            instance.signature({'type': 'string'}, ['answer'], 'answer')

    def test_current_feedback_reaches_signature_without_replacing_original_context(self):
        service = ScriptedService(signatures=['emit {bit:range(0,2)}'])
        instance = builder(service)
        instance.feedback = {'operation': 'revise', 'active_declaration': {'action': 'emit ${f0}'},
            'current_action_observations': [{'action': 'emit 1', 'value': 'Previous action was rejected.'}],
            'execution': {'current_feedback': ['actual-event']}}
        instance.signature({'type': 'string'}, ['action'], 'action')
        user = service.messages[0][1]['content']
        self.assertIn(TASK['context'], user)
        self.assertIn('Previous action was rejected.', user)
        control = {key: value for key, value in instance.feedback.items() if key != 'execution'}
        self.assertIn(json.dumps(control, **EXECUTION['serialization']), user)
        self.assertEqual(json.loads(instance.state({}))['feedback'], instance.feedback)

    def test_missing_full_state_answer_is_explicit_and_preserves_action_trace(self):
        service = ScriptedService(['execute'])
        instance = builder(service)
        instance.answer_schema = {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'integer'}}}
        with self.assertRaisesRegex(ValueError, 'state reconstruction'):
            instance.build()
        self.assertEqual(instance.trace['declaration']['action']['arguments'], {'action': '${f0}'})
        self.assertIsNone(instance.trace['declaration']['answer'])


if __name__ == '__main__':
    unittest.main()
