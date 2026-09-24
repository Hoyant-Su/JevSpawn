import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from baselines.common.errors import InvalidOutputError
from typed_answer import finalize_answer
from jev_spawn.runtime.state import pack_state


PROMPTS = {'finite_question': '{kind}|{path}|{schema}', 'leaf_system': 'Provide the raw scalar value.',
           'leaf_user': '{context}\n{state}\n{answer_schema}\n{path}\n{leaf_schema}\n{token_budget}\n{selected_action_history}'}


class Service:
    def __init__(self, selections, strings):
        self.selections, self.strings, self.finite_calls, self.generation_calls = selections, strings, [], []
        self.backend = SimpleNamespace(answer_labels=tuple(range(16)))

    def decide(self, requests, *, task_id):
        self.finite_calls.append(requests)
        results = []
        for request in requests:
            kind, path, _ = request['question'].split('|', 2)
            path = tuple(json.loads(path))
            options = [json.loads(option['description']) for option in request['options']]
            value = self.selections(kind, path, options)
            results.append({'id': request['id'], 'choice': str(options.index(value))})
        return results

    def complete_batch(self, requests, cap, temperature, stop, *, task_id, return_tokens):
        self.generation_calls.append((requests, cap))
        return [{'text': self.strings[tuple(json.loads(messages[1]['content'].splitlines()[3]))],
                 'token_ids': [13]} for messages in requests]


class TypedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.settings = {'prompts': 'qualification', 'min_array_items': 0, 'max_array_items': 12,
                         'max_depth': 8, 'max_candidates': 16}
        self.budget = {'max_new_tokens': 10, 'temperature': 0}
        self.state = {'execution_events': [{'id': 'a', 'action': 'move left', 'value': 'Actual feedback'},
                                           {'id': 'sibling', 'action': 'move up', 'value': 'Sibling feedback'}],
                      'current_path': ['a'], 'computed_fields': [], 'current_feedback': ['a'],
                      'declaration_feedback': [], 'bound_fields': {}, 'field_evidence': {}}
        self.trace = {}

    def run_answer(self, schema, service):
        with patch('typed_answer.load_prompt', return_value=PROMPTS):
            return finalize_answer('Official task context', self.state, schema, service,
                                   'sample', self.budget, self.settings, self.trace)

    def test_fixed_nested_grid_has_81_independent_finite_values(self):
        schema = {'type': 'array', 'minItems': 9, 'maxItems': 9, 'items': {
            'type': 'array', 'minItems': 9, 'maxItems': 9,
            'items': {'type': 'integer', 'minimum': 1, 'maximum': 9}}}
        service = Service(lambda kind, path, options: (path[-1] + 1 if kind == 'value' else 9), {})
        answer = self.run_answer(schema, service)
        self.assertEqual(answer, [list(range(1, 10)) for _ in range(9)])
        self.assertEqual([len(call) for call in service.finite_calls], [1, 9, 81])
        self.assertEqual(service.generation_calls, [])
        payload = json.loads(service.finite_calls[-1][0]['state'])
        self.assertEqual(payload['actual_state'], pack_state(self.state))
        self.assertEqual(payload['selected_action_history'], ['move left'])

    def test_strings_are_preserved_and_total_allocated_tokens_obey_shared_budget(self):
        schema = {'type': 'object', 'properties': {key: {'type': 'string'} for key in ('x', 'y', 'z')},
                  'required': ['x', 'y', 'z'], 'additionalProperties': False}
        strings = {('x',): 'a "quoted" value', ('y',): 'literal { } brackets', ('z',): 'not JSON at all'}
        service = Service(lambda kind, path, options: options[0], strings)
        answer = self.run_answer(schema, service)
        self.assertEqual(answer, {path[0]: value for path, value in strings.items()})
        self.assertEqual(sum(len(messages) * cap for messages, cap in service.generation_calls), 10)
        self.assertEqual(sorted(cap for _, cap in service.generation_calls), [3, 4])
        self.assertEqual(self.trace['generated_tokens'], 3)
        self.assertEqual(json.loads(service.generation_calls[0][0][0][1]['content'].splitlines()[1]),
                         pack_state(self.state))
        self.assertEqual(self.trace['allocated_generation_tokens'], self.budget['max_new_tokens'])
        self.assertEqual(json.loads(json.dumps(answer)), answer)

    def test_optional_union_variable_array_and_bool(self):
        schema = {'type': 'object', 'properties': {
            'omit': {'type': 'string'}, 'items': {'type': 'array', 'minItems': 1, 'maxItems': 3,
                'items': {'oneOf': [{'type': 'boolean'}, {'type': 'null'}]}}}, 'required': ['items']}
        def select(kind, path, options):
            return {'presence': False, 'length': 2, 'schema': {'type': 'boolean'}, 'value': True}[kind]
        self.assertEqual(self.run_answer(schema, Service(select, {})), {'items': [True, True]})

    def test_numeric_errors_remain_real_failures_with_raw_outputs(self):
        schema = {'type': 'object', 'properties': {'x': {'type': 'number'}}, 'required': ['x']}
        for value in ('1 trailing text', 'NaN', '1e999', 'true', '"1"'):
            with self.subTest(value=value), self.assertRaises(InvalidOutputError):
                self.run_answer(schema, Service(lambda *args: None, {('x',): value}))
            self.assertEqual(self.trace['calls'][-1]['outputs'][0]['text'], value)
            self.assertNotIn('answer', self.trace)

    def test_declared_domains_are_never_truncated_to_capacity(self):
        service = Service(lambda *args: None, {})
        with self.assertRaisesRegex(ValueError, 'bounds conflict'):
            self.run_answer({'type': 'array', 'minItems': 13, 'maxItems': 13,
                             'items': {'type': 'boolean'}}, service)
        service.backend.answer_labels = ('A',)
        with self.assertRaisesRegex(ValueError, 'choice count exceeds'):
            self.run_answer({'type': 'boolean'}, service)
        self.assertEqual(service.finite_calls, [])

    def test_unsupported_schema_and_insufficient_budget_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported terminal schema keywords'):
            self.run_answer({'$ref': '#/$defs/n', '$defs': {'n': {'type': 'number'}}}, Service(None, {}))
        self.budget['max_new_tokens'] = 1
        with self.assertRaisesRegex(ValueError, 'scalar count exceeds'):
            self.run_answer({'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'string'}},
                            Service(lambda kind, path, options: 2, {}))


if __name__ == '__main__':
    unittest.main()
