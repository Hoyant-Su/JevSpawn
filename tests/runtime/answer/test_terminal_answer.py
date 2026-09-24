from copy import deepcopy
import json
from pathlib import Path
import unittest

import jsonschema

from baselines.common.errors import InvalidOutputError
from jev_spawn.runtime.answer import evidence_bindings, finalize_answer


class RecordedService:
    def __init__(self, select, texts):
        self.select, self.texts, self.calls = select, iter(texts), []

    def decide(self, requests, *, task_id):
        answers = []
        for request in requests:
            options = [json.loads(item['description']) for item in request['options']]
            index = self.select(request, options)
            self.calls.append({'request': deepcopy(request), 'choice': index})
            answers.append({'id': request['id'], 'choice': str(index)})
        return answers

    def complete_batch(self, messages, cap, temperature, stop, *, task_id, return_tokens):
        self.calls.append({'messages': deepcopy(messages), 'cap': cap, 'temperature': temperature})
        return [{'text': next(self.texts), 'token_ids': [5, 12, 8]}]


def construct(request, options):
    choices = [index for index, value in enumerate(options)
               if isinstance(value, dict) and value.get('operation') == 'construct']
    return choices[0] if choices else 0


def bind_commands(request, options):
    path = ['selected_action_history', '*', 'arguments', 'move']
    choices = [index for index, value in enumerate(options)
               if isinstance(value, dict) and path in value.get('sources', [])]
    return choices[0] if choices else construct(request, options)


class TerminalAnswerTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(Path('configs/baselines/common/methods/jevspawn.json').read_text())['settings']['terminal_answer']
        self.budget = {'max_new_tokens': 2048, 'temperature': 0}
        self.state = {'execution_events': [
            {'id': 'sibling', 'parent_feedback': [], 'action': {'arguments': {'move': 'wrong'}}, 'value': 'Sibling observation'},
            {'id': 'a', 'parent_feedback': [], 'action': {'arguments': {'move': 'left'}}, 'value': 'First actual observation'},
            {'id': 'b', 'parent_feedback': ['a'], 'action': {'arguments': {'move': 'down'}}, 'value': 'Second actual observation'}],
            'current_path': ['a', 'b'], 'current_feedback': ['b'], 'computed_fields': [],
            'field_evidence': {}, 'bound_fields': {}, 'declaration_feedback': []}
        self.schema = {'type': 'object', 'properties': {'answer': {'type': 'string'}},
                       'required': ['answer'], 'additionalProperties': False}
        self.trace = {}

    def run_answer(self, service):
        return finalize_answer('Shared official question and rules.', self.state, self.schema,
                               service, 'sample', self.budget, self.settings, self.trace)

    def test_unseen_answer_key_and_quoted_string_are_serialized(self):
        self.schema['properties'] = {'unseen_result_name': {'type': 'string'}}
        self.schema['required'] = ['unseen_result_name']
        text = 'A quote: "value"\nA brace: }'
        service = RecordedService(construct, [text])
        answer = self.run_answer(service)
        self.assertEqual(answer, {'unseen_result_name': text})
        self.assertEqual(json.loads(json.dumps(answer)), answer)
        self.assertEqual(self.trace['generated_tokens'], 3)
        self.assertEqual(self.trace['mode'], 'typed_terminal_answer')

    def test_executed_history_is_bound_without_text_generation(self):
        self.schema = {'type': 'object', 'required': ['record'], 'additionalProperties': False,
                       'properties': {'record': {'type': 'array', 'items': {'type': 'string'}}}}
        service = RecordedService(bind_commands, [])
        self.assertEqual(self.run_answer(service), {'record': ['left', 'down']})
        self.assertEqual(self.trace['generated_tokens'], 0)
        self.assertTrue(all('request' in call for call in service.calls))
        self.assertNotIn('wrong', json.dumps(self.trace['answer']))

    def test_projections_preserve_order_types_and_real_values(self):
        sources = {'log': [{'value': 5, 'tag': 'first'}, {'value': 8, 'tag': 'second'}]}
        candidates = evidence_bindings(sources, {'type': 'array', 'items': {'type': 'integer'}})
        self.assertEqual(candidates, [{'value': [5, 8], 'sources': [['log', '*', 'value']]}])
        self.assertEqual(sources['log'][0]['value'], 5)

    def test_nonuniform_projection_does_not_invent_missing_values(self):
        candidates = evidence_bindings({'log': [{'value': 5}, {'other': 8}]},
                                       {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1})
        self.assertEqual(candidates, [])

    def test_generated_scalar_budget_is_shared_across_properties(self):
        self.schema = {'type': 'object', 'properties': {'x': {'type': 'string'}, 'y': {'type': 'integer'}},
                       'required': ['x', 'y'], 'additionalProperties': False}
        service = RecordedService(construct, ['result', '17'])
        self.assertEqual(self.run_answer(service), {'x': 'result', 'y': 17})
        calls = [call for call in service.calls if 'messages' in call]
        self.assertEqual([call['cap'] for call in calls], [2048, 2045])
        self.assertEqual(self.trace['generated_tokens'], 6)

    def test_invalid_numeric_leaf_is_not_repaired(self):
        self.schema = {'type': 'integer'}
        for text in ['seventeen', '1.5', 'NaN', 'Infinity']:
            with self.subTest(text=text), self.assertRaises(InvalidOutputError):
                self.run_answer(RecordedService(construct, [text]))

    def test_optional_presence_uses_configured_values(self):
        self.settings['presence_options'] = [True, False]
        self.schema = {'type': 'object', 'properties': {'extra': {'type': 'string'}},
                       'additionalProperties': False}
        self.assertEqual(self.run_answer(RecordedService(construct, ['included'])), {'extra': 'included'})

    def test_array_construction_uses_declared_length_and_schema(self):
        self.schema = {'type': 'array', 'minItems': 2, 'maxItems': 2, 'items': {'type': 'integer'}}
        self.assertEqual(self.run_answer(RecordedService(construct, ['7', '11'])), [7, 11])

    def test_constraints_apply_to_evidence_candidates(self):
        candidates = evidence_bindings({'x': [1, 2], 'y': [3, 4]},
            {'type': 'array', 'items': {'type': 'integer', 'minimum': 3}, 'minItems': 2})
        self.assertEqual([item['value'] for item in candidates], [[3, 4]])

    def test_schema_and_missing_event_fail_before_model_calls(self):
        service = RecordedService(construct, [])
        self.schema = {'type': 'nonexistent-type'}
        with self.assertRaises(jsonschema.SchemaError):
            self.run_answer(service)
        self.schema = {'type': 'string'}
        self.state['current_path'] = ['missing']
        with self.assertRaises(KeyError):
            self.run_answer(service)
        self.assertEqual(service.calls, [])


if __name__ == '__main__':
    unittest.main()
