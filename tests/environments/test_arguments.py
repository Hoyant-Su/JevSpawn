from copy import deepcopy
import unittest

import jsonschema

from environments.arguments import validate_arguments


class ArgumentTransportTests(unittest.TestCase):
    def setUp(self):
        self.schema = {'type': 'object', 'properties': {
            'action': {'type': 'integer', 'minimum': 0, 'maximum': 3},
            'history': {'type': 'array', 'items': {'type': 'integer'}},
            'literal': {'type': 'string'}},
            'required': ['action', 'history', 'literal'], 'additionalProperties': False}

    def test_exact_json_numbers_are_typed_but_string_arguments_stay_literal(self):
        arguments = {'action': '3', 'history': ['0', 2], 'literal': '003'}
        validate_arguments(arguments, self.schema)
        self.assertEqual(arguments, {'action': 3, 'history': [0, 2], 'literal': '003'})

    def test_invalid_type_does_not_round_or_modify_arguments(self):
        for value in ['3.5', 'true', '4']:
            arguments = {'action': value, 'history': [], 'literal': 'x'}
            original = deepcopy(arguments)
            with self.assertRaises(jsonschema.ValidationError):
                validate_arguments(arguments, self.schema)
            self.assertEqual(arguments, original)

    def test_semantic_words_are_not_mapped_to_action_codes(self):
        with self.assertRaises(ValueError):
            validate_arguments({'action': 'south', 'history': [], 'literal': 'x'}, self.schema)

    def test_missing_fields_are_not_filled(self):
        with self.assertRaises(jsonschema.ValidationError):
            validate_arguments({'action': '3'}, self.schema)

    def test_unknown_argument_is_rejected_without_mapping_or_mutation(self):
        arguments = {'action': '3', 'history': [], 'literal': 'x', 'direction': 'left'}
        original = deepcopy(arguments)
        with self.assertRaises(jsonschema.ValidationError) as error:
            validate_arguments(arguments, self.schema)
        self.assertEqual(error.exception.validator, 'additionalProperties')
        self.assertEqual(arguments, original)


if __name__ == '__main__':
    unittest.main()
