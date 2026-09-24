import json
from pathlib import Path

import pytest

from jev_spawn.runtime.function_signature import SignatureDialectError, convert_function


ROOT = Path(__file__).parents[3]


def load_fixtures():
    fixture = json.loads((Path(__file__).parent / 'fixtures.json').read_text())
    dialect = json.loads((ROOT / fixture['dialect']).read_text())
    return fixture, dialect


def test_nested_signature_preserves_types_defaults_and_open_object_semantics():
    fixture, dialect = load_fixtures()
    converted = convert_function(fixture['signature'], dialect)
    assert converted['parameters'] == fixture['expected']
    assert 'additionalProperties' not in converted['parameters']


def test_unsupported_unconstrained_type_fails_explicitly():
    _, dialect = load_fixtures()
    function = {'name': 'opaque', 'parameters': {'type': 'any'}}
    with pytest.raises(SignatureDialectError, match='Unsupported signature type'):
        convert_function(function, dialect)


def test_optional_string_annotation_is_removed_from_required():
    _, dialect = load_fixtures()
    function = {'name': 'x', 'parameters': {'type': 'dict', 'properties': {'a': {'type': 'integer'}},
                                             'required': ['a'], 'optional': 'a'}}
    assert 'required' not in convert_function(function, dialect)['parameters']
