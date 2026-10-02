import json

import jsonschema

from demo.environments.common import ROOT


TYPES = json.loads((ROOT / 'configs/environments/argument_transport.json').read_text())


def typed_value(value, schema):
    kind = schema['type']
    if isinstance(value, str) and kind in TYPES['json_scalar_types']:
        return json.loads(value)
    if kind == TYPES['object_type'] and isinstance(value, dict):
        return {key: typed_value(item, schema['properties'][key])
                if key in schema['properties'] else item for key, item in value.items()}
    if kind == TYPES['array_type'] and isinstance(value, list):
        return [typed_value(item, schema['items']) for item in value]
    return value


def validate_arguments(arguments, schema):
    """Decode scalar JSON representations using the declared tool interface."""
    normalized = typed_value(arguments, schema)
    jsonschema.validate(normalized, schema)
    arguments.clear()
    arguments.update(normalized)
