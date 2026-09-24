"""Conversion between external function-signature dialects and JSON Schema."""

from copy import deepcopy


class SignatureDialectError(ValueError):
    """Raised when a source signature cannot be represented faithfully."""


def convert_schema(schema, dialect):
    """Convert one source schema using an explicit dialect configuration.

    The converter retains source annotations and does not infer closed objects,
    defaults, cardinalities, or candidate values. Unsupported source types fail
    before a task can enter the execution path.
    """
    if not isinstance(schema, dict):
        raise SignatureDialectError('A signature schema must be an object.')
    source_type = schema.get('type')
    if source_type in dialect.get('unsupported_types', []):
        raise SignatureDialectError(f'Unsupported signature type: {source_type!r}')
    type_map = dialect['type_map']
    if source_type not in type_map:
        raise SignatureDialectError(f'Unsupported signature type: {source_type!r}')
    target_type = type_map[source_type]
    converted = {}
    for key in dialect['preserved_keywords']:
        if key in schema:
            converted[key] = deepcopy(schema[key])
    converted['type'] = target_type
    if source_type == dialect['object_type']:
        properties = schema.get('properties')
        if properties is None:
            if 'additionalProperties' in schema:
                converted['additionalProperties'] = _convert_additional(schema['additionalProperties'], dialect)
            return converted
        converted['properties'] = {name: convert_schema(value, dialect)
                                   for name, value in properties.items()}
        required = set(schema.get('required', []))
        optional = _optional_names(schema.get('optional', []))
        if not required <= set(properties):
            raise SignatureDialectError('Required signature fields must be declared properties.')
        required -= optional
        if required:
            converted['required'] = sorted(required)
        if 'additionalProperties' in schema:
            converted['additionalProperties'] = _convert_additional(schema['additionalProperties'], dialect)
    elif source_type in (dialect['array_type'], dialect['tuple_type']):
        items = schema.get('items')
        if isinstance(items, list):
            converted['prefixItems'] = [convert_schema(item, dialect) for item in items]
            if 'additionalItems' in schema:
                converted['items'] = _convert_additional(schema['additionalItems'], dialect)
        elif isinstance(items, dict):
            converted['items'] = convert_schema(items, dialect)
        else:
            raise SignatureDialectError('Array signatures require an items schema.')
    return converted


def convert_function(function, dialect):
    """Convert one function descriptor while retaining its public identity."""
    if not isinstance(function, dict) or not isinstance(function.get('parameters'), dict):
        raise SignatureDialectError('A function descriptor requires parameters.')
    result = {key: deepcopy(value) for key, value in function.items() if key != 'parameters'}
    result['parameters'] = convert_schema(function['parameters'], dialect)
    return result


def convert_functions(functions, dialect):
    """Convert an ordered function catalog without merging signatures."""
    if not isinstance(functions, list):
        raise SignatureDialectError('The function catalog must be a list.')
    return [convert_function(function, dialect) for function in functions]


def _optional_names(value):
    if isinstance(value, list):
        return set(value)
    if isinstance(value, bool):
        return set()
    if isinstance(value, str):
        return {value} if value else set()
    raise SignatureDialectError('The optional annotation must be a list, string, or boolean.')


def _convert_additional(value, dialect):
    if isinstance(value, dict):
        return convert_schema(value, dialect)
    if isinstance(value, bool):
        return value
    raise SignatureDialectError('additionalProperties must be a schema or boolean.')
