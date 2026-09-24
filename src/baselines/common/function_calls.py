from copy import deepcopy

from jev_spawn.infra.prompts import load_prompt


def normalize_schema(schema, settings):
    value = deepcopy(schema)
    if 'type' in value:
        value['type'] = settings['type_aliases'].get(value['type'], value['type'])
    if 'properties' in value:
        value['properties'] = {name: normalize_schema(child, settings)
                               for name, child in value['properties'].items()}
        value.setdefault('additionalProperties', settings['additional_properties'])
    if 'items' in value:
        value['items'] = normalize_schema(value['items'], settings)
    return value


def parameter_map_schema(parameters, settings):
    return {'type': 'object', 'properties': parameters, 'required': list(parameters),
            'additionalProperties': settings['additional_properties']}


def function_call_task(task, functions, settings):
    parameter_schemas = {'json_schema': lambda value, configuration: value,
                         'parameter_map': parameter_map_schema}
    variants = []
    for function in functions:
        properties = {settings['name_field']: {'const': function['name']},
                      settings['arguments_field']: normalize_schema(parameter_schemas[settings['parameters_dialect']](
                          function[settings['parameters_key']], settings), settings)}
        variants.append({'type': 'object', 'description': function['description'],
                         'properties': properties, 'required': list(properties),
                         'additionalProperties': settings['additional_properties']})
    properties = {settings['answer_field']: {'type': 'array', 'items': {'oneOf': variants}}}
    return {**deepcopy(task), 'kind': 'function_calls',
            'instruction': load_prompt(settings['prompt'])['instruction'].format(original=task['instruction']),
            'answer_schema': {'type': 'object', 'properties': properties,
                              'required': list(properties),
                              'additionalProperties': settings['additional_properties']}}
