from copy import deepcopy
import json

from baselines.common.environment import TaskEnvironment
from baselines.common.schema_environment import SchemaObservationEnvironment


def enum_sources(value, path):
    if isinstance(value, dict):
        return [pair for key, item in value.items() for pair in enum_sources(item, [*path, key])]
    if not isinstance(value, list):
        return []
    pairs = [(value, {'path': path, 'projection': []})]
    if value and all(isinstance(item, dict) for item in value):
        keys = sorted(set.intersection(*(set(item) for item in value)))
        pairs.extend(([item[key] for item in value], {'path': path, 'projection': [key]}) for key in keys)
    pairs.extend(pair for index, item in enumerate(value) for pair in enum_sources(item, [*path, index]))
    return pairs


def compact_schema(schema, sources, policy, references, path):
    if isinstance(schema, list):
        return [compact_schema(item, sources, policy, references, [*path, index]) for index, item in enumerate(schema)]
    if not isinstance(schema, dict):
        return schema
    result = {}
    for key, value in schema.items():
        matches = [source for values, source in sources if json.dumps(values, sort_keys=True) ==
                   json.dumps(value, sort_keys=True)] if key == 'enum' else []
        if matches:
            references.append({'path': [*path, key], 'input': deepcopy(matches[0])})
            result[key] = {policy['enum_reference_key']: deepcopy(matches[0])}
        else:
            result[key] = compact_schema(value, sources, policy, references, [*path, key])
    return result


def compile_contract(task, tools, policy):
    sources = enum_sources(task['input'], [])
    references = []
    answer_path = [policy['definitions_key'], policy['answer_definition']]
    answer = compact_schema(task['answer_schema'], sources, policy, references, answer_path)
    displayed = deepcopy(tools)
    for name, definition in displayed.items():
        schema = definition['input_schema']
        path = ['tools', name, 'input_schema']
        if json.dumps(schema, sort_keys=True) == json.dumps(task['answer_schema'], sort_keys=True):
            definition['input_schema'] = {policy['reference_key']: policy['answer_reference']}
            references.append({'path': path, 'definition': answer_path})
        else:
            definition['input_schema'] = compact_schema(schema, sources, policy, references, path)
    references.append({'path': ['answer_schema'], 'definition': answer_path})
    return {'instruction': task['instruction'], 'input': deepcopy(task['input']),
            policy['definitions_key']: {policy['answer_definition']: answer},
            'answer_schema': {policy['reference_key']: policy['answer_reference']}, 'tools': displayed,
            policy['references_key']: references}


def expand_contract_value(value, contract, policy, path):
    reference = next((item for item in contract[policy['references_key']] if item['path'] == path), None)
    if reference is not None:
        if 'definition' in reference:
            target = contract
            for part in reference['definition']:
                target = target[part]
            return expand_contract_value(target, contract, policy, reference['definition'])
        values = contract['input']
        for part in reference['input']['path']:
            values = values[part]
        for part in reference['input']['projection']:
            values = [row[part] for row in values]
        return deepcopy(values)
    if isinstance(value, list):
        return [expand_contract_value(item, contract, policy, [*path, index]) for index, item in enumerate(value)]
    if not isinstance(value, dict):
        return value
    return {key: expand_contract_value(item, contract, policy, [*path, key]) for key, item in value.items()}


class DisplayContractEnvironment(SchemaObservationEnvironment):
    def __init__(self, *args, display_policy, display_prompts, **kwargs):
        super().__init__(*args, **kwargs)
        self.display_policy, self.display_prompts = deepcopy(display_policy), deepcopy(display_prompts)

    def contract(self, include_finish):
        tools = TaskEnvironment.display_tool_interface(self, include_finish)
        return compile_contract(self.task, tools, self.display_policy)

    def context(self, include_finish, serialization):
        return self.display_prompts['context'].format(
            enum_reference_key=self.display_policy['enum_reference_key'],
            contract=json.dumps(self.contract(include_finish), **serialization))

    def display_tool_interface(self, include_finish):
        return self.contract(include_finish)['tools']
