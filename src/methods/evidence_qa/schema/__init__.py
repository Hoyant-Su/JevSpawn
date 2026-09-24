from copy import deepcopy

from methods.evidence_interfaces.schema import questions


def interface(field, limits):
    item = deepcopy(field)
    item['properties']['search_query'] = {'type': 'string', 'minLength': 1}
    item['required'].append('search_query')
    return {'type': 'object', 'properties': {'questions': questions(item, limits)},
            'required': ['questions'], 'additionalProperties': False}


def answer(count):
    return {'type': 'object', 'properties': {
        'reasoning': {'type': 'string'}, 'answer': {'type': 'string', 'minLength': 1},
        'evidence': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0, 'maximum': count - 1}}},
        'required': ['reasoning', 'answer', 'evidence'], 'additionalProperties': False}


def receiver(field, limits, count):
    finish = answer(count)
    finish['properties'] = {'action': {'const': 'finish'}, **finish['properties']}
    finish['required'] = ['action', *finish['required']]
    refine = interface(field, limits)
    refine['properties'] = {'action': {'const': 'refine'}, **refine['properties']}
    refine['required'] = ['action', 'questions']
    return {'anyOf': [finish, refine]}
