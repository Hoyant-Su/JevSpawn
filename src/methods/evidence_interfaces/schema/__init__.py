from copy import deepcopy


def worker(letters):
    return {'type': 'object', 'properties': {'choice': {'enum': letters}},
            'required': ['choice'], 'additionalProperties': False}


def questions(field, limits):
    item = deepcopy(field)
    item['properties']['options']['maxItems'] = limits['max_options_per_field']
    return {'type': 'array', 'minItems': 1, 'maxItems': limits['max_fields_per_round'], 'items': item}


def planner(field, limits):
    return {'type': 'object', 'properties': {'questions': questions(field, limits)},
            'required': ['questions'], 'additionalProperties': False}


def ranking(count, cutoff):
    return {'type': 'object', 'properties': {'ranking': {'type': 'array', 'minItems': cutoff,
            'maxItems': cutoff, 'items': {'type': 'integer', 'minimum': 0, 'maximum': count - 1}}},
            'required': ['ranking'], 'additionalProperties': False}


def coordinator(field, limits, count, cutoff):
    finish = ranking(count, cutoff)
    finish['properties'] = {'action': {'const': 'finish'}, **finish['properties']}
    finish['required'] = ['action', 'ranking']
    refine = {'type': 'object', 'properties': {'action': {'const': 'refine'},
              'documents': {'type': 'array', 'minItems': 1, 'maxItems': count,
                            'items': {'type': 'integer', 'minimum': 0, 'maximum': count - 1}},
              'questions': questions(field, limits)},
              'required': ['action', 'documents', 'questions'], 'additionalProperties': False}
    return {'anyOf': [finish, refine]}
