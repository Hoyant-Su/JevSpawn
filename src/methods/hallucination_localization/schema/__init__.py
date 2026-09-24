def partition(count):
    return {'type': 'object', 'properties': {
        'ends': {'type': 'array', 'items': {'type': 'integer', 'minimum': 1, 'maximum': count},
                 'minItems': 1, 'maxItems': count}}, 'required': ['ends'], 'additionalProperties': False}


def spans():
    return {'type': 'object', 'properties': {'spans': {'type': 'array', 'items': {
        'type': 'object', 'properties': {'text': {'type': 'string', 'minLength': 1},
                                        'occurrence': {'type': 'integer', 'minimum': 1}},
        'required': ['text', 'occurrence'], 'additionalProperties': False}}},
        'required': ['spans'], 'additionalProperties': False}


def choice(letters):
    return {'type': 'object', 'properties': {'choice': {'type': 'string', 'enum': letters}},
            'required': ['choice'], 'additionalProperties': False}
