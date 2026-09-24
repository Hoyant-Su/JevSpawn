import json

import jsonschema


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class InvalidResponse(Exception):
    pass


def unique(values, name):
    if len(values) != len(set(values)):
        raise InvalidResponse(f'Duplicate {name} in model response')


def fields(value, round_index, previous):
    questions = [item['question'] for item in value]
    unique(questions, 'evidence questions')
    if set(questions) & {item['question'] for item in previous}:
        raise InvalidResponse('Refinement repeats an earlier evidence question')
    output = []
    for index, item in enumerate(value):
        unique(item['options'], 'option descriptions')
        output.append({'id': f'r{round_index}q{index}', 'question': item['question'],
                       'options': [{'id': chr(65 + slot), 'description': text}
                                   for slot, text in enumerate(item['options'])]})
    return output


def parsed(record, contract):
    result = record['result']
    if any(result['truncated']):
        raise InvalidResponse('Generation reached its token limit before EOS')
    values = []
    for text in result['texts']:
        try:
            value = json.loads(text)
            jsonschema.validate(value, contract)
        except (json.JSONDecodeError, jsonschema.ValidationError) as error:
            raise InvalidResponse(str(error)) from error
        values.append(value)
    return values


def evidence_view(definitions, messages):
    return {'questions': definitions, 'columns': [field['id'] for field in definitions],
            'null_semantics': 'No judgment is supplied for this item and question.',
            'items': [[index, [value[field['id']] if field['id'] in value else None
                              for field in definitions]] for index, value in enumerate(messages)]}


def receiver_input(collection, definitions, messages, contract, settings, prompts):
    return prompts['receiver_user'].format(task=settings['task'], query=collection['query'],
        evidence=compact(evidence_view(definitions, messages)), schema=compact(contract))

