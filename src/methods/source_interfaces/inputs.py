from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.evidence_interfaces.interfaces import InvalidResponse, compact
from methods.source_interfaces.schema import NONE


def fields(units, questions, prompts):
    state = compact({'source_passages': [{'index': i, **unit} for i, unit in enumerate(units)]})
    options = [{'id': unit['id'], 'description': prompts['candidate_description'].format(index=i)}
               for i, unit in enumerate(units)]
    options.append({'id': NONE, 'description': prompts['no_evidence']})
    return [{'id': identity, 'state': state, 'options': options,
             'question': prompts['worker_question'].format(question=question)}
            for identity, question in questions]


def input_size(backend, group, prompts):
    lengths = []
    for field in group:
        labels = [chr(65 + i) for i in range(len(field['options']))]
        for instruction in [CONTROLLER['output_instruction'], prompts['worker_json_instruction']]:
            user = controller_prompts([field['state']], field['question'], field['options'], labels, instruction)
            rendered = backend._render(user, CONTROLLER['system'])[0]
            lengths.append(len(backend.tokenizer(rendered, add_special_tokens=False)['input_ids']))
    return max(lengths)


def partition(backend, units, questions, fixed, prompts):
    groups = []
    capacity = fixed['source_group_capacity']

    def place(items):
        group = fields(items, questions, prompts)
        if input_size(backend, group, prompts) <= fixed['context_tokens']:
            groups.append(group)
        elif len(items) == 1:
            raise InvalidResponse('An original source unit exceeds the configured input limit.')
        else:
            middle = len(items) // 2
            place(items[:middle])
            place(items[middle:])

    for start in range(0, len(units), capacity):
        place(units[start:start + capacity])
    return groups


