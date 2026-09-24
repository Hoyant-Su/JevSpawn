from methods.evidence_interfaces.interfaces import InvalidResponse, compact
from methods.source_interfaces.inputs import input_size
from methods.source_interfaces.schema import answer_schema


def fields(units, questions, prompts):
    state = compact({'source_passages': [{'index': i, **unit} for i, unit in enumerate(units)]})
    options = []
    for mask in range(1 << len(units)):
        indices = [i for i in range(len(units)) if mask & (1 << i)]
        options.append({'id': f'set{mask}', 'source_ids': [units[i]['id'] for i in indices],
                        'description': prompts['candidate_description'].format(indices=compact(indices))
                        if indices else prompts['no_evidence']})
    return [{'id': identity, 'state': state, 'options': options,
             'question': prompts['worker_question'].format(question=question)}
            for identity, question in questions]


def partition(backend, units, questions, fixed, prompts):
    groups = []
    capacity = fixed['source_group_capacity']
    assert 2 <= capacity and 2 ** capacity <= fixed['max_options_per_field'] <= 26

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


def final_prompt(query, references, units, prompts):
    contract = answer_schema(len(units))
    positions = {unit['id']: i for i, unit in enumerate(units)}
    requests = [{'question': value['question'],
                 'source_indices': [positions[identity] for identity in value['source_ids']]}
                for value in references]
    prompt = prompts['final_user'].format(query=query, questions=compact(requests),
        evidence=compact([{'index': i, **unit} for i, unit in enumerate(units)]), schema=compact(contract))
    return prompt, contract


def selected_ids(references):
    return list(dict.fromkeys(identity for value in references for identity in value['source_ids']))


def final_size(backend, query, references, indexed, prompts):
    units = [indexed[identity] for identity in selected_ids(references)]
    prompt, _ = final_prompt(query, references, units, prompts)
    rendered = backend._render([prompt], prompts['final_system'])[0]
    return len(backend.tokenizer(rendered, add_special_tokens=False)['input_ids'])
