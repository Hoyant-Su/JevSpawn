from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import time

import jsonschema

from baselines.common.errors import InvalidOutputError
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.state import pack_state


def assign(root, path, value):
    parent = root
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value


def finalize_answer(context, state, answer_schema, service, task_id, budget, settings, trace):
    validator_type = jsonschema.validators.validator_for(answer_schema)
    validator_type.check_schema(answer_schema)
    prompts = load_prompt(settings['prompts'])
    events = {event['id']: event for event in state['execution_events']}
    packed_state = pack_state(state)
    payload = {'actual_state': packed_state, 'answer_schema': answer_schema,
               'selected_action_history': [events[key]['action'] for key in state['current_path']]}
    root, pending, leaves = {}, [(('answer',), answer_schema, 0)], []
    trace.update(mode='typed_terminal_answer', calls=[], generated_tokens=0)
    started = time.perf_counter()
    while pending:
        choices, requests, following = [], [], []

        def choose(kind, path, schema, depth, values):
            if not values or len(values) > min(settings['max_candidates'], len(service.backend.answer_labels)):
                raise ValueError('Terminal choice count exceeds configured capacity: ' + str(path))
            choices.append((kind, path, schema, depth, values))
            requests.append({'id': str(len(requests)), 'context': context,
                'state': json.dumps(payload, ensure_ascii=False),
                'question': prompts['finite_question'].format(kind=kind,
                    path=json.dumps(path[1:]), schema=json.dumps(schema)),
                'options': [{'id': str(index), 'description': json.dumps(value, ensure_ascii=False)}
                            for index, value in enumerate(values)]})

        for path, schema, depth in pending:
            if depth > settings['max_depth']:
                raise ValueError('Terminal schema exceeds configured depth: ' + str(path))
            unsupported = set(schema) & {'$ref', '$dynamicRef', 'allOf', 'not', 'if', 'then', 'else',
                                         'dependentSchemas', 'dependentRequired', 'patternProperties', 'contains'}
            if unsupported:
                raise ValueError('Unsupported terminal schema keywords: ' + str(sorted(unsupported)))
            kind = schema.get('type')
            alternatives = schema.get('oneOf', schema.get('anyOf'))
            if alternatives is not None:
                base = {key: value for key, value in schema.items() if key not in ('oneOf', 'anyOf')}
                choose('schema', path, schema, depth, [{**base, **item} for item in alternatives])
            elif isinstance(kind, list):
                choose('schema', path, schema, depth, [{**schema, 'type': item} for item in kind])
            elif 'const' in schema or 'enum' in schema:
                values = [schema['const']] if 'const' in schema else schema['enum']
                choose('value', path, schema, depth, [value for value in values if validator_type(schema).is_valid(value)])
            elif kind == 'object':
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                if not set(required) <= set(properties):
                    raise ValueError('Terminal objects require declared property schemas: ' + str(path))
                assign(root, path, {})
                for key, child in properties.items():
                    child_path = (*path, key)
                    if key in required:
                        following.append((child_path, child, depth + 1))
                    else:
                        choose('presence', child_path, child, depth + 1, [False, True])
            elif kind == 'array':
                lower = schema.get('minItems', settings['min_array_items'])
                upper = schema.get('maxItems', settings['max_array_items'])
                if upper < lower or lower < settings['min_array_items'] or upper > settings['max_array_items']:
                    raise ValueError('Terminal array bounds conflict with configuration: ' + str(path))
                choose('length', path, schema, depth, list(range(lower, upper + 1)))
            elif kind in ('boolean', 'null'):
                choose('value', path, schema, depth, [False, True] if kind == 'boolean' else [None])
            elif kind == 'integer' and any(key in schema for key in ('minimum', 'exclusiveMinimum')) and any(
                    key in schema for key in ('maximum', 'exclusiveMaximum')):
                lower = math.ceil(schema['minimum']) if 'minimum' in schema else math.floor(schema['exclusiveMinimum']) + 1
                upper = math.floor(schema['maximum']) if 'maximum' in schema else math.ceil(schema['exclusiveMaximum']) - 1
                if 'exclusiveMinimum' in schema:
                    lower = max(lower, math.floor(schema['exclusiveMinimum']) + 1)
                if 'exclusiveMaximum' in schema:
                    upper = min(upper, math.ceil(schema['exclusiveMaximum']) - 1)
                if upper - lower + 1 > settings['max_candidates']:
                    raise ValueError('Terminal integer domain exceeds configured capacity: ' + str(path))
                choose('value', path, schema, depth,
                       [value for value in range(lower, upper + 1) if validator_type(schema).is_valid(value)])
            elif kind in ('string', 'number', 'integer'):
                leaves.append((path, schema))
            else:
                raise ValueError('Unsupported terminal schema: ' + json.dumps(schema))
        if requests:
            call = {'kind': 'terminal_finite', 'requests': requests}
            trace['calls'].append(call)
            begin = time.perf_counter()
            outputs = service.decide(requests, task_id=task_id)
            call.update(outputs=outputs, elapsed_seconds=time.perf_counter() - begin)
            for (kind, path, schema, depth, values), request, output in zip(choices, requests, outputs, strict=True):
                assert output['id'] == request['id'], 'Finite terminal output identity mismatch.'
                selected = values[int(output['choice'])]
                if kind == 'schema':
                    following.append((path, selected, depth + 1))
                elif kind == 'presence':
                    if selected:
                        following.append((path, schema, depth))
                elif kind == 'length':
                    assign(root, path, [None] * selected)
                    prefix = schema.get('prefixItems', [])
                    for index in range(selected):
                        item = prefix[index] if index < len(prefix) else schema.get('items')
                        if not isinstance(item, dict):
                            raise ValueError('Terminal array requires an item schema: ' + str(path))
                        following.append(((*path, index), item, depth + 1))
                else:
                    assign(root, path, selected)
        pending = following
    if leaves:
        quotient, remainder = divmod(budget['max_new_tokens'], len(leaves))
        if quotient == 0:
            raise ValueError('Terminal scalar count exceeds shared generation token budget.')
        groups = {}
        for index, (path, schema) in enumerate(leaves):
            cap = quotient + (index < remainder)
            messages = [{'role': 'system', 'content': prompts['leaf_system']},
                        {'role': 'user', 'content': prompts['leaf_user'].format(
                            context=context, state=json.dumps(packed_state, ensure_ascii=False),
                            answer_schema=json.dumps(answer_schema), path=json.dumps(path[1:]),
                            leaf_schema=json.dumps(schema), token_budget=cap,
                            selected_action_history=json.dumps(payload['selected_action_history']))}]
            groups.setdefault(cap, []).append((path, schema, messages))

        trace['allocated_generation_tokens'] = sum(cap * len(rows) for cap, rows in groups.items())
        jobs = []
        for cap, rows in groups.items():
            call = {'kind': 'terminal_raw_leaves', 'paths': [list(row[0][1:]) for row in rows],
                    'requests': [row[2] for row in rows], 'max_new_tokens_per_row': cap}
            trace['calls'].append(call)
            jobs.append((cap, rows, call))

        def generate(item):
            cap, rows, call = item
            begin = time.perf_counter()
            outputs = service.complete_batch(call['requests'], cap, budget['temperature'], None,
                                             task_id=task_id, return_tokens=True)
            call.update(outputs=outputs, elapsed_seconds=time.perf_counter() - begin)
            return rows, outputs, call

        with ThreadPoolExecutor(max_workers=len(groups)) as pool:
            generated = list(pool.map(generate, jobs))
        trace['generated_tokens'] = sum(len(output['token_ids'])
                                        for _, outputs, _ in generated for output in outputs)
        for rows, outputs, call in generated:
            for (path, schema, _), output in zip(rows, outputs, strict=True):
                try:
                    value = output['text'] if schema['type'] == 'string' else json.loads(output['text'])
                    json.dumps(value, allow_nan=False)
                    validator_type(schema).validate(value)
                except (ValueError, jsonschema.ValidationError) as error:
                    raise InvalidOutputError(str(error)) from error
                assign(root, path, value)
    try:
        validator_type(answer_schema).validate(root['answer'])
        json.dumps(root['answer'], allow_nan=False)
    except (ValueError, jsonschema.ValidationError) as error:
        raise InvalidOutputError(str(error)) from error
    trace.update(answer=deepcopy(root['answer']), elapsed_seconds=time.perf_counter() - started)
    return root['answer']
