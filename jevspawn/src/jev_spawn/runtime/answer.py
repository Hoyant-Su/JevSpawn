from copy import deepcopy
import json
import time

import jsonschema

from baselines.common.errors import InvalidOutputError
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.state import pack_state


def value_nodes(value, path):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from value_nodes(child, [*path, key])
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from value_nodes(child, [*path, index])


def evidence_bindings(sources, schema):
    validator = jsonschema.Draft202012Validator(schema)
    candidates = {}
    for path, value in value_nodes(sources, []):
        values = [(path, value)]
        if isinstance(value, list) and value:
            columns = [dict((tuple(location), item) for location, item in value_nodes(row, [])) for row in value]
            common = set.intersection(*(set(column) for column in columns))
            values.extend(([*path, '*', *location], [column[location] for column in columns])
                          for location in sorted(common, key=json.dumps))
        for location, candidate in values:
            if validator.is_valid(candidate):
                key = json.dumps(candidate, sort_keys=True, ensure_ascii=False, allow_nan=False)
                if key not in candidates:
                    candidates[key] = {'value': candidate, 'sources': []}
                candidates[key]['sources'].append(location)
    return list(candidates.values())


class AnswerBuilder:
    def __init__(self, context, state, schema, service, task_id, budget, settings, trace):
        self.context, self.schema, self.service, self.task_id = context, schema, service, task_id
        self.budget, self.settings, self.trace = budget, settings, trace
        self.prompts = load_prompt(settings['prompts'])
        events = {event['id']: event for event in state['execution_events']}
        selected = [events[identity] for identity in state['current_path']]
        self.sources = {'selected_action_history': [event['action'] for event in selected],
                        'selected_observations': [event['value'] for event in selected],
                        'computed_values': {field['id']: field['value'] for field in state['computed_fields']}}
        self.evidence = {'actual_state': pack_state(state), 'selected_action_history': self.sources['selected_action_history'],
                         'answer_schema': schema, 'construction_limits': settings,
                         'source_format': self.prompts['sources']}
        trace.update(calls=[], generated_tokens=0, finite_decisions=0, mode='typed_terminal_answer')

    def choose(self, question, options, component):
        request = {'id': self.settings['decision_id'].format(index=self.trace['finite_decisions']),
                   'context': self.context, 'state': json.dumps({**self.evidence, 'component': component}),
                   'question': question, 'options': [{'id': str(index), 'description': json.dumps(option)}
                                                    for index, option in enumerate(options)]}
        started = time.perf_counter()
        decision, = self.service.decide([request], task_id=self.task_id)
        self.trace['finite_decisions'] += 1
        self.trace['calls'].append({'kind': 'terminal_finite', 'requests': [request], 'outputs': [decision],
                                   'elapsed_seconds': time.perf_counter() - started})
        return int(decision['choice'])

    def scalar(self, schema, path):
        remaining = self.budget['max_new_tokens'] - self.trace['generated_tokens']
        if remaining <= 0:
            raise InvalidOutputError('Final answer scalar generation exhausted its shared token budget.')
        messages = [{'role': 'system', 'content': self.prompts['scalar_system'].format(
                        schema=json.dumps(schema), path=json.dumps(path))},
                    {'role': 'user', 'content': self.prompts['scalar_user'].format(
                        context=self.context, evidence=json.dumps(self.evidence), schema=json.dumps(schema), path=json.dumps(path))}]
        started = time.perf_counter()
        output, = self.service.complete_batch([messages], remaining, self.budget['temperature'], None,
                                              task_id=self.task_id, return_tokens=True)
        self.trace['generated_tokens'] += len(output['token_ids'])
        self.trace['calls'].append({'kind': 'terminal_scalar', 'requests': [messages], 'outputs': [output],
                                   'elapsed_seconds': time.perf_counter() - started})
        value = output['text'] if schema['type'] == 'string' else json.loads(output['text'])
        jsonschema.validate(value, schema)
        json.dumps(value, allow_nan=False)
        return value

    def node(self, schema, path):
        if 'const' in schema:
            return deepcopy(schema['const'])
        if 'enum' in schema:
            index = self.choose(self.prompts['enum'], schema['enum'], {'path': path, 'schema': schema})
            return deepcopy(schema['enum'][index])
        if 'oneOf' in schema or 'anyOf' in schema:
            alternatives = schema['oneOf'] if 'oneOf' in schema else schema['anyOf']
            index = self.choose(self.prompts['variant'], alternatives, {'path': path, 'schema': schema})
            return self.node(alternatives[index], path)
        kinds = schema['type'] if isinstance(schema['type'], list) else [schema['type']]
        index = self.choose(self.prompts['type'], kinds, {'path': path, 'schema': schema})
        schema = {**schema, 'type': kinds[index]}
        component = {'path': path, 'schema': schema}
        bindings = evidence_bindings(self.sources, schema)
        options = [{'operation': 'copy_evidence', 'sources': binding['sources']} for binding in bindings]
        options.append({'operation': 'construct', 'description': self.prompts['construct']})
        selected = self.choose(self.prompts['binding'], options, component)
        if selected < len(bindings):
            return deepcopy(bindings[selected]['value'])
        kind = schema['type']
        if kind == 'object':
            result = {}
            for key, child in schema['properties'].items():
                present = key in schema.get('required', []) or self.settings['presence_options'][self.choose(
                    self.prompts['optional'], self.settings['presence_options'], {'path': [*path, key], 'schema': child})]
                if present:
                    result[key] = self.node(child, [*path, key])
            return result
        if kind == 'array':
            lower = schema.get('minItems', self.settings['min_array_items'])
            upper = min(schema.get('maxItems', self.settings['max_array_items']), self.settings['max_array_items'])
            counts = list(range(lower, upper + 1))
            count = counts[self.choose(self.prompts['count'], counts, component)]
            return [self.node(schema['items'], [*path, index]) for index in range(count)]
        if kind in self.settings['finite_scalar_values']:
            values = self.settings['finite_scalar_values'][kind]
            return values[self.choose(self.prompts['enum'], values, component)]
        return self.scalar(schema, path)


def finalize_answer(context, state, answer_schema, service, task_id, budget, settings, trace):
    jsonschema.Draft202012Validator.check_schema(answer_schema)
    started = time.perf_counter()
    builder = AnswerBuilder(context, state, answer_schema, service, task_id, budget, settings, trace)
    try:
        answer = builder.node(answer_schema, [])
        jsonschema.validate(answer, answer_schema)
        json.dumps(answer, allow_nan=False)
    except (ValueError, jsonschema.ValidationError) as error:
        raise InvalidOutputError(str(error)) from error
    trace.update(answer=deepcopy(answer), elapsed_seconds=time.perf_counter() - started)
    return answer
