from copy import deepcopy
from itertools import count
import json

from baselines.common.errors import TaskLimitError


class OutputSlots:
    """Own output addresses while the model supplies semantic keys and values."""

    def __init__(self, settings):
        self.settings = settings
        self.serial = count(settings['initial_index'])
        self.output, self.pending = {}, {}
        self.root = self.allocate(self.output, settings['root_key'], [])

    def allocate(self, container, key, path):
        identity = self.settings['slot_id'].format(index=next(self.serial))
        self.pending[identity] = (container, key, path)
        return identity

    def frontier(self):
        return [{'id': identity, 'path': deepcopy(path)}
                for identity, (_, _, path) in self.pending.items()]

    def ready(self, schemas):
        pending_paths = [tuple(path) for _, _, path in self.pending.values()]
        ready = []
        for slot in self.frontier():
            dependencies = [(*slot['path'][:-1], key)
                            for key in schemas[slot['id']].get('depends_on', [])]
            if all(not any(path[:len(dependency)] == dependency for path in pending_paths)
                   for dependency in dependencies):
                ready.append(slot)
        if self.pending and not ready:
            raise ValueError('Output slot dependencies contain a cycle.')
        return ready

    def assign(self, identity, value):
        container, key, _ = self.pending.pop(identity)
        container[key] = deepcopy(value)

    def object(self, identity, keys):
        if len(set(keys)) != len(keys):
            raise ValueError('Object keys must be unique.')
        parent, key, path = self.pending.pop(identity)
        value = dict.fromkeys(keys)
        parent[key] = value
        return [self.allocate(value, child, [*path, child]) for child in keys]

    def array(self, identity, size):
        if size < self.settings['initial_index']:
            raise ValueError('Array size must be nonnegative.')
        parent, key, path = self.pending.pop(identity)
        value = [None] * size
        parent[key] = value
        return [self.allocate(value, child, [*path, child]) for child in range(size)]

    def result(self):
        if self.pending:
            raise ValueError('The output still contains unresolved positions.')
        return deepcopy(self.output[self.settings['root_key']])


def fill_slots(query, state, slots, *, service, task_id, prompts, budget):
    """Return semantic slot values; the caller owns keys, containers and serialization."""
    finite = [slot for slot in slots if slot['options']]
    opened = [slot for slot in slots if not slot['options']]
    values = {}
    if finite:
        requests = [{'id': slot['id'], 'context': query, 'state': state,
                     'question': slot['question'], 'options': slot['options']} for slot in finite]
        choices = service.decide(requests, task_id=task_id)
        domains = {slot['id']: {option['id']: option['value'] for option in slot['options']}
                   for slot in finite}
        if len(choices) != len(domains) or {choice['id'] for choice in choices} != domains.keys():
            raise ValueError('Finite slot outputs must match submitted identities.')
        values.update({choice['id']: domains[choice['id']][choice['choice']] for choice in choices})
    if opened:
        messages = [[{'role': 'system', 'content': prompts['system']},
                     {'role': 'user', 'content': prompts['input'].format(
                         query=query, state=state, question=slot['question'])}] for slot in opened]
        outputs = service.complete_batch(messages, budget['max_new_tokens'], budget['temperature'],
                                         budget['stop'], task_id=task_id)
        values.update({slot['id']: output for slot, output in zip(opened, outputs, strict=True)})
    return values


def assemble_slots(query, state, known_values, *, service, task_id, settings, prompts, budget, trace, schema):
    output = OutputSlots(settings)
    schemas = {output.root: schema}
    readout_budget = {**budget, 'stop': settings['stop']}
    kinds = [option for option in settings['kind_options']
             if option['id'] != settings['reference_kind'] or known_values]
    for round_index in range(budget['max_turns']):
        if not output.pending:
            return output.result()
        frontier = output.ready(schemas)
        context = prompts['state'].format(state=state, output=json.dumps(output.output))
        requests = [{'id': slot['id'], 'question': prompts['typed_value'].format(
                         request=prompts['kind'].format(path=json.dumps(slot['path'])),
                         description=schemas[slot['id']].get('description', '')),
                     'options': kinds} for slot in frontier if 'type' not in schemas[slot['id']]]
        types = {slot['id']: schemas[slot['id']]['type'] for slot in frontier
                 if 'type' in schemas[slot['id']]}
        types.update(fill_slots(query, context, requests, service=service, task_id=task_id,
                                prompts=prompts, budget=readout_budget))
        payloads = []
        resolved = {}
        references = [{'id': key, 'description': json.dumps(value), 'value': value}
                      for key, value in known_values.items()]
        for slot in frontier:
            identity, kind = slot['id'], types[slot['id']]
            definition = schemas[identity]
            if kind == settings['object_kind'] and 'properties' in definition:
                resolved[identity] = list(definition['properties'])
            elif kind in settings['literals']:
                resolved[identity] = settings['literals'][kind]
            elif kind == settings['reference_kind'] and len(references) == 1:
                reference, = references
                resolved[identity] = reference['value']
            else:
                payloads.append({'id': identity,
                    'question': prompts['typed_value'].format(
                        request=prompts[kind].format(path=json.dumps(slot['path'])),
                        description=definition.get('description', '')),
                    'options': references if kind == settings['reference_kind'] else []})
        resolved.update(fill_slots(query, context, payloads, service=service, task_id=task_id,
                                   prompts=prompts, budget=readout_budget))
        trace.append({'round': round_index, 'frontier': frontier, 'types': types, 'values': deepcopy(resolved)})
        for identity, value in resolved.items():
            kind = types[identity]
            if kind == settings['object_kind']:
                definition = schemas[identity]
                keys = list(definition['properties']) if 'properties' in definition else value.splitlines()
                children = output.object(identity, keys)
                schemas.update({child: definition.get('properties', {}).get(key, {})
                                for child, key in zip(children, keys, strict=True)})
            elif kind == settings['array_kind']:
                children = output.array(identity, int(value))
                schemas.update({child: schemas[identity].get('items', {}) for child in children})
            elif kind == settings['number_kind']:
                number = json.loads(value)
                if type(number) not in (int, float):
                    raise ValueError('A numeric output slot requires a number.')
                output.assign(identity, number)
            else:
                output.assign(identity, value)
    if output.pending:
        raise TaskLimitError('The shared turn budget ended with unresolved output slots.')
    return output.result()
