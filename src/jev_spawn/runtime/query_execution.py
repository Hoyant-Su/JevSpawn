from copy import deepcopy
from functools import cache, reduce
import json
from operator import getitem
from string import Template
import time

from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.configuration import resolve_symbol
from jev_spawn.runtime.state import extend_event_history, history_state


@cache
def compiled_string(value):
    template = Template(value)
    names = template.get_identifiers()
    return template, names, bool(names and template.pattern.fullmatch(value))


class QueryExecution:
    """Execute a model-proposed batch against the query and completed fields."""

    def __init__(self, query, service, task_id, settings):
        self.query, self.service, self.task_id = query, service, task_id
        self.settings = settings
        self.prompts = load_prompt(settings['prompts'])
        self.field_readout = resolve_symbol(settings['field_readout']['implementation'])
        self.values, self.fields, self.trace = {}, {}, []
        self.tool_values, self.tool_observations = {}, []
        self.latest_feedback = []
        self.field_evidence = {}
        self.action_template = {}
        self.bound_fields = {}
        self.declaration_feedback = []
        self.history = ''
        self.history_events = ()
        self.active_declaration = None

    def computed_fields(self):
        return [{'id': identity, 'question': self.fields[identity]['question'],
                 'domain': deepcopy(self.fields[identity]['values']), 'value': deepcopy(value)}
                for identity, value in self.values.items()]

    def observation(self):
        return self.computed_fields() + list(self.tool_observations)

    def state(self):
        return {'execution_events': list(self.tool_observations),
                'current_path': [event['id'] for event in self.tool_observations],
                'current_feedback': [event['id'] for event in self.latest_feedback],
                'computed_fields': self.computed_fields(),
                'field_evidence': self.field_evidence,
                'bound_fields': self.bound_fields,
                'declaration_feedback': self.declaration_feedback}

    def resolved_values(self):
        return {**self.values, **self.tool_values}

    def execute(self, fields):
        identities = [field['id'] for field in fields]
        if len(set(identities)) != len(identities) or set(identities) & (self.values.keys() | self.tool_values.keys()):
            raise ValueError('Each computation must have a new, unique field identity.')
        return self.evaluate(fields)

    def fork(self):
        """Share immutable event payloads; isolate each branch's mutable state."""
        child = QueryExecution(self.query, self.service, self.task_id, self.settings)
        child.values = dict(self.values)
        child.fields = dict(self.fields)
        child.tool_values = dict(self.tool_values)
        child.tool_observations = list(self.tool_observations)
        child.latest_feedback = list(self.latest_feedback)
        child.field_evidence = dict(self.field_evidence)
        child.action_template = self.action_template
        child.bound_fields = dict(self.bound_fields)
        child.declaration_feedback = list(self.declaration_feedback)
        child.history = self.history
        child.history_events = self.history_events
        child.active_declaration = self.active_declaration
        return child

    @staticmethod
    def evaluate_branches(branches, fields):
        """Read independent branch states in one model batch, preserving local identities."""
        prepared = {identity: branch.prepare(fields[identity]) for identity, branch in branches.items()}
        requests, destinations = [], {}
        for identity, (_, local_requests, _) in prepared.items():
            for request in local_requests:
                wire_id = json.dumps([identity, request['id']])
                destinations[wire_id] = (identity, request['id'])
                requests.append({**request, 'id': wire_id})
        owners = {(id(branch.service), branch.task_id) for branch in branches.values()}
        if len(owners) != 1 or branches.keys() != fields.keys():
            raise ValueError('A branch batch must share one task and service with matching field groups.')
        source = next(iter(branches.values()))
        decisions = source.field_readout(source.service, requests, source.task_id,
                                         source.settings['field_readout'])
        if len(decisions) != len(destinations) or {item['id'] for item in decisions} != destinations.keys():
            raise ValueError('Branch outputs must match all submitted computation identities.')
        grouped = {identity: [] for identity in branches}
        for decision in decisions:
            identity, field = destinations[decision['id']]
            grouped[identity].append({**decision, 'id': field})
        return {identity: branch.commit(fields[identity], *prepared[identity], grouped[identity])
                for identity, branch in branches.items()}

    def evaluate(self, fields):
        domains, requests, evidence = self.prepare(fields)
        decisions = self.field_readout(self.service, requests, self.task_id, self.settings['field_readout'])
        return self.commit(fields, domains, requests, evidence, decisions)

    def prepare(self, fields):
        domains = {field['id']: {
            self.settings['candidate_id'].format(index=index): deepcopy(value)
            for index, value in enumerate(field['values'])} for field in fields}
        if any(not domain for domain in domains.values()):
            raise ValueError('A finite batch requires nonempty fields and candidate domains.')
        state_value = self.state()
        evidence = [event['id'] for event in self.history_events]
        state = json.dumps({**history_state(state_value), 'pending_fields': fields,
                            'current_action_observations': self.latest_feedback,
                            'action_template': self.action_template,
                            'active_declaration': self.active_declaration}, **self.settings['serialization'])
        requests = [{'id': field['id'], 'field_id': field['id'],
                     'readout_prompt': self.settings['value_prompt'],
                     'context': self.query, 'history': self.history, 'state': state,
                     'question': self.prompts['select_field'].format(identity=field['id']),
                     'value_fields': [{'id': field['id'], 'question': field['question']}],
                     'options': [{'id': identity,
                                  'description': json.dumps(value, **self.settings['serialization']), 'values': [value]}
                                 for identity, value in domains[field['id']].items()]}
                    for field in fields]
        return domains, requests, evidence

    def commit(self, fields, domains, requests, evidence, decisions):
        decisions_by_id = {decision['id']: decision for decision in decisions}
        if len(decisions_by_id) != len(decisions) or decisions_by_id.keys() != domains.keys():
            raise ValueError('Finite outputs must match the submitted field identities exactly.')
        values = {identity: domain[decisions_by_id[identity]['choice']]
                  for identity, domain in domains.items()}
        self.values.update(values)
        self.fields.update({field['id']: deepcopy(field) for field in fields})
        self.field_evidence.update({identity: list(evidence) for identity in values})
        self.trace.append({'fields': deepcopy(fields), 'requests': requests,
                           'decisions': deepcopy(decisions), 'values': deepcopy(values),
                           'based_on_feedback': evidence})
        return deepcopy(values)

    def execute_actions(self, calls, tool_batch):
        """Submit resolved actions and retain their actual feedback for the next spawn."""
        resolved = self.materialize(calls)
        identities = [call['id'] for call in resolved]
        if len(set(identities)) != len(identities) or set(identities) & (self.values.keys() | self.tool_values.keys()):
            raise ValueError('Action identities must be unique across the current execution.')
        tool_started = time.perf_counter()
        results = tool_batch(resolved)
        tool_seconds = time.perf_counter() - tool_started
        if results.keys() != set(identities):
            raise ValueError('Tool feedback must match the submitted action identities exactly.')
        parent = [item['id'] for item in self.latest_feedback]
        originals = {call['id']: call for call in calls}
        observations = [{'id': call['id'], 'source': self.settings['tool_observation_source'],
                         'parent_feedback': list(parent),
                         'argument_bindings': deepcopy(originals[call['id']]['arguments']),
                         'action': deepcopy(call), 'value': deepcopy(results[call['id']])}
                        for call in resolved]
        self.tool_values.update(deepcopy(results))
        self.tool_observations.extend(observations)
        self.history += extend_event_history(observations, self.history_events)
        self.history_events += tuple(observations)
        self.latest_feedback = list(observations)
        self.trace.append({'actions': resolved, 'observations': observations,
                           'tool_batch_seconds': tool_seconds})
        return deepcopy(results)

    def materialize(self, value):
        """Insert completed values without asking the model to regenerate them."""
        if isinstance(value, dict):
            reference = self.settings['history_reference']
            if set(value) == {reference}:
                return [deepcopy(reduce(getitem, value[reference], event['action']))
                        for event in self.tool_observations]
            join = self.settings['join_reference']
            if set(value) == {join}:
                separator, items = value[join]
                return self.materialize(separator).join(str(item) for item in self.materialize(items))
            return {key: self.materialize(child) for key, child in value.items()}
        if isinstance(value, list):
            return [self.materialize(child) for child in value]
        if isinstance(value, str):
            template, names, reference = compiled_string(value)
            values = {name: self.materialize(self.resolved_values()[name]) for name in names}
            if reference:
                name, = names
                return values[name]
            return template.substitute(values)
        return deepcopy(value)
