from copy import copy, deepcopy
from dataclasses import dataclass
from importlib import import_module
import json

from baselines.common.jevspawn_service import DecisionRequest
from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.parallel_service import SETTINGS as PARALLEL_SETTINGS
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.query_execution import QueryExecution
from jev_spawn.runtime import query_execution, branches
from jev_spawn.rollout import branching
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import CONTROLLER, controller_prompts
from tests.infra.action_prefix.messages import ActionMessages


class ActionExecution(QueryExecution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_messages = None

    def fork(self):
        child = super().fork()
        child.action_messages = None if self.action_messages is None else self.action_messages.fork()
        return child

    def prepare(self, fields):
        field, = fields
        if self.action_messages is None:
            domains, requests, evidence = super().prepare(fields)
            request, = requests
            user, = controller_prompts([request['state']], request['question'], request['options'],
                self.service.backend.answer_labels, CONTROLLER['output_instruction'],
                contexts=[self.query], histories=[self.history])
            messages = self.service.contract_messages([
                {'role': 'system', 'content': CONTROLLER['system']}, {'role': 'user', 'content': user}])
            self.action_messages = ActionMessages(messages, field, self.bound_fields,
                self.service.backend.answer_labels, load_prompt(self.action_prefix_settings['prompts']),
                CONTROLLER, self.settings['serialization'])
        else:
            domains = {field['id']: {self.settings['candidate_id'].format(index=index): deepcopy(value)
                                     for index, value in enumerate(field['values'])}}
            options = [{'id': identity, 'description': json.dumps(value, **self.settings['serialization'])}
                       for identity, value in domains[field['id']].items()]
            messages = self.action_messages.append(field, self.bound_fields, self.fields, options)
            evidence = [event['id'] for event in self.history_events]
            requests = [{'id': field['id'], 'context': self.query, 'history': self.history, 'state': '',
                         'question': self.prompts['select_field'].format(identity=field['id']), 'options': options}]
        requests[0]['action_messages'] = messages
        return domains, requests, evidence


@dataclass
class ActionRequest(DecisionRequest):
    def __post_init__(self):
        if 'action_messages' in self.field:
            self.messages = self.field['action_messages']


class ResidentBases:
    def __init__(self, roots, residents):
        self.roots, self.residents = roots, residents

    def get_many(self, sequences, compute):
        keys = list(map(tuple, sequences))
        missing = [key for key in dict.fromkeys(keys) if key not in self.residents]
        states, hits = self.roots.get_many([list(key) for key in missing], compute)
        values = {**self.residents, **dict(zip(missing, states, strict=True))}
        hit_by_key = {**{key: True for key in self.residents}, **dict(zip(missing, hits, strict=True))}
        return [values[key] for key in keys], [hit_by_key[key] for key in keys]


class ActionPrefixTail(StableFiniteGraphTail):
    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        grouped, lengths, residents, reused = [], [], {}, []
        for request, base in zip(requests, base_lengths, strict=True):
            parent = self.prefix_cache.longest_parent(request.admitted.tokens[:-1], base)
            use = 'action_messages' in request.field and parent is not None
            length = len(parent[0]) if use else base
            if use:
                residents[tuple(parent[0])] = parent[1]
            row = copy(request)
            row.task_id = (request.task_id, tuple(request.admitted.tokens[:length]))
            grouped.append(row)
            lengths.append(length)
            reused.append(length - base)
        result = super().score(grouped, lengths, ResidentBases(base_cache, residents), physical_batch_size)
        result.update(action_prefix_reused_tokens=sum(reused), action_prefix_reused_by_row=reused,
                      action_prefix_hit_rows=sum(value > 0 for value in reused))
        return result


def install(settings):
    PARALLEL_SETTINGS['request_types'].extend(settings['request_types'])
    ActionExecution.action_prefix_settings = settings
    for module in (query_execution, branches, branching):
        module.QueryExecution = ActionExecution
    StableGraphFiniteService.request_type = ActionRequest
    definition = settings['finite_tail']
    tail_type = getattr(import_module(definition['module']), definition['class'])

    def make_tail(service, backend, shared, inference):
        return tail_type(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
                         inference['state_copy'], inference['graph_shape'])

    StableGraphFiniteService._make_tail = make_tail
