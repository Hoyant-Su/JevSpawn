import json

import torch

from jev_spawn.service.graph import DirectHistoryFiniteService
from jev_spawn.service.finite import DecisionRequest
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.values import ValueTail
from jev_spawn.runtime.prefix_cache import PrefixCache
from jev_spawn.schema import controller_prefix


class ValueRequest(DecisionRequest):
    @property
    def signature(self):
        return 'values'


def messages(field, prompts, serialization):
    values = sorted({json.dumps(option['values'], **serialization) for option in field['options']})
    return [{'role': 'system', 'content': prompts['system']},
        {'role': 'user', 'content': prompts['user'].format(**field,
            prefix=controller_prefix(field['context'], field['history']),
            candidates='\n'.join(values), targets=json.dumps(field['value_fields'], **serialization))}]


class ValueService(DirectHistoryFiniteService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.value_settings = settings['values']
        self.value_tail = ValueTail(backend, shared.runtime, PrefixCache(shared.runtime.root_batch_size),
            settings['state_copy'], settings['graph_shape'], packing=settings['packing'], values=settings['values'])

    def decide(self, fields, *, task_id):
        requests = [{**field, 'readout_prompt': self.value_settings['prompts'],
                     'value_fields': [{'id': field['id'], 'question': field['question']}]}
                    for field in fields]
        return self.enqueue_decisions(requests, task_id=task_id, request_type=ValueRequest)

    def enqueue_decisions(self, fields, *, task_id, request_type):
        ordered = [{**field, 'options': sorted(field['options'], key=lambda option: (
            json.dumps(option['values'], **self.value_settings['serialization']), option['id']))}
            for field in fields]
        outputs = super().enqueue_decisions(ordered, task_id=task_id, request_type=request_type)
        restored = []
        for field, output in zip(fields, outputs, strict=True):
            positions = {identity: index for index, identity in enumerate(output['option_ids'])}
            order = [positions[option['id']] for option in field['options']]
            restored.append({**output, 'options': field['options'],
                'option_ids': [option['id'] for option in field['options']],
                'probabilities': [output['probabilities'][index] for index in order],
                'option_logits': [output['option_logits'][index] for index in order]})
            if len(order) > 1:
                key = task_id, field['id']
                distribution = self.field_distributions[key]
                self.field_distributions[key] = distribution.index_select(0,
                    torch.tensor(order, device=distribution.device, dtype=torch.long))
        return restored

    @property
    def value_cache(self):
        return self.value_tail.prefix_cache

    @value_cache.setter
    def value_cache(self, value):
        self.value_tail.prefix_cache = value

    def _validate_inputs(self, batch):
        for request in batch:
            if isinstance(request, ValueRequest):
                request.messages = self.contract_messages(messages(request.field,
                    load_prompt(request.field['readout_prompt']),
                    self.value_settings['serialization']))
        return super()._validate_inputs(batch)

    def _finite_score(self, batch, lengths):
        if isinstance(batch[0], ValueRequest):
            return self.value_tail.score(batch, lengths, self.prefix_cache, self.finite_physical_batch)
        return super()._finite_score(batch, lengths)

    def _execution_trace(self, result, shapes):
        trace = super()._execution_trace(result, shapes)
        if 'values' in result:
            trace['decode_engine'] = 'native_value_forks'
        return trace

    def close(self):
        super().close()
        self.value_tail.history_cache.clear()
        self.value_tail.history_tokens.cache_clear()
        self.value_cache.clear()
