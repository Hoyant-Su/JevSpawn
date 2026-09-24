from concurrent.futures import Future
from dataclasses import fields
from functools import partial
import json

import torch

from baselines.common.jevspawn_service import StructuredService
from jev_spawn.infra.configuration import ROOT, resolve_symbol
from jev_spawn.runtime.prefix_cache import PrefixCache


SETTINGS = json.loads((ROOT / 'configs/baselines/common/parallel_service.json').read_text())


class ParallelPrefixCache(PrefixCache):
    def __init__(self, original, commands):
        assert not original.entries
        super().__init__(original.capacity)
        self.commands = commands

    def resident_plan(self, keys):
        hits = super().resident_plan(keys)
        plan = self.commands.leader_value({'hits': hits, 'resident': len(self.entries)}
                                          if self.commands.is_leader else None)
        assert hits == plan['hits'] and len(self.entries) == plan['resident']
        return plan['hits']

    def get(self, sequences, compute):
        key = tuple(tuple(sequence) for sequence in sequences)
        hit = key in self.entries
        decision = self.commands.leader_value(
            {'hit': hit, 'evict': not hit and len(self.entries) == self.capacity,
             'resident': len(self.entries)} if self.commands.is_leader else None)
        assert hit == decision['hit'] and len(self.entries) == decision['resident']
        if not decision['hit']:
            if decision['evict']:
                self.entries.popitem(last=False)
            self.entries[key] = compute()
        self.entries.move_to_end(key)
        return self.entries[key], decision['hit']


class ParallelStops:
    def __init__(self, stopping, commands):
        self.stopping, self.commands = stopping, commands

    def __getattr__(self, name):
        return getattr(self.stopping, name)

    def __call__(self, input_ids, scores, **kwargs):
        done = (self.stopping(input_ids, scores, **kwargs) if self.commands.is_leader
                else torch.empty_like(self.stopping.lengths, dtype=torch.bool))
        state = torch.stack((done.to(self.stopping.lengths.dtype),
                             self.stopping.lengths, self.stopping.causes))
        self.commands.broadcast_tensor(state)
        done, lengths, causes = state.unbind()
        self.stopping.lengths.copy_(lengths)
        self.stopping.causes.copy_(causes)
        return done.bool()


def encode_request(request):
    symbol = type(request).__module__ + '.' + type(request).__qualname__
    assert symbol in SETTINGS['request_types']
    return {'type': symbol, 'values': {field.name: getattr(request, field.name)
            for field in fields(request) if field.name not in SETTINGS['excluded_request_fields']}}


def decode_request(record):
    assert record['type'] in SETTINGS['request_types']
    constructor = resolve_symbol(record['type'])
    initial = {field.name: record['values'][field.name] for field in fields(constructor)
               if field.init and field.name not in SETTINGS['excluded_request_fields']}
    request = constructor(**initial, future=Future())
    for field in fields(constructor):
        if not field.init:
            setattr(request, field.name, record['values'][field.name])
    return request


class ParallelService:
    def __init__(self, backend, shared, deadlines, **kwargs):
        self.parallel_commands = backend.parallel_commands
        self.parallel_batches = []
        self.interleaving_finite = False
        super().__init__(backend, shared, deadlines, **kwargs)
        for name in SETTINGS['prefix_caches'][self.parallel_source_service]:
            setattr(self, name, ParallelPrefixCache(getattr(self, name), self.parallel_commands))
        self.parallel_commands.register(SETTINGS['generate_command'], self._parallel_generate)
        self.execution_metadata.update(host_scheduling='leader_rank_only',
            numerical_execution='tensor_parallel_cohorts', world_size=shared.runtime.world_size,
            stop_authority='leader_rank', nested_finite_commands=isinstance(self, StructuredService))

    def close(self):
        super().close()
        self.decoders.clear()
        self.cache_arena = None

    def _serve(self):
        if self.parallel_commands.is_leader:
            return super()._serve()

    def _generate(self, batch):
        assert self.parallel_commands.is_leader
        payload = {'requests': [encode_request(request) for request in batch],
                   'started': {request.task_id: self.deadlines.started[request.task_id] for request in batch},
                   'interleaving_finite': self.interleaving_finite}
        self.parallel_batches.append(batch)
        try:
            return self.parallel_commands.call(SETTINGS['generate_command'], payload)
        finally:
            self.parallel_batches.pop()

    def _parallel_generate(self, payload):
        self.deadlines.started.update(payload['started'])
        batch = (self.parallel_batches[-1] if self.parallel_commands.is_leader
                 else [decode_request(record) for record in payload['requests']])
        interleaving_finite = self.interleaving_finite
        self.interleaving_finite = payload['interleaving_finite']
        try:
            return super()._generate(batch)
        except TimeoutError:
            if self.parallel_commands.is_leader:
                raise
        finally:
            self.interleaving_finite = interleaving_finite

    def _stopping(self, batch, prompt_width):
        stopping = ParallelStops(super()._stopping(batch, prompt_width), self.parallel_commands)
        self.stopping = stopping
        return stopping

    def _between_decode_steps(self):
        callback = super()._between_decode_steps
        if isinstance(self, StructuredService) and self.finite_scheduling == 'decode_step':
            return self.parallel_commands.window(callback)
        return callback()

    def _deliver(self, tokens, indices):
        if self.parallel_commands.is_leader:
            return super()._deliver(tokens, indices)


def parallel_factory(factory):
    constructor = factory.func if isinstance(factory, partial) else factory
    source = constructor.__module__ + '.' + constructor.__qualname__
    if source == SETTINGS['refill']['source']:
        service = resolve_symbol(SETTINGS['refill']['implementation'])
        return partial(service, *factory.args, **factory.keywords) if isinstance(factory, partial) else service
    assert source in SETTINGS['prefix_caches']
    service = type('Parallel' + constructor.__name__, (ParallelService, constructor),
                   {'parallel_source_service': source})
    return partial(service, *factory.args, **factory.keywords) if isinstance(factory, partial) else service
