from collections import OrderedDict
from functools import lru_cache

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.schema import controller_prefix


class HistoryPrefixTail(StableFiniteGraphTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings):
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings)
        self.history_cache = OrderedDict()
        self.history_tokens = lru_cache(maxsize=runtime.root_batch_size)(self._history_tokens)

    def _history_tokens(self, system, context, history):
        prefix = controller_prefix(context, history).rstrip()
        rendered, = self.backend._render([prefix], system)
        boundary = rendered.index(prefix) + len(prefix)
        encoded = self.backend.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        # Exclude the boundary token if it also contains the mutable trailing whitespace.
        stop = max(index + 1 for index, (_, end) in enumerate(encoded['offset_mapping'])
                   if 0 < end <= boundary)
        return tuple(encoded['input_ids'][:stop])

    def extend_prefixes(self, backend, states, prefixes, bases, work, extend_states):
        parents, tails, boundaries = [], [], []
        for key, request, state, prefix, base in zip(
                self.owners, self.owners.values(), states, prefixes, bases, strict=True):
            tokens = self.history_tokens(request.messages[0]['content'],
                                         request.field['context'], request.field['history'])
            length = max(len(base), common_prefix([prefix, tokens]))
            boundary = tuple(prefix[:length])
            sources = [(tuple(base), state), *self.history_cache.get(key, ())]
            old_tokens, parent = max(
                (source for source in sources if boundary[:len(source[0])] == source[0]),
                key=lambda source: len(source[0]))
            parents.append(parent)
            tails.append(list(boundary[len(old_tokens):]))
            boundaries.append(boundary)
            work['reused_state_tokens'] += len(old_tokens) - len(base)
        histories = extend_states(backend, parents, tails, work)
        for key, tokens, state in zip(self.owners, boundaries, histories, strict=True):
            self.history_cache[key] = [(tokens, state)]
            self.history_cache.move_to_end(key)
            if len(self.history_cache) > self.runtime.root_batch_size:
                self.history_cache.popitem(last=False)
        return extend_states(backend, histories,
            [prefix[len(boundary):] for prefix, boundary in zip(prefixes, boundaries, strict=True)], work)

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        self.owners = {}
        for request in requests:
            key = request.task_id, request.field['context']
            self.owners.setdefault(key, request)
            assert self.owners[key].field['history'] == request.field['history']
        result = super().score(requests, base_lengths, base_cache, physical_batch_size)
        self.prefix_cache.clear()
        result['persistent_prefix_scope'] = 'task_root_and_native_history'
        return result
