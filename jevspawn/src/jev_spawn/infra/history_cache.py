from collections import OrderedDict
from functools import lru_cache
import time

from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.cached_suffix import ragged_suffix
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.schema import controller_prefix


class HistoryCache:
    def __init__(self, backend, capacity):
        self.backend, self.capacity = backend, capacity
        self.entries = OrderedDict()

    def get_many(self, prefixes, prefill):
        roots = [self.root_by_prefix[tuple(prefix)] for prefix in prefixes]
        states, root_hits = self.root_cache.get_many(roots, prefill)
        parents, tails, reused = [], [], []
        for prefix, root, state in zip(prefixes, roots, states, strict=True):
            key = tuple(root)
            prior = self.entries.get(key)
            if prior is not None and tuple(prefix[:len(prior[0])]) == prior[0]:
                tokens, state = prior
            else:
                tokens = key
            parents.append(state)
            tails.append(prefix[len(tokens):])
            reused.append(len(tokens) - len(root))
        states = ragged_suffix(self.backend, parents, tails, self.work)
        for root, prefix, state in zip(roots, prefixes, states, strict=True):
            key = tuple(root)
            self.entries[key] = (tuple(prefix), state)
            self.entries.move_to_end(key)
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.reused_history_tokens = sum(reused)
        self.history_hits = [value > 0 for value in reused]
        self.reused_root_tokens = sum(len(root) for root, hit in zip(roots, root_hits, strict=True) if hit)
        return states, [not tail for tail in tails]


class HistoryTail(StableFiniteGraphTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings, history_settings):
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings)
        self.history = HistoryCache(backend, runtime.root_batch_size)
        self.history_settings = history_settings
        self.history_tokens = lru_cache(maxsize=runtime.root_batch_size)(self.render_history)

    def render_history(self, system, context, history):
        rendered, = self.backend._render([controller_prefix(context, history)], system)
        return tuple(self.backend.tokenizer(rendered, add_special_tokens=False)['input_ids'])

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        started = time.perf_counter()
        roots, checkpoints, lengths = [], [], []
        block = self.history_settings['chunk_tokens']
        for request, base in zip(requests, base_lengths, strict=True):
            system, = [message['content'] for message in request.messages if message['role'] == 'system']
            tokens = request.admitted.tokens
            boundary = common_prefix([tokens, self.history_tokens(
                system, request.field['context'], request.field['history'])])
            assert boundary >= base
            length = base + (boundary - base) // block * block
            roots.append(tuple(tokens[:base]))
            checkpoints.append(tuple(tokens[:length]))
            lengths.append(length)
        self.history.root_by_prefix = dict(zip(checkpoints, roots, strict=True))
        self.history.root_cache = base_cache
        self.history.work = {'computed_input_tokens': self.history_settings['initial_count'],
                             'padded_input_tokens': self.history_settings['initial_count']}
        result = super().score(requests, lengths, self.history, physical_batch_size)
        for name in ('computed_input_tokens', 'padded_input_tokens'):
            result[name] += self.history.work[name]
        result.update(reused_state_tokens=self.history.reused_history_tokens,
                      reused_root_tokens=self.history.reused_root_tokens,
                      history_prefix_hits=self.history.history_hits,
                      history_checkpoint_tokens=lengths,
                      persistent_prefix_scope='task_root_and_native_history',
                      root_prefix_tokens=[len(root) for root in dict.fromkeys(roots)],
                      elapsed_seconds=time.perf_counter() - started)
        return result
