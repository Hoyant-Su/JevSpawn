from unittest.mock import patch

from jev_spawn.infra import finite_batch
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache


class SharedPrefixTail(StableFiniteGraphTail):
    def __init__(self, backend, runtime, prefix_cache, state_copy_settings, settings):
        super().__init__(backend, runtime, prefix_cache, state_copy_settings, settings)
        self.shared_cache = PrefixCache(runtime.root_batch_size)

    def extend_prefixes(self, backend, states, prefixes, bases, work, extend_states):
        def compute():
            return extend_states(backend, states,
                [prefix[len(base):] for prefix, base in zip(prefixes, bases, strict=True)], work)

        # Preserve the entire prefill cohort and its original computation boundary.
        values, hit = self.shared_cache.get([*bases, *prefixes], compute)
        work['reused_state_tokens'] += hit * sum(len(prefix) - len(base)
            for prefix, base in zip(prefixes, bases, strict=True))
        return values

    def score(self, requests, base_lengths, base_cache, physical_batch_size=None):
        with patch.object(finite_batch, 'extend_prefixes', self.extend_prefixes):
            return super().score(requests, base_lengths, base_cache, physical_batch_size)
