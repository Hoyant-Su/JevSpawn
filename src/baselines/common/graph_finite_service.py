from baselines.common.mixed_prefix_service import MixedPrefixService
from jev_spawn.infra.finite_graph import RaggedFiniteGraphTail
from jev_spawn.infra.history_prefix import HistoryPrefixTail
from jev_spawn.infra.stable_finite_graph import StableFiniteGraphTail
from jev_spawn.runtime.prefix_cache import PrefixCache


class GraphFiniteService(MixedPrefixService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.finite_physical_batch = settings.get('finite_physical_batch')
        self.finite_tail = self._make_tail(backend, shared, settings)
        self.execution_metadata.update(cached_extensions='packed valid suffix recurrence with exact row boundaries',
                                       finite_decode_engine='finite_cuda_graph',
                                       finite_graph_state='separate arena from suspended text decoding')

    def _make_tail(self, backend, shared, settings):
        return RaggedFiniteGraphTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), settings['state_copy'])

    @property
    def finite_cache(self):
        return self.finite_tail.prefix_cache

    @finite_cache.setter
    def finite_cache(self, value):
        self.finite_tail.prefix_cache = value

    def _finite_score(self, batch, lengths):
        return self.finite_tail.score(batch, lengths, self.prefix_cache,
                                      self.finite_physical_batch)

    def _execution_trace(self, result, shapes):
        assert result['graph_replays'] > 0
        return {'graph_capture_seconds': result['timings']['capture_seconds'],
                'decode_engine': self.execution_metadata['finite_decode_engine'],
                'graph_replays': result['graph_replays'],
                'graph_captures': result['graph_captures'],
                'graph_input_shape': list(self.finite_tail.graphs[next(reversed(self.finite_tail.graphs))].ids.shape)}

    def close(self):
        super().close()
        self.finite_tail.graphs.clear()
        self.finite_cache.clear()


class StableGraphFiniteService(GraphFiniteService):
    def _make_tail(self, backend, shared, settings):
        return StableFiniteGraphTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), settings['state_copy'], settings['graph_shape'])


class HistoryGraphFiniteService(GraphFiniteService):
    def _make_tail(self, backend, shared, settings):
        return HistoryPrefixTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), settings['state_copy'], settings['graph_shape'])

    def close(self):
        super().close()
        self.finite_tail.history_cache.clear()
        self.finite_tail.history_tokens.cache_clear()
