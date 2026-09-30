from collections import defaultdict

from jev_spawn.service.finite import StructuredService
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.merged_tail import LayerMergedTail
from jev_spawn.infra.history_prefix import HistoryPrefixTail
from jev_spawn.runtime.prefix_cache import PrefixCache


class GraphFiniteService(StructuredService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.prefix_cache = PrefixCache(shared.runtime.root_batch_size)
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.finite_physical_batch = settings.get('finite_physical_batch')
        self.finite_tail = self._make_tail(backend, shared, settings)
        self.execution_metadata.update(finite_batch_grouping='shared task roots with dynamic state suffixes',
                                       cached_extensions='packed valid suffix recurrence with exact row boundaries',
                                       finite_decode_engine='finite_cuda_graph',
                                       finite_graph_state='separate arena from suspended text decoding')

    def _score(self, batch):
        groups = defaultdict(list)
        for request in batch:
            groups[(request.task_id, request.field['context'])].append(request)
        lengths = {key: common_prefix([request.root_tokens for request in requests])
                   for key, requests in groups.items()}
        return self._finite_score(batch, [lengths[(request.task_id, request.field['context'])]
                                         for request in batch])

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
        self.prefix_cache.clear()
        self.finite_tail.graphs.clear()
        self.finite_cache.clear()


class HistoryGraphFiniteService(GraphFiniteService):
    def _make_tail(self, backend, shared, settings):
        return HistoryPrefixTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), settings['state_copy'], settings['graph_shape'])

    def close(self):
        super().close()
        self.finite_tail.history_cache.clear()
        self.finite_tail.history_tokens.cache_clear()


class DirectHistoryFiniteService(HistoryGraphFiniteService):
    def _make_tail(self, backend, shared, settings):
        return LayerMergedTail(backend, shared.runtime,
            PrefixCache(shared.runtime.root_batch_size), settings['state_copy'], settings['graph_shape'],
            packing=settings['packing'])

    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata.update(finite_decode_engine='native_history_terminal_prefill',
            finite_graph_state='Exact-request reuse with selected-head readout from terminal prefill')

    def _execution_trace(self, result, shapes):
        return {'graph_capture_seconds': result['timings']['capture_seconds'],
            'decode_engine': self.execution_metadata['finite_decode_engine'],
            'graph_replays': result['graph_replays'], 'graph_captures': result['graph_captures']}
