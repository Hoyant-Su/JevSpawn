from baselines.common.service import BatchService
from jev_spawn.infra.history_prefill import HistoryDecode, HistoryPrefill


class HistoryBatchService(BatchService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.history = HistoryPrefill(backend, settings['history_cache'])
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata['history_cache'] = settings['history_cache']

    def make_decoder(self, size, capacity):
        decoder = HistoryDecode(self.backend, size, capacity, arena=self.cache_arena,
                                graph_pool=self.graph_pool, graph_stream=self.graph_stream)
        decoder.history = self.history
        return decoder

    def validate_forward_batch(self, size, requests):
        assert self.history.settings['minimum_rows'] <= size <= len(requests)

    def _generate_tokens(self, inputs, options, stopping):
        self.history.sequences = [request.input_ids for request in self.current_batch]
        output = super()._generate_tokens(inputs, options, stopping)
        self.graph_stats['history_prefill'] = self.history.records[-self.history.settings['record_offset']]
        return output

    def close(self):
        super().close()
        self.history.cache.clear()
