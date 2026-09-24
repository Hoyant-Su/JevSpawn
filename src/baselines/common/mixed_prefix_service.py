from collections import defaultdict

from baselines.common.jevspawn_service import DecisionRequest, StructuredService
from jev_spawn.algo.structured import common_prefix
from jev_spawn.infra.finite_batch import score_finite_batch
from jev_spawn.runtime.prefix_cache import PrefixCache


class MixedPrefixService(StructuredService):
    request_type = DecisionRequest

    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.prefix_cache = PrefixCache(shared.runtime.root_batch_size)
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata.update(finite_batch_grouping='shared task roots with dynamic state suffixes',
                                       cached_extensions='equal-length cohorts without recurrent padding')

    def task_prefix_length(self, batch):
        return common_prefix([request.root_tokens for request in batch])

    def _score(self, batch):
        groups = defaultdict(list)
        for request in batch:
            groups[(request.task_id, request.field['context'])].append(request)
        by_context = {key: self.task_prefix_length(requests) for key, requests in groups.items()}
        lengths = [by_context[(request.task_id, request.field['context'])] for request in batch]
        return self._finite_score(batch, lengths)

    def _finite_score(self, batch, lengths):
        return score_finite_batch(self.backend, batch, lengths, self.prefix_cache)

    def close(self):
        super().close()
        self.prefix_cache.clear()
