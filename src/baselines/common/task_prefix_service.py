from baselines.common.jevspawn_service import DecisionRequest, StructuredService
from jev_spawn.runtime.prefix_cache import PrefixCache
from methods.program_execution.grouped import score_grouped
from jev_spawn.algo.structured import common_prefix


class TaskDecisionRequest(DecisionRequest):
    @property
    def signature(self):
        return 'finite', self.task_id, self.field['context']


class TaskPrefixService(StructuredService):
    request_type = TaskDecisionRequest

    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.prefix_cache = PrefixCache(shared.runtime.root_batch_size)
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        assert self.field_mode == 'tiled_shared'
        self.execution_metadata.update(finite_context=settings['finite_context'],
                                       finite_batch_grouping='task_id and exact declared context',
                                       prefix_cache_capacity=shared.runtime.root_batch_size,
                                       planner_cache_reused=False)

    def decide(self, fields, *, task_id):
        return super().decide(self.prepare_task_fields(fields), task_id=task_id)

    def prepare_task_fields(self, fields):
        return fields

    def task_prefix_length(self, batch):
        assert len({request.task_id for request in batch}) == 1
        assert len({request.field['context'] for request in batch}) == 1
        return common_prefix([request.root_tokens for request in batch])

    def _score(self, batch):
        length = self.task_prefix_length(batch)
        fields = [{**request.field, 'id': str(index)} for index, request in enumerate(batch)]
        return score_grouped(self.backend, [fields], self.field_mode,
                             prefix_cache=self.prefix_cache, prefix_lengths=[length],
                             admitted_prompts=[request.admitted for request in batch])

    def close(self):
        super().close()
        self.prefix_cache.clear()
