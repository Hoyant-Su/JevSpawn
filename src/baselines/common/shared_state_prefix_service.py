import json

from baselines.common.task_prefix_service import TaskPrefixService
from jev_spawn.infra.configuration import load_resource
from jev_spawn.runtime.prefix_cache import PrefixCache
from methods.program_execution.grouped import score_grouped


class SharedStatePrefixService(TaskPrefixService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.state_cache = PrefixCache(shared.runtime.root_batch_size)
        self.serialization = load_resource('finite_control')['serialization']
        self.shared_fields = settings['shared_state_fields']
        self.state_template = prompts['encoded_state']
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata.update(prefix_reuse='immutable_task_and_exact_execution_state',
                                       state_cache_capacity=shared.runtime.root_batch_size,
                                       shared_state_fields=self.shared_fields)

    def decide(self, fields, *, task_id):
        return super().decide(self.prepare_state_fields(fields), task_id=task_id)

    def prepare_state_fields(self, fields):
        ordered = []
        for field in fields:
            payload = {key: field['input'][key] for key in self.shared_fields if key in field['input']}
            payload.update(field['input'])
            ordered.append({**field, 'state': self.state_template.format(
                value=json.dumps(payload, **self.serialization))})
        return ordered

    def _score(self, batch):
        length = self.task_prefix_length(batch)
        fields = [{**request.field, 'id': str(index)} for index, request in enumerate(batch)]
        return score_grouped(self.backend, [fields], self.field_mode, prefix_cache=self.state_cache,
                             base_prefix_cache=self.prefix_cache, base_prefix_length=length,
                             admitted_prompts=[request.admitted for request in batch])

    def close(self):
        super().close()
        self.state_cache.clear()
