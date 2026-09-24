import json

from baselines.common.graph_finite_service import StableGraphFiniteService
from baselines.common.task_prefix_service import TaskPrefixService
from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.shared_reference_state import relocate


class SharedReferenceStateService(StableGraphFiniteService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        layout = settings['shared_reference_state']
        self.reference_keys = layout['shared_keys']
        self.reference_transport = load_resource(layout['transport_resource'])
        self.reference_template = load_prompt(layout['prompt'])['encoded_state']
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)

    def decide(self, fields, *, task_id):
        relocated = []
        for field in fields:
            segments = relocate(field['input'], self.reference_transport, self.reference_keys)
            state = self.reference_template.format(**{name: json.dumps(value, **self.serialization)
                                                      for name, value in segments.items()})
            relocated.append({**field, 'state': state})
        return TaskPrefixService.decide(self, relocated, task_id=task_id)
