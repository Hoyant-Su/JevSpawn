import json

from baselines.common.recorded_finite_service import RecordedStableGraphFiniteService
from jev_spawn.infra.configuration import load_resource
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.reference_transport import unpack_references


class LiteralStateService(RecordedStableGraphFiniteService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        layout = settings['communication_layout']
        self.literal_prompts = load_prompt(layout['prompts'])
        self.literal_transport = load_resource(layout['transport'])
        self.literal_shared_keys = layout['shared_keys']
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)
        self.execution_metadata.update(communication_layout=layout,
                                       communication='literal state with shared evidence before local fields')

    def prepare_state_fields(self, fields):
        ordered = []
        for field in fields:
            semantic = unpack_references(field['input'], self.literal_transport)
            common = {key: semantic['input'][key] for key in self.literal_shared_keys if key in semantic['input']}
            common.update(semantic['input'])
            literal = {self.literal_transport['root_key']: {**semantic, 'input': common}}
            options = [{'id': option['id'], 'description': self.literal_prompts['reference_option'].format(index=index)}
                       for index, option in enumerate(field['options'])]
            state = self.literal_prompts['encoded_state'].format(value=json.dumps(literal, **self.serialization))
            ordered.append({**field, 'input': literal, 'state': state, 'options': options})
        return ordered
