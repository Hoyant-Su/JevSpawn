import json

from baselines.common.literal_scored_state_service import LiteralScoredStateService
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.runtime.action_catalog import encode_catalog


class CompactActionService(LiteralScoredStateService):
    def __init__(self, backend, shared, deadlines, *, settings, prompts):
        self.catalog_settings = settings['action_catalog']
        self.catalog_prompts = load_prompt(settings['action_catalog_prompts'])
        super().__init__(backend, shared, deadlines, settings=settings, prompts=prompts)

    def prepare_state_fields(self, fields):
        prepared = super().prepare_state_fields(fields)
        for field in prepared:
            root_key = self.literal_transport['root_key']
            semantic = field['input'][root_key]
            if any(self.catalog_settings['metadata_key'] in candidate for candidate in semantic['candidates']):
                field['input'] = {root_key: {**semantic, 'candidates': encode_catalog(
                    semantic['candidates'], self.catalog_settings)}}
                field['state'] = self.catalog_prompts['encoded_state'].format(
                    value=json.dumps(field['input'], **self.serialization))
        return prepared
