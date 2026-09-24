from unittest.mock import patch

from tests.infra.history_cache.checkpoints import CheckpointHistory, ColdCheckpoint, FLA_CHUNK
from tests.infra.latent_history.candidate import RoleZeroHistory


class CheckpointRoleZeroHistory(RoleZeroHistory):
    def __init__(self, backend, settings):
        super().__init__(backend, settings)
        self.history = CheckpointHistory(backend, settings['history'])

    def cold_prefill(self, transport, ids, mask):
        recorder = ColdCheckpoint(self.backend, {'input_ids': ids, 'attention_mask': mask},
                                  self.settings['history']['checkpoint'])
        bindings = [(layer.linear_attn, layer.linear_attn.forward) for layer in transport.trunk.layers
                    if hasattr(layer, 'linear_attn')]
        with patch.object(FLA_CHUNK, 'chunk_gated_delta_rule_fwd_h', recorder.capture_native):
            for module, original in bindings:
                module.forward = recorder.patched_forward.__get__(module, type(module))
            output = super().cold_prefill(transport, ids, mask)
            for module, original in bindings:
                module.forward = original
        recorder.save(transport.decoder, self.history)
        return output
