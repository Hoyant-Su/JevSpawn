import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from jev_spawn.runtime.rolling_decode import RollingDecode


class PaddedRollingDecode(RollingDecode):
    def configure_rows(self, live_count):
        self.active_count = torch.tensor(live_count, device=self.ids.device, dtype=torch.long)
        self.row_indices = torch.arange(len(self.ids), device=self.ids.device)

    def step(self):
        inactive = self.row_indices >= self.active_count
        self.ids.masked_fill_(inactive[:, None], self.backend.tokenizer.pad_token_id)
        self.positions.masked_fill_(inactive[:, None], 0)
        self.key_valid.masked_fill_(inactive[:, None], True)
        for layer in self.cache.layers:
            if not isinstance(layer, LinearAttentionCacheLayerMixin):
                layer.cumulative_length.masked_fill_(inactive, 0)
        # Padding rows have no request, history, deadline, or published output.
        super().step()
