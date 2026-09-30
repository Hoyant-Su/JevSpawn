import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer, StaticLayer

from jev_spawn.runtime.native_restore import load_native_prefixes
from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.decoding import CapturedDecode


class CapturedFinite(CapturedDecode):
    """Evaluate the final real prompt token from immutable native prefix states."""

    @torch.inference_mode()
    def __init__(self, backend, batch_size, capacity, native_ids, graph_pool, graph_stream, state_copy_settings):
        self.backend = backend
        self.trunk = backend.model.model.language_model
        self.capacity = capacity
        self.copy_settings = state_copy_settings
        self.arena = StaticCacheArena(backend.cache_config, batch_size, capacity,
                                      backend.model.lm_head.weight.dtype, backend.device)
        self.cache = self.arena.bind(batch_size, capacity)
        self.ids = torch.empty((batch_size, 1), device=backend.device, dtype=torch.long)
        self.positions = torch.empty_like(self.ids)
        self.key_valid = torch.empty((batch_size, capacity), device=backend.device, dtype=torch.bool)
        self.key_positions = torch.arange(capacity, device=backend.device)
        self.native_ids = tuple(native_ids)
        self.selected_weights = backend.selected_output_weights(self.native_ids).float().contiguous()
        self.logits = torch.empty((batch_size, len(self.native_ids)), device=backend.device, dtype=torch.float32)
        self.graph = None
        self.graph_pool, self.graph_stream = graph_pool, graph_stream
        self.configure_attention()

    @torch.inference_mode()
    def load(self, states, last_tokens):
        return load_native_prefixes(self, states, last_tokens, self.copy_settings)

    @torch.inference_mode()
    def step(self):
        valid = self.key_valid & (self.key_positions <= self.cache.get_seq_length())
        output = self.trunk(input_ids=self.ids, position_ids=self.positions,
                            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
                            past_key_values=self.cache, use_cache=True, **self.attention_arguments())
        self.logits.copy_(F.linear(output.last_hidden_state[:, -1].float(), self.selected_weights))
