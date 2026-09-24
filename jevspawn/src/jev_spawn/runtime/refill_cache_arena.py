from dataclasses import dataclass

import torch
from transformers.cache_utils import LinearAttentionLayer, StaticLayer

from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.rolling_decode import RowStaticLayer


@dataclass
class CacheRows:
    cache: object
    start: int
    stop: int
    ids: torch.Tensor
    positions: torch.Tensor
    key_valid: torch.Tensor
    logits: torch.Tensor
    scalar_metadata_bytes: int


class RefillCacheArena(StaticCacheArena):
    """Fixed-stride hybrid storage with disjoint prefill and live decode rows."""

    def __init__(self, config, max_batch_size, max_cache_len, dtype, device):
        super().__init__(config, max_batch_size, max_cache_len, dtype, device)
        self.backing = super().bind(max_batch_size, max_cache_len)
        self.lengths = {index: torch.zeros(max_batch_size, dtype=torch.long, device=self.device)
                        for index, layer in enumerate(self.backing.layers) if type(layer) is StaticLayer}
        self.ids = torch.zeros((max_batch_size, 1), dtype=torch.long, device=self.device)
        self.positions = torch.zeros_like(self.ids)
        self.key_valid = torch.ones((max_batch_size, max_cache_len), dtype=torch.bool, device=self.device)
        self.logits = torch.empty((max_batch_size, config.get_text_config(decoder=True).vocab_size),
                                  dtype=dtype, device=self.device)
        self.live_count = 0
        self.pending = None
        for tensor in self.row_tensors():
            torch._dynamo.mark_static_address(tensor)

    def row_tensors(self):
        tensors = [self.ids, self.positions, self.key_valid, self.logits, *self.lengths.values()]
        for layer in self.backing.layers:
            tensors.extend([layer.keys, layer.values] if type(layer) is StaticLayer
                           else [*layer.conv_states.values(), *layer.recurrent_states.values()])
        return tensors

    @property
    def nbytes(self):
        metadata = [self.ids, self.positions, self.key_valid, self.logits, *self.lengths.values()]
        return super().nbytes + sum(t.numel() * t.element_size() for t in metadata)

    @property
    def compaction_scratch_bytes(self):
        return 0

    def _view(self, start, stop, decode):
        cache = super().bind(self.max_batch_size, self.max_cache_len)
        scalar_bytes = 0
        for index, layer in enumerate(cache.layers):
            if type(layer) is StaticLayer:
                layer.keys, layer.values = layer.keys[start:stop], layer.values[start:stop]
                layer.batch_size = stop - start
                if decode:
                    layer.__class__ = RowStaticLayer
                    layer.cumulative_length = self.lengths[index][start:stop]
                else:
                    layer.cumulative_length = layer.cumulative_length.new_zeros(())
                    scalar_bytes += layer.cumulative_length.numel() * layer.cumulative_length.element_size()
                tensors = [layer.keys, layer.values, layer.cumulative_length]
            else:
                assert type(layer) is LinearAttentionLayer
                for name in ('conv_states', 'recurrent_states'):
                    setattr(layer, name, {key: tensor[start:stop]
                                         for key, tensor in getattr(layer, name).items()})
                layer.has_previous_state = dict.fromkeys(layer.has_previous_state, decode)
                tensors = [*layer.conv_states.values(), *layer.recurrent_states.values()]
            for tensor in tensors:
                assert tensor.is_contiguous()
                torch._dynamo.mark_static_address(tensor)
        return CacheRows(cache, start, stop, self.ids[start:stop], self.positions[start:stop],
                         self.key_valid[start:stop], self.logits[start:stop], scalar_bytes)

    def reserve(self, count):
        assert self.pending is None and 0 < count <= self.max_batch_size - self.live_count
        view = self._view(self.live_count, self.live_count + count, False)
        view.cache.reset()
        view.key_valid.fill_(True)
        view.ids.zero_()
        view.positions.zero_()
        self.pending = view
        return view

    def commit_prefill(self, view):
        assert self.pending is view
        for index, layer in enumerate(view.cache.layers):
            if type(layer) is StaticLayer:
                self.lengths[index][view.start:view.stop].copy_(layer.cumulative_length)
            else:
                assert all(layer.has_previous_state.values())
        self.live_count = view.stop
        self.pending = None

    def compact(self, survivors):
        assert self.pending is None
        assert survivors == sorted(set(survivors))
        assert all(0 <= row < self.live_count for row in survivors)
        # Ascending stable compaction cannot overwrite a later surviving source row.
        for destination, source in enumerate(survivors):
            if destination != source:
                for tensor in self.row_tensors():
                    tensor[destination].copy_(tensor[source])
        self.live_count = len(survivors)

    def decode_view(self):
        assert self.pending is None and self.live_count > 0
        return self._view(0, self.live_count, True)

    def padded_decode_view(self, size):
        assert self.pending is None and 0 < self.live_count <= size <= self.max_batch_size
        for tensor in self.row_tensors():
            tensor[self.live_count:size].copy_(tensor[self.live_count - 1:self.live_count].expand(
                size - self.live_count, *tensor.shape[1:]))
        self.key_valid[self.live_count:size].fill_(True)
        return self._view(0, size, True)
