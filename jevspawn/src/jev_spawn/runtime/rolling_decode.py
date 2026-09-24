from copy import copy

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin, StaticLayer

from jev_spawn.runtime.decoding import CapturedDecode


class RowStaticLayer(StaticLayer):
    def update(self, key_states, value_states, *args, **kwargs):
        assert key_states.shape[2] == 1
        position = self.cumulative_length[:, None, None, None]
        self.keys.scatter_(2, position.expand_as(key_states), key_states)
        self.values.scatter_(2, position.expand_as(value_states), value_states)
        self.cumulative_length.add_(1)
        return self.keys, self.values


class RollingDecode(CapturedDecode):
    """Repack real hybrid-attention states when requests enter or leave a batch."""

    def __init__(self, backend, batch_size, capacity, template):
        super().__init__(backend, batch_size, capacity)
        layers = []
        for original in template.cache.layers:
            layer = copy(original)
            if isinstance(original, LinearAttentionCacheLayerMixin):
                for name in ('is_conv_states_initialized', 'is_recurrent_states_initialized',
                             'has_previous_state', 'conv_kernel_size'):
                    setattr(layer, name, getattr(original, name).copy())
                for name in ('conv_states', 'recurrent_states'):
                    setattr(layer, name, {key: value.new_empty((batch_size, *value.shape[1:]))
                                         for key, value in getattr(original, name).items()})
            else:
                assert type(original) in (StaticLayer, RowStaticLayer)
                layer.__class__ = RowStaticLayer
                layer.max_cache_len = capacity
                layer.batch_size = batch_size
                layer.keys = original.keys.new_zeros((batch_size, original.keys.shape[1], capacity,
                                                     original.keys.shape[3]))
                layer.values = original.values.new_zeros((batch_size, original.values.shape[1], capacity,
                                                         original.values.shape[3]))
                layer.cumulative_length = torch.empty(batch_size, device=backend.device, dtype=torch.long)
            layers.append(layer)
        self.cache.layers = layers

    def load(self, sources):
        # Concatenation snapshots each source on the GPU before an in-place destination is overwritten.
        for name in ('ids', 'positions', 'logits'):
            values = torch.cat([getattr(source, name).index_select(0, rows) for source, rows in sources])
            getattr(self, name).copy_(values)
        masks = []
        for source, rows in sources:
            width = min(self.capacity, source.capacity)
            mask = torch.ones((len(rows), self.capacity), device=self.backend.device, dtype=torch.bool)
            mask[:, :width].copy_(source.key_valid.index_select(0, rows)[:, :width])
            masks.append(mask)
        self.key_valid.copy_(torch.cat(masks))
        for index, destination in enumerate(self.cache.layers):
            if isinstance(destination, LinearAttentionCacheLayerMixin):
                for name in ('conv_states', 'recurrent_states'):
                    for key, tensor in getattr(destination, name).items():
                        tensor.copy_(torch.cat([getattr(source.cache.layers[index], name)[key].index_select(0, rows)
                                                for source, rows in sources]))
            else:
                lengths = torch.cat([source.cache.layers[index].cumulative_length.expand(source.ids.shape[0])
                                     .index_select(0, rows) for source, rows in sources])
                for name in ('keys', 'values'):
                    chunks = [getattr(source.cache.layers[index], name).index_select(0, rows)
                              for source, rows in sources]
                    target = getattr(destination, name)
                    target.zero_()
                    offset = 0
                    for chunk in chunks:
                        width = min(self.capacity, chunk.shape[2])
                        target[offset:offset + len(chunk), :, :width].copy_(chunk[:, :, :width])
                        offset += len(chunk)
                destination.cumulative_length.copy_(lengths)
        self.validate_attention()

    def step(self):
        lengths = self.cache.get_seq_length()
        valid = self.key_valid & (self.key_positions <= lengths[:, None])
        output = self.trunk(input_ids=self.ids, position_ids=self.positions,
                            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
                            past_key_values=self.cache, use_cache=True, **self.attention_arguments())
        self.logits.copy_(self.backend.model.lm_head(output.last_hidden_state[:, -1]))
        self.ids.copy_(self.logits.argmax(-1)[:, None])
        self.positions.add_(1)
