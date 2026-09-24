import math

import torch
from transformers import StaticCache
from transformers.cache_utils import LinearAttentionLayer, StaticLayer

from jev_spawn.infra.configuration import EXECUTION_POLICY


class StaticCacheArena:
    """Shared storage for serialized Qwen3.5 caches, reset before every prefill."""

    def __init__(self, config, max_batch_size, max_cache_len, dtype, device):
        text = config.get_text_config(decoder=True)
        assert text.model_type == EXECUTION_POLICY['cache_arena']['model_type']
        assert dtype == getattr(torch, EXECUTION_POLICY['cache_arena']['dtype'])
        assert text.mamba_ssm_dtype == EXECUTION_POLICY['cache_arena']['recurrent_dtype']
        assert max_batch_size > 0 and max_cache_len > 0
        self.config = config
        self.max_batch_size, self.max_cache_len = max_batch_size, max_cache_len
        self.dtype, self.device = dtype, torch.device(device)
        self.heads, self.head_dim = text.num_key_value_heads, text.head_dim
        self.conv_channels = 2 * text.linear_num_key_heads * text.linear_key_head_dim + text.linear_num_value_heads * text.linear_value_head_dim
        self.conv_width = text.linear_conv_kernel_dim
        self.recurrent_shape = (text.linear_num_value_heads, text.linear_key_head_dim, text.linear_value_head_dim)
        assert text.linear_key_head_dim == text.linear_value_head_dim
        self.storage = []
        for kind in text.layer_types:
            assert kind in EXECUTION_POLICY['cache_arena']['layer_types']
            shapes = {'keys': (max_batch_size, self.heads, max_cache_len, self.head_dim),
                      'values': (max_batch_size, self.heads, max_cache_len, self.head_dim)} if kind == 'full_attention' else {
                      'conv': (max_batch_size, self.conv_channels, self.conv_width),
                      'recurrent': (max_batch_size, *self.recurrent_shape)}
            layer = {name: torch.zeros(math.prod(shape), device=self.device,
                     dtype=getattr(torch, EXECUTION_POLICY['cache_arena']['recurrent_dtype']) if name == 'recurrent' else dtype) for name, shape in shapes.items()}
            if kind == 'full_attention':
                layer['length'] = torch.zeros((), dtype=torch.long, device=self.device)
            self.storage.append(layer)

    @property
    def nbytes(self):
        return sum(tensor.numel() * tensor.element_size() for layer in self.storage for tensor in layer.values())

    def bind(self, batch_size, capacity):
        assert 0 < batch_size <= self.max_batch_size and 0 < capacity <= self.max_cache_len
        cache = StaticCache(config=self.config, max_cache_len=capacity)
        for layer, storage in zip(cache.layers, self.storage, strict=True):
            layer.dtype, layer.device = self.dtype, self.device
            if type(layer) is StaticLayer:
                shape = (batch_size, self.heads, capacity, self.head_dim)
                layer.keys = storage['keys'][:math.prod(shape)].view(shape)
                layer.values = storage['values'][:math.prod(shape)].view(shape)
                layer.cumulative_length = storage['length']
                layer.batch_size, layer.num_heads = batch_size, self.heads
                layer.k_head_dim = layer.v_head_dim = self.head_dim
                layer.is_initialized = True
                tensors = [layer.keys, layer.values, layer.cumulative_length]
            else:
                assert type(layer) is LinearAttentionLayer and layer.number_of_states == 1
                shape = (batch_size, self.conv_channels, self.conv_width)
                layer.conv_states[0] = storage['conv'][:math.prod(shape)].view(shape)
                shape = (batch_size, *self.recurrent_shape)
                layer.recurrent_states[0] = storage['recurrent'][:math.prod(shape)].view(shape)
                layer.conv_kernel_size[0] = self.conv_width
                layer.is_conv_states_initialized[0] = True
                layer.is_recurrent_states_initialized[0] = True
                tensors = [layer.conv_states[0], layer.recurrent_states[0]]
            for tensor in tensors:
                assert tensor.is_contiguous()
                torch._dynamo.mark_static_address(tensor)
        return cache
