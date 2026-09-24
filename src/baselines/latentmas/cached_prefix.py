from types import MethodType

import torch
from causal_conv1d import causal_conv1d_update
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
from torch.nn import functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, apply_mask_to_padding_states

from baselines.latentmas.native_hybrid_transport import ChunkedHybridTransport


def cached_delta_forward(module, hidden_states, cache_params, attention_mask, **kwargs):
    assert cache_params.has_previous_state(module.layer_idx)
    layer = cache_params.layers[module.layer_idx]
    assert not layer.record_past
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    mixed = module.in_proj_qkv(hidden_states).transpose(1, 2)
    mixed = causal_conv1d_update(mixed, layer.conv_states[0],
        module.conv1d.weight.squeeze(1), module.conv1d.bias, module.activation).transpose(1, 2)
    query, key, value = torch.split(mixed, [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    key = key.reshape(batch, length, module.num_k_heads, module.head_k_dim)
    value = value.reshape(batch, length, module.num_v_heads, module.head_v_dim)
    repetitions = module.num_v_heads // module.num_k_heads
    query = query.repeat_interleave(repetitions, dim=2)
    key = key.repeat_interleave(repetitions, dim=2)
    beta = module.in_proj_b(hidden_states).sigmoid()
    decay = -module.A_log.float().exp() * F.softplus(module.in_proj_a(hidden_states).float() + module.dt_bias)
    output, state = fused_recurrent_gated_delta_rule(query, key, value, g=decay, beta=beta,
        initial_state=layer.recurrent_states[0], output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    cache_params.update_recurrent_state(state, module.layer_idx)
    gate = module.in_proj_z(hidden_states).reshape(-1, module.head_v_dim)
    output = module.norm(output.reshape(-1, module.head_v_dim), gate)
    return module.out_proj(output.reshape(batch, length, -1))


class CachedPrefixTransport(ChunkedHybridTransport):
    def _forward(self, embeddings, mask, cache):
        modules = [module for module in self.trunk.modules() if isinstance(module, Qwen3_5GatedDeltaNet)]
        original = [module.forward for module in modules]
        for module in modules:
            module.forward = MethodType(cached_delta_forward, module)
        try:
            return super()._forward(embeddings, mask, cache)
        finally:
            for module, forward in zip(modules, original, strict=True):
                module.forward = forward
