import torch
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.infra.qwen35.gates import gdn_gates
from jev_spawn.infra.qwen35.ragged_gdn import ragged_gdn_forward


def gdn_forward(module, hidden_states, cache_params=None, attention_mask=None, decode_attention=None,
                ragged_suffix=None, **kwargs):
    if ragged_suffix is not None:
        return ragged_gdn_forward(module, hidden_states, cache_params, attention_mask, ragged_suffix)
    hidden_states = modeling_qwen3_5.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    previous = cache_params is not None and cache_params.has_previous_state(module.layer_idx)
    mixed_qkv, z, b, a = module.input_projections(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = z.reshape(batch, length, -1, module.head_v_dim)
    if previous and length == 1 and not cache_params.layers[module.layer_idx].record_past:
        mixed_qkv = modeling_qwen3_5.causal_conv1d_update(
            mixed_qkv, cache_params.layers[module.layer_idx].conv_states[0],
            module.conv1d.weight.squeeze(1), module.conv1d.bias, module.activation)
    else:
        if cache_params is not None:
            mixed_qkv = cache_params.update_conv_state(
                mixed_qkv, module.layer_idx, conv_kernel_size=module.conv_kernel_size)
        mixed_qkv = modeling_qwen3_5.causal_conv1d_fn(
            mixed_qkv, module.conv1d.weight.squeeze(1), module.conv1d.bias,
            activation=module.activation, **kwargs)
        if cache_params is not None:
            mixed_qkv = mixed_qkv[:, :, -length:]
    query, key, value = mixed_qkv.transpose(1, 2).split(
        [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    value = value.reshape(batch, length, -1, module.head_v_dim)
    repeat = module.num_v_heads // module.num_k_heads
    if repeat > 1:
        query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
    g, beta = gdn_gates(module, a, b)
    recurrent = cache_params.layers[module.layer_idx].recurrent_states[0] if previous else None
    transition = (modeling_qwen3_5.torch_recurrent_gated_delta_rule if previous and length == 1 else
                  modeling_qwen3_5.torch_chunk_gated_delta_rule)
    output, final_state = transition(
        query, key, value, g=g, beta=beta, initial_state=recurrent,
        output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
        cu_seqlens=kwargs.pop('cu_seq_lens_q', None), **kwargs)
    if cache_params is not None:
        cache_params.update_recurrent_state(final_state, module.layer_idx)
    output = module.norm(output.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
    return module.out_proj(output.reshape(batch, length, -1))
