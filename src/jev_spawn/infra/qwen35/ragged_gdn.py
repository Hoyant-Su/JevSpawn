import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.infra.qwen35.finite_gates import gdn_gates


def ragged_gdn_forward(module, hidden_states, cache_params, attention_mask, ragged_suffix):
    hidden_states = modeling_qwen3_5.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    assert (batch, length) == (ragged_suffix.batch_size, ragged_suffix.width)
    assert cache_params is not None and cache_params.has_previous_state(module.layer_idx)
    layer = cache_params.layers[module.layer_idx]
    assert not layer.record_past and layer.number_of_states == 1
    mixed_qkv, z, b, a = module.input_projections(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = z.reshape(batch, length, -1, module.head_v_dim)
    history = torch.cat([layer.conv_states[0], mixed_qkv], dim=-1)
    mixed_qkv = modeling_qwen3_5.causal_conv1d_fn(
        history, module.conv1d.weight.squeeze(1), module.conv1d.bias,
        activation=module.activation)[:, :, -length:]
    # The raw K-token history ends at K + the actual suffix length, before any padding.
    positions = ragged_suffix.lengths[:, None, None] + torch.arange(
        module.conv_kernel_size, device=history.device)[None, None, :]
    layer.conv_states[0].copy_(history.gather(-1, positions.expand(-1, history.shape[1], -1)))
    query, key, value = mixed_qkv.transpose(1, 2).split(
        [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    value = value.reshape(batch, length, -1, module.head_v_dim)
    repeat = module.num_v_heads // module.num_k_heads
    if repeat > 1:
        query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
    g, beta = gdn_gates(module, a, b)
    output, final_state = chunk_gated_delta_rule(
        ragged_suffix.pack(query), ragged_suffix.pack(key), ragged_suffix.pack(value),
        g=ragged_suffix.pack(g), beta=ragged_suffix.pack(beta),
        initial_state=layer.recurrent_states[0], output_final_state=True,
        use_qk_l2norm_in_kernel=True, cu_seqlens=ragged_suffix.cu_seqlens,
        cu_seqlens_cpu=ragged_suffix.cu_seqlens_cpu)
    cache_params.update_recurrent_state(final_state, module.layer_idx)
    output = ragged_suffix.unpack(output)
    output = module.norm(output.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
    return module.out_proj(output.reshape(batch, length, -1))
