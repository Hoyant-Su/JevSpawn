import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.infra.qwen35.gates import gdn_gates


def dense_suffix_forward(module, hidden_states, cache_params, attention_mask, ragged_suffix):
    hidden_states = modeling_qwen3_5.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch, length, _ = hidden_states.shape
    layer = cache_params.layers[module.layer_idx]
    mixed_qkv, z, b, a = module.input_projections(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    z = z.reshape(batch, length, -1, module.head_v_dim)
    history = torch.cat([layer.conv_states[0], mixed_qkv], dim=-1)
    mixed_qkv = modeling_qwen3_5.causal_conv1d_fn(
        history, module.conv1d.weight.squeeze(1), module.conv1d.bias,
        activation=module.activation)[:, :, -length:]
    positions = ragged_suffix.lengths[:, None, None] + torch.arange(
        module.conv_kernel_size, device=history.device)[None, None, :]
    layer.conv_states[0].copy_(history.gather(-1, positions.expand(-1, history.shape[1], -1)))
    query, key, value = mixed_qkv.transpose(1, 2).split(
        [module.key_dim, module.key_dim, module.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, module.head_k_dim)
    key = key.reshape(batch, length, -1, module.head_k_dim)
    value = value.reshape(batch, length, -1, module.head_v_dim)
    repeat = module.num_v_heads // module.num_k_heads
    query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
    g, beta = gdn_gates(module, a, b)
    valid = torch.arange(length, device=hidden_states.device)[None, :] < ragged_suffix.lengths[:, None]
    # g=0 and beta=0 make each padded recurrence step the identity on its state.
    g = g.masked_fill(~valid[..., None], 0)
    beta = beta.masked_fill(~valid[..., None], 0)
    output, final_state = chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=layer.recurrent_states[0], output_final_state=True,
        use_qk_l2norm_in_kernel=True)
    cache_params.update_recurrent_state(final_state, module.layer_idx)
    output = output.masked_fill(~valid[..., None, None], 0)
    output = module.norm(output.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
    return module.out_proj(output.reshape(batch, length, -1))
