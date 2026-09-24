import torch
from transformers.models.qwen3_5 import modeling_qwen3_5


class Qwen35Attention(modeling_qwen3_5.Qwen3_5Attention):
    @classmethod
    def from_native(cls, source):
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        for name in ('config', 'layer_idx', 'head_dim', 'num_key_value_groups', 'scaling',
                     'attention_dropout', 'is_causal', 'q_proj', 'k_proj', 'v_proj',
                     'o_proj', 'q_norm', 'k_norm'):
            setattr(module, name, getattr(source, name))
        module.train(source.training)
        return module

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, decode_attention=None, ragged_suffix=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query, key, value = self.input_projections(hidden_states)
        query, gate = torch.chunk(query.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = self.k_norm(key.view(hidden_shape)).transpose(1, 2)
        value = value.view(hidden_shape).transpose(1, 2)
        query, key = modeling_qwen3_5.apply_rotary_pos_emb(query, key, *position_embeddings)
        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)
        attention = (modeling_qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, modeling_qwen3_5.eager_attention_forward)
            if decode_attention is None else decode_attention)
        output, weights = attention(self, query, key, value, attention_mask,
            dropout=self.attention_dropout if self.training else 0.0, scaling=self.scaling, **kwargs)
        output = output.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
        return self.o_proj(output), weights
