from collections import defaultdict
from copy import copy
from dataclasses import dataclass

from flash_attn import flash_attn_varlen_func
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicLayer
from transformers.models.qwen3_5 import modeling_qwen3_5

from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.infra.qwen35.finite_gates import gdn_gates
from jev_spawn.runtime.native_cache_batch import _metadata, _with_layers
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


@dataclass(eq=False)
class Segment:
    parent: object
    tokens: list
    length: int
    depth: int
    layers: list

    def get_seq_length(self):
        return self.length

    def materialize(self):
        ancestor = self.parent
        while isinstance(ancestor, Segment):
            ancestor = ancestor.parent
        assert len(self.layers) == len(ancestor.layers)
        return _with_layers(ancestor, self.layers)


def pack_layer(parents, index, lengths):
    sources = [parent.layers[index] for parent in parents]
    original = sources[0]
    attention = type(original) is DynamicLayer
    names = ('keys', 'values') if attention else ('conv_states', 'recurrent_states')
    result = copy(original)
    result.__dict__ = _metadata(original, names)
    for name in names:
        if attention:
            tensor = torch.cat([F.pad(getattr(source, name),
                (0, 0, max(lengths) - length, 0))
                for source, length in zip(sources, lengths, strict=True)])
        else:
            tensor = {key: torch.cat([getattr(source, name)[key] for source in sources])
                      for key in getattr(original, name)}
        setattr(result, name, tensor)
    return result


def save_layer(nodes, layer, prefix_width, retained):
    attention = type(layer) is DynamicLayer
    names = ('keys', 'values') if attention else ('conv_states', 'recurrent_states')
    for row, node in enumerate(nodes):
        if node not in retained:
            continue
        result = copy(layer)
        result.__dict__ = _metadata(layer, names)
        stop = prefix_width + len(node.tokens)
        for name in names:
            if attention:
                tensor = getattr(layer, name)[row:row + 1, :, stop - node.length:stop].clone()
            else:
                tensor = {key: value[row:row + 1].clone()
                          for key, value in getattr(layer, name).items()}
            setattr(result, name, tensor)
        node.layers.append(result)


class LayerMergedPrefill:
    def __init__(self, language, nodes, device):
        levels = defaultdict(list)
        for node in nodes:
            levels[node.depth].append(node)
        self.levels = []
        self.nodes = [node for depth in sorted(levels) for node in levels[depth]]
        self.retained = {node.parent for node in self.nodes if isinstance(node.parent, Segment)}
        self.sizes = []
        for depth in sorted(levels):
            group = levels[depth]
            lengths = [len(node.tokens) for node in group]
            prefixes = [node.parent.get_seq_length() for node in group]
            suffix = RaggedSuffix(lengths, device)
            attention = self.attention_layout(suffix, prefixes, group)
            self.levels.append((group, prefixes, suffix, attention))
            self.sizes.append(sum(lengths))
        self.language = language
        self.ids = torch.tensor([[token for node in self.nodes for token in node.tokens]], device=device)
        self.positions = torch.tensor([[position for node in self.nodes
            for position in range(node.parent.get_seq_length(), node.length)]], device=device)
        self.ends = {}
        offset = 0
        for node in self.nodes:
            offset += len(node.tokens)
            self.ends[node] = offset - 1

    def attention_layout(self, suffix, prefixes, nodes):
        return RaggedCacheAttention(suffix, prefixes)

    def attend_level(self, nodes, prefixes, suffix, attention, q, k, v, index, scale):
        layer = pack_layer([node.parent for node in nodes], index, prefixes)
        keys, values = layer.update(suffix.unpack(k).transpose(1, 2), suffix.unpack(v).transpose(1, 2))
        keys = keys.transpose(1, 2).flatten(0, 1).index_select(0, attention.key_indices)
        values = values.transpose(1, 2).flatten(0, 1).index_select(0, attention.key_indices)
        output = flash_attn_varlen_func(q.squeeze(0), keys, values, suffix.cu_seqlens,
            attention.key_offsets, suffix.width, attention.key_width, softmax_scale=scale, causal=True)
        save_layer(nodes, layer, max(prefixes), self.retained)
        return output

    def gdn(self, module, hidden, index):
        mixed, z, b, a = module.input_projections(hidden)
        gates, beta = gdn_gates(module, a, b)
        outputs = []
        for (nodes, prefixes, suffix, _), projected, g, be in zip(self.levels,
                mixed.split(self.sizes, dim=1), gates.split(self.sizes, dim=1),
                beta.split(self.sizes, dim=1), strict=True):
            layer = pack_layer([node.parent for node in nodes], index, prefixes)
            history = torch.cat([layer.conv_states[0], suffix.unpack(projected).transpose(1, 2)], dim=-1)
            convolved = modeling_qwen3_5.causal_conv1d_fn(history,
                module.conv1d.weight.squeeze(1), module.conv1d.bias,
                activation=module.activation)[:, :, -suffix.width:]
            positions = suffix.lengths[:, None, None] + torch.arange(
                module.conv_kernel_size, device=history.device)[None, None, :]
            layer.conv_states[0].copy_(history.gather(-1, positions.expand(-1, history.shape[1], -1)))
            query, key, value = suffix.pack(convolved.transpose(1, 2)).split(
                [module.key_dim, module.key_dim, module.value_dim], dim=-1)
            shape = projected.shape[:-1]
            query = query.reshape(*shape, -1, module.head_k_dim)
            key = key.reshape(*shape, -1, module.head_k_dim)
            value = value.reshape(*shape, -1, module.head_v_dim)
            repeat = module.num_v_heads // module.num_k_heads
            query, key = query.repeat_interleave(repeat, dim=2), key.repeat_interleave(repeat, dim=2)
            output, final = chunk_gated_delta_rule(query, key, value, g=g, beta=be,
                initial_state=layer.recurrent_states[0], output_final_state=True,
                use_qk_l2norm_in_kernel=True, cu_seqlens=suffix.cu_seqlens,
                cu_seqlens_cpu=suffix.cu_seqlens_cpu)
            layer.recurrent_states[0].copy_(final)
            save_layer(nodes, layer, max(prefixes), self.retained)
            outputs.append(output)
        output = torch.cat(outputs, dim=1)
        output = module.norm(output.reshape(-1, module.head_v_dim), z.reshape(-1, module.head_v_dim))
        return module.out_proj(output.reshape(*hidden.shape[:-1], -1))

    def attention(self, module, hidden, index, positions):
        shape = hidden.shape[:-1]
        heads = (*shape, -1, module.head_dim)
        query, key, value = module.input_projections(hidden)
        query, gate = query.view(*shape, -1, module.head_dim * 2).chunk(2, dim=-1)
        query = module.q_norm(query.reshape(heads)).transpose(1, 2)
        key = module.k_norm(key.view(heads)).transpose(1, 2)
        query, key = modeling_qwen3_5.apply_rotary_pos_emb(query, key, *positions)
        outputs = []
        for (nodes, prefixes, suffix, attention), q, k, v in zip(self.levels,
                query.transpose(1, 2).split(self.sizes, dim=1),
                key.transpose(1, 2).split(self.sizes, dim=1),
                value.view(heads).split(self.sizes, dim=1), strict=True):
            output = self.attend_level(nodes, prefixes, suffix, attention, q, k, v, index, module.scaling)
            outputs.append(output.unsqueeze(0))
        output = torch.cat(outputs, dim=1).reshape(*shape, -1) * torch.sigmoid(gate.reshape(*shape, -1))
        return module.o_proj(output)

    def all_hidden(self):
        hidden = self.language.embed_tokens(self.ids)
        positions = self.language.rotary_emb(hidden, self.positions)
        for index, layer in enumerate(self.language.layers):
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            if layer.block_type == 'linear_attention':
                hidden = self.gdn(layer.linear_attn, normalized, index)
            else:
                hidden = self.attention(layer.self_attn, normalized, index, positions)
            hidden = residual + hidden
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.language.norm(hidden).squeeze(0)

    def __call__(self, leaves):
        hidden = self.all_hidden()
        ends = torch.tensor([self.ends[node] for node in leaves], device=hidden.device)
        return hidden.index_select(0, ends)
