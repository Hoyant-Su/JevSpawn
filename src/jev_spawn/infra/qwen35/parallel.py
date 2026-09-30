from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _reduce(module, inputs, output):
    dist.all_reduce(output, op=dist.ReduceOp.SUM, group=module._tp_group)
    return output


def _partition(parameter, dimension, group):
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    assert parameter.shape[dimension] % size == 0
    width = parameter.shape[dimension] // size
    expected = parameter.detach().narrow(dimension, rank * width, width)
    local = expected.contiguous().clone()
    return torch.nn.Parameter(local, requires_grad=False)


def _linear(module, dimension, group):
    assert module.bias is None
    module.weight = _partition(module.weight, dimension, group)
    module.out_features, module.in_features = module.weight.shape
    if dimension == 1:
        module._tp_group = group
        module.register_forward_hook(_reduce)


def _gdn(mixer, group):
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    assert mixer.num_k_heads % size == mixer.num_v_heads % size == 0
    key_heads, value_heads = mixer.num_k_heads // size, mixer.num_v_heads // size
    key_width, value_width = key_heads * mixer.head_k_dim, value_heads * mixer.head_v_dim
    key_start, value_start = rank * key_width, rank * value_width
    indices = torch.cat([
        torch.arange(key_start, key_start + key_width, device=mixer.in_proj_qkv.weight.device),
        torch.arange(mixer.key_dim + key_start, mixer.key_dim + key_start + key_width,
                     device=mixer.in_proj_qkv.weight.device),
        torch.arange(2 * mixer.key_dim + value_start, 2 * mixer.key_dim + value_start + value_width,
                     device=mixer.in_proj_qkv.weight.device)])
    assert mixer.in_proj_qkv.bias is None and mixer.conv1d.bias is None
    original = mixer.in_proj_qkv.weight.detach()
    mixer.in_proj_qkv.weight = torch.nn.Parameter(original.index_select(0, indices), requires_grad=False)
    mixer.in_proj_qkv.out_features = len(indices)
    original = mixer.conv1d.weight.detach()
    mixer.conv1d.weight = torch.nn.Parameter(original.index_select(0, indices), requires_grad=False)
    mixer.conv1d.in_channels = mixer.conv1d.out_channels = mixer.conv1d.groups = len(indices)
    for module in [mixer.in_proj_z, mixer.in_proj_b, mixer.in_proj_a]:
        _linear(module, 0, group)
    _linear(mixer.out_proj, 1, group)
    mixer.A_log = _partition(mixer.A_log, 0, group)
    mixer.dt_bias = _partition(mixer.dt_bias, 0, group)
    mixer.num_k_heads, mixer.num_v_heads = key_heads, value_heads
    mixer.key_dim, mixer.value_dim, mixer.conv_dim = key_width, value_width, len(indices)


class ParallelVocabulary(torch.nn.Module):
    def __init__(self, source, group):
        super().__init__()
        assert source.bias is None
        self.vocab_size, self.hidden_size = source.weight.shape
        self.weight = _partition(source.weight, 0, group)
        self.group, self.world_size = group, dist.get_world_size(group)
        self._selected_weights = {}
        self.train(source.training)

    def selected_weight(self, token_ids):
        assert isinstance(token_ids, tuple) and all(0 <= token < self.vocab_size for token in token_ids)
        if token_ids not in self._selected_weights:
            width = self.weight.shape[0]
            selected = self.weight.new_empty((len(token_ids), self.hidden_size))
            for owner in range(self.world_size):
                positions = [index for index, token in enumerate(token_ids) if token // width == owner]
                if not positions:
                    continue
                indices = torch.tensor([token_ids[index] % width for index in positions], device=self.weight.device)
                values = (self.weight.index_select(0, indices) if dist.get_rank(self.group) == owner
                          else self.weight.new_empty((len(positions), self.hidden_size)))
                dist.broadcast(values, src=dist.get_global_rank(self.group, owner), group=self.group)
                selected.index_copy_(0, torch.tensor(positions, device=self.weight.device), values)
            self._selected_weights[token_ids] = selected
        return self._selected_weights[token_ids]

    def forward(self, hidden_states):
        local = F.linear(hidden_states, self.weight).reshape(-1, self.weight.shape[0])
        gathered = torch.empty((self.world_size * local.shape[0], local.shape[1]),
                               dtype=local.dtype, device=local.device)
        dist.all_gather_into_tensor(gathered, local, group=self.group)
        return gathered.view(self.world_size, *hidden_states.shape[:-1], self.weight.shape[0]).movedim(
            0, -2).reshape(*hidden_states.shape[:-1], self.vocab_size)


@torch.no_grad()
def shard_qwen35(model, settings, group):
    assert not model.training
    language = model.get_submodule(settings['language_model_path'])
    for layer in language.layers:
        if layer.block_type == 'full_attention':
            attention = layer.self_attn
            assert (attention.config.num_attention_heads % dist.get_world_size(group) == 0
                    and attention.config.num_key_value_heads % dist.get_world_size(group) == 0)
            for name in settings['attention_projections']:
                _linear(getattr(attention, name), 0, group)
            _linear(attention.o_proj, 1, group)
        elif layer.block_type == 'linear_attention':
            _gdn(layer.linear_attn, group)
        else:
            raise ValueError(f'Unsupported Qwen decoder block: {layer.block_type}')
        _linear(layer.mlp.gate_proj, 0, group)
        _linear(layer.mlp.up_proj, 0, group)
        _linear(layer.mlp.down_proj, 1, group)
        layer.mlp.intermediate_size = layer.mlp.gate_proj.out_features
    model.lm_head = ParallelVocabulary(model.lm_head, group)
    return model


def local_cache_config(model, group):
    config = deepcopy(model.config)
    text = config.get_text_config(decoder=True)
    size = dist.get_world_size(group)
    for name in ('num_attention_heads', 'num_key_value_heads', 'linear_num_key_heads', 'linear_num_value_heads'):
        count = getattr(text, name)
        assert count % size == 0
        setattr(text, name, count // size)
    return config
