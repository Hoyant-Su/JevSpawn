import torch
from transformers.cache_utils import DynamicLayer

from tests.infra.native_cache_views.read_only import split_read_only


def split_foreach(cache, lengths, stops):
    rows = split_read_only(cache, lengths, stops)
    targets, sources = [], []

    def allocate(source):
        target = torch.empty_like(source, memory_format=torch.contiguous_format)
        sources.append(source)
        targets.append(target)
        return target

    for state in rows:
        for layer in state.layers:
            if type(layer) is DynamicLayer:
                layer.keys, layer.values = allocate(layer.keys), allocate(layer.values)
            else:
                layer.conv_states = {key: allocate(value) for key, value in layer.conv_states.items()}
                layer.recurrent_states = {key: allocate(value) for key, value in layer.recurrent_states.items()}
    torch._foreach_copy_(targets, sources)
    return rows
