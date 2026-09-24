from copy import copy

import torch
from transformers.cache_utils import DynamicLayer

from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import _metadata, _with_layers


def pack_batched(states, settings):
    lengths = [state.get_seq_length() for state in states]
    width = max(lengths)
    layers, pairs, padded = [], [], []
    for siblings in zip(*(state.layers for state in states), strict=True):
        original = siblings[0]
        names = (('keys', 'values') if type(original) is DynamicLayer
                 else ('conv_states', 'recurrent_states'))
        layer = copy(original)
        layer.__dict__ = _metadata(original, names)
        if type(original) is DynamicLayer:
            for name in names:
                source = getattr(original, name)
                destination = torch.empty((len(states), source.shape[1], width, source.shape[3]),
                                          dtype=source.dtype, device=source.device)
                padded.append(destination)
                for row, (sibling, length) in enumerate(zip(siblings, lengths, strict=True)):
                    pairs.append((destination[row:row + 1, :, width - length:], getattr(sibling, name)))
                setattr(layer, name, destination)
        else:
            for name in names:
                setattr(layer, name, {index: torch.cat(
                    [getattr(sibling, name)[index] for sibling in siblings], dim=0)
                    for index in getattr(original, name)})
        layers.append(layer)
    torch._foreach_zero_(padded)
    copy_states(pairs, settings)
    device = padded[0].device
    mask = torch.arange(width, device=device)[None] >= width - torch.tensor(lengths, device=device)[:, None]
    return _with_layers(states[0], layers), mask
