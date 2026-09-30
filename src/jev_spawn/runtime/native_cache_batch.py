from copy import copy, deepcopy

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer, StaticLayer


def _metadata(value, tensors):
    return {name: deepcopy(item) for name, item in vars(value).items() if name not in tensors}


def _with_layers(cache, layers):
    result = copy(cache)
    result.__dict__ = _metadata(cache, ('layers',))
    result.layers = layers
    return result


def _concatenate(tensors):
    first, *_ = tensors
    assert all(tensor.dtype == first.dtype and tensor.device == first.device
               and tensor.shape[1:] == first.shape[1:] for tensor in tensors)
    return torch.cat(tensors, dim=0)


def pack_native_caches(states):
    """Batch compact single-row Qwen hybrid states; only attention KV is padded."""
    assert states and all(type(state) is DynamicCache and not state.offloading for state in states)
    first, *_ = states
    assert all(_metadata(state, ('layers',)) == _metadata(first, ('layers',)) for state in states)
    assert all(len(state.layers) == len(first.layers) for state in states)
    lengths = [state.get_seq_length() for state in states]
    assert all(length > 0 for length in lengths)
    width = max(lengths)
    layers = []
    for siblings in zip(*(state.layers for state in states), strict=True):
        original, *_ = siblings
        assert type(original) in (DynamicLayer, LinearAttentionLayer)
        assert all(type(layer) is type(original) for layer in siblings)
        tensor_names = (('keys', 'values') if type(original) is DynamicLayer
                        else ('conv_states', 'recurrent_states'))
        metadata = _metadata(original, tensor_names)
        assert all(_metadata(layer, tensor_names) == metadata for layer in siblings)
        layer = copy(original)
        layer.__dict__ = metadata
        if type(original) is DynamicLayer:
            assert all(item.is_initialized and item.keys.shape[0] == 1 and item.get_seq_length() == length
                       for item, length in zip(siblings, lengths, strict=True))
            for name in tensor_names:
                setattr(layer, name, _concatenate([
                    F.pad(getattr(item, name), (0, 0, width - length, 0))
                    for item, length in zip(siblings, lengths, strict=True)]))
        else:
            assert all(original.is_conv_states_initialized.values())
            assert all(original.is_recurrent_states_initialized.values())
            for name in tensor_names:
                assert all(tensor.shape[0] == 1 for item in siblings for tensor in getattr(item, name).values())
                setattr(layer, name, {index: _concatenate([getattr(item, name)[index] for item in siblings])
                                     for index in getattr(original, name)})
        layers.append(layer)
    result = _with_layers(first, layers)
    device = next(layer.keys.device for layer in layers if type(layer) is DynamicLayer)
    mask = torch.arange(width, device=device)[None, :] >= width - torch.tensor(lengths, device=device)[:, None]
    return result, mask


def split_native_cache(cache, lengths):
    """Clone rows whose real attention histories are right-aligned in the cache."""
    return split_native_cache_at(cache, lengths, [cache.get_seq_length()] * len(lengths))


def split_native_cache_at(cache, lengths, stops):
    """Exclude right padding while preserving each row's contiguous real history."""
    assert type(cache) is DynamicCache and not cache.offloading
    width = cache.get_seq_length()
    assert lengths and all(0 < length <= width for length in lengths)
    rows = []
    for row, (length, stop) in enumerate(zip(lengths, stops, strict=True)):
        assert length <= stop <= width
        layers = []
        for original in cache.layers:
            assert type(original) in (DynamicLayer, LinearAttentionLayer)
            tensor_names = (('keys', 'values') if type(original) is DynamicLayer
                            else ('conv_states', 'recurrent_states'))
            layer = copy(original)
            layer.__dict__ = _metadata(original, tensor_names)
            if type(original) is DynamicLayer:
                assert original.is_initialized and original.keys.shape[0] == len(lengths)
                assert original.get_seq_length() == width
                for name in tensor_names:
                    setattr(layer, name, getattr(original, name)[row:row + 1, :, stop - length:stop, :].clone())
            else:
                assert all(original.is_conv_states_initialized.values())
                assert all(original.is_recurrent_states_initialized.values())
                for name in tensor_names:
                    assert all(tensor.shape[0] == len(lengths) for tensor in getattr(original, name).values())
                    setattr(layer, name, {index: tensor[row:row + 1].clone()
                                         for index, tensor in getattr(original, name).items()})
            layers.append(layer)
        rows.append(_with_layers(cache, layers))
    return rows
