from copy import copy

from transformers.cache_utils import DynamicLayer

from jev_spawn.runtime.native_cache_batch import _metadata, _with_layers


def split_read_only(cache, lengths, stops):
    """Expose immutable rows; packing and graph loading allocate independent state."""
    rows = []
    for row, (length, stop) in enumerate(zip(lengths, stops, strict=True)):
        layers = []
        for original in cache.layers:
            names = (('keys', 'values') if type(original) is DynamicLayer
                     else ('conv_states', 'recurrent_states'))
            layer = copy(original)
            layer.__dict__ = _metadata(original, names)
            if type(original) is DynamicLayer:
                for name in names:
                    setattr(layer, name, getattr(original, name)[row:row + 1, :, stop - length:stop, :])
            else:
                for name in names:
                    setattr(layer, name, {index: tensor[row:row + 1]
                                         for index, tensor in getattr(original, name).items()})
            layers.append(layer)
        rows.append(_with_layers(cache, layers))
    return rows
