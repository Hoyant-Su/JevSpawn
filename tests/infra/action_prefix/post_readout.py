from copy import copy
import time

import torch
from transformers.cache_utils import DynamicCache, StaticLayer

from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import _metadata
from tests.infra.action_prefix.runtime import ActionPrefixTail


def snapshot_prefixes(decoder, sequences, indices, settings):
    stop = max(map(len, sequences))
    rows, pairs = [], []
    for index in indices:
        start = stop - len(sequences[index])
        state = DynamicCache(config=decoder.backend.cache_config, offloading=False)
        for layer_index, source in enumerate(decoder.cache.layers):
            if type(source) is StaticLayer:
                layer = state.layers[layer_index]
                for name in ('keys', 'values'):
                    view = getattr(source, name)[index:index + 1, :, start:stop, :]
                    target = torch.empty_like(view)
                    setattr(layer, name, target)
                    pairs.append((target, view))
                layer.dtype, layer.device = layer.keys.dtype, layer.keys.device
                layer.is_initialized = True
            else:
                layer = copy(source)
                layer.__dict__ = _metadata(source, ('conv_states', 'recurrent_states'))
                for name in ('conv_states', 'recurrent_states'):
                    storage = {}
                    for key, tensor in getattr(source, name).items():
                        view = tensor[index:index + 1]
                        storage[key] = torch.empty_like(view)
                        pairs.append((storage[key], view))
                    setattr(layer, name, storage)
                state.layers[layer_index] = layer
        rows.append(state)
    copy_states(pairs, settings)
    return rows


class PostReadoutTail(ActionPrefixTail):
    def __call__(self, backend, sequences, prefixes, owners, states, suffixes, native_ids):
        logits, work, phases = super().__call__(
            backend, sequences, prefixes, owners, states, suffixes, native_ids)
        started = time.perf_counter()
        decoder = self.graphs[self.last_layout['key']]
        positions = {tuple(sequence): index for index, sequence in enumerate(sequences)}

        def snapshot(missing):
            indices = [positions[tuple(sequence)] for sequence in missing]
            return snapshot_prefixes(decoder, sequences, indices, self.state_copy_settings)

        self.prefix_cache.get_many(sequences, snapshot)
        phases['post_readout_snapshot_seconds'] = time.perf_counter() - started
        return logits, work, phases
