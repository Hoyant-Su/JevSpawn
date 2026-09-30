import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer, StaticLayer

from jev_spawn.runtime.batched_state_copy import copy_states


@torch.inference_mode()
def load_native_prefixes(decoder, states, last_tokens, settings):
    """Restore prefixes before each evaluation; states contain exactly prompt[:-1]."""
    assert len(states) == decoder.ids.shape[0] == len(last_tokens)
    assert all(type(state) is DynamicCache and not state.offloading for state in states)
    assert all(len(state.layers) == len(decoder.cache.layers) for state in states)
    lengths = [state.get_seq_length() for state in states]
    width = max(lengths)
    assert all(length > 0 for length in lengths) and width < decoder.capacity
    pairs = []
    for index, destination in enumerate(decoder.cache.layers):
        sources = [state.layers[index] for state in states]
        if type(destination) is StaticLayer:
            assert all(type(source) is DynamicLayer and source.is_initialized
                       and source.get_seq_length() == length
                       for source, length in zip(sources, lengths, strict=True))
            for row, (source, length) in enumerate(zip(sources, lengths, strict=True)):
                for name in ('keys', 'values'):
                    pairs.append((getattr(destination, name)[row:row + 1, :, width - length:width, :],
                                  getattr(source, name)))
            destination.cumulative_length.fill_(width)
        else:
            assert type(destination) is LinearAttentionLayer
            first, *_ = sources
            assert all(type(source) is LinearAttentionLayer and not source.record_past for source in sources)
            for name in ('number_of_states', 'conv_kernel_size', 'is_conv_states_initialized',
                         'is_recurrent_states_initialized', 'has_previous_state'):
                expected = getattr(first, name)
                assert all(getattr(source, name) == expected for source in sources)
                assert getattr(destination, name) == expected or name == 'has_previous_state'
            assert all(first.has_previous_state.values())
            destination.has_previous_state = dict(first.has_previous_state)
            destination.record_past = first.record_past
            for row, source in enumerate(sources):
                for name in ('conv_states', 'recurrent_states'):
                    for state_index, tensor in getattr(source, name).items():
                        pairs.append((getattr(destination, name)[state_index][row:row + 1], tensor))
    copy_states(pairs, settings)
    lengths_gpu = torch.tensor(lengths, device=decoder.ids.device, dtype=decoder.positions.dtype)
    decoder.positions.copy_(lengths_gpu[:, None])
    decoder.ids.copy_(torch.tensor(last_tokens, device=decoder.ids.device, dtype=decoder.ids.dtype)[:, None])
    decoder.key_valid.copy_(decoder.key_positions[None, :] >= width - lengths_gpu[:, None])
    decoder.validate_attention()
