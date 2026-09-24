from collections import defaultdict
import time

import torch

from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache, split_native_cache_at
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


def grouped_suffix(backend, states, tails, work):
    cohorts = defaultdict(list)
    results = dict(enumerate(states))
    for index, tail in enumerate(tails):
        if tail:
            cohorts[len(tail)].append(index)
    for width, indices in cohorts.items():
        cache, prefix_mask = pack_native_caches([states[index] for index in indices])
        ids, mask = padded([tails[index] for index in indices], backend.tokenizer.pad_token_id,
                           backend.device, 'right')
        full_mask = torch.cat([prefix_mask, mask], dim=-1)
        positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -width:]
        output = backend.model.model(input_ids=ids, attention_mask=full_mask,
            position_ids=positions, past_key_values=cache, use_cache=True)
        lengths = [states[index].get_seq_length() + len(tails[index]) for index in indices]
        results.update(zip(indices, split_native_cache(output.past_key_values, lengths), strict=True))
        work['computed_input_tokens'] += sum(len(tails[index]) for index in indices)
        work['padded_input_tokens'] += ids.numel()
    return [results[index] for index in range(len(states))]


def ragged_suffix(backend, states, tails, work):
    for name in ('suffix_pack_seconds', 'suffix_forward_seconds', 'suffix_split_seconds'):
        work.setdefault(name, 0.0)
    indices = [index for index, tail in enumerate(tails) if tail]
    results = dict(enumerate(states))
    if indices:
        phase = time.perf_counter()
        selected = [states[index] for index in indices]
        lengths = [len(tails[index]) for index in indices]
        descriptor = RaggedSuffix(lengths, backend.device)
        attention = RaggedCacheAttention(descriptor, [state.get_seq_length() for state in selected])
        cache, prefix_mask = pack_native_caches(selected)
        ids, mask = padded([tails[index] for index in indices], backend.tokenizer.pad_token_id,
                           backend.device, 'right')
        full_mask = torch.cat([prefix_mask, mask], dim=-1)
        positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        work['suffix_pack_seconds'] += time.perf_counter() - phase
        phase = time.perf_counter()
        output = backend.model.model(input_ids=ids, attention_mask=full_mask,
            position_ids=positions, past_key_values=cache, use_cache=True,
            ragged_suffix=descriptor, decode_attention=attention)
        work['suffix_forward_seconds'] += time.perf_counter() - phase
        phase = time.perf_counter()
        total_lengths = [state.get_seq_length() + length for state, length in zip(selected, lengths, strict=True)]
        stops = [prefix_mask.shape[1] + length for length in lengths]
        results.update(zip(indices, split_native_cache_at(output.past_key_values, total_lengths, stops), strict=True))
        work['suffix_split_seconds'] += time.perf_counter() - phase
        work['computed_input_tokens'] += sum(lengths)
        work['padded_input_tokens'] += ids.numel()
    return [results[index] for index in range(len(states))]
