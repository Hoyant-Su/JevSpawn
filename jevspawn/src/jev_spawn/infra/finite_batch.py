from collections import defaultdict
import time

import torch
import torch.nn.functional as F

from jev_spawn.algo.structured import common_prefix, padded
from jev_spawn.infra.readout_labels import validate_admitted_boundaries
from jev_spawn.infra.cached_suffix import grouped_suffix
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache


def score_finite_batch(backend, requests, base_lengths, base_cache, physical_batch_size=None):
    return score_finite_with_tail(backend, requests, base_lengths, base_cache,
                                  eager_finite_tail, grouped_suffix, common_prefix, extend_prefixes, physical_batch_size)


@torch.inference_mode()
def score_finite_with_tail(backend, requests, base_lengths, base_cache, tail, extend_states, prefix_length, prefix_extension,
                           physical_batch_size=None):
    device, tokenizer = backend.device, backend.tokenizer
    started = time.perf_counter()
    active_count = len(requests)
    if physical_batch_size is not None and active_count < physical_batch_size:
        padding = physical_batch_size - active_count
        requests = [*requests, *([requests[-1]] * padding)]
        base_lengths = [*base_lengths, *([base_lengths[-1]] * padding)]
    sequences = [list(request.admitted.tokens) for request in requests]
    counts = [len(request.field['options']) for request in requests]
    labels = list(backend.answer_labels[:max(counts)])
    native_ids = backend.answer_label_ids[:max(counts)]
    validate_admitted_boundaries(tokenizer, [request.admitted for request in requests],
                                labels, native_ids, counts, backend.answer_boundary_cache)
    groups = defaultdict(list)
    for index, request in enumerate(requests):
        groups[(request.task_id, request.field['context'])].append(index)
    owners = {}
    prefixes, bases = [], []
    for owner, indices in enumerate(groups.values()):
        first = indices[0]
        length = min(prefix_length([sequences[index] for index in indices]),
                     min(len(sequences[index]) - 1 for index in indices))
        assert len({base_lengths[index] for index in indices}) == 1
        assert base_lengths[first] <= length
        prefixes.append(sequences[first][:length])
        bases.append(sequences[first][:base_lengths[first]])
        owners.update((index, owner) for index in indices)
    suffixes = [sequence[len(prefixes[owners[index]]):] for index, sequence in enumerate(sequences)]
    work = {'suffix_pack_seconds': 0.0, 'suffix_forward_seconds': 0.0, 'suffix_split_seconds': 0.0,
                'computed_input_tokens': 0, 'padded_input_tokens': 0, 'graph_replays': 0, 'graph_captures': 0,
                'reused_state_tokens': 0, 'reused_root_tokens': 0}
    phases = {'prepare_seconds': time.perf_counter() - started, 'capture_seconds': 0.0,
              'root_prefill_seconds': 0.0, 'state_extension_seconds': 0.0}

    def prefill(sequences):
        ids, mask = padded(sequences, tokenizer.pad_token_id, device, 'left')
        output = backend.model.model(input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
        work['computed_input_tokens'] += sum(map(len, sequences))
        work['padded_input_tokens'] += ids.numel()
        return split_native_cache(output.past_key_values, list(map(len, sequences)))

    phase = time.perf_counter()
    states, hits = base_cache.get_many(bases, prefill)
    work['reused_root_tokens'] = sum(len(base) for base, hit in zip(bases, hits, strict=True) if hit)
    phases['root_prefill_seconds'] = time.perf_counter() - phase
    phase_extension = time.perf_counter()
    states = prefix_extension(backend, states, prefixes, bases, work, extend_states)
    phases['state_extension_seconds'] += time.perf_counter() - phase_extension
    phases['prefix_seconds'] = time.perf_counter() - phase
    logits, tail_work, tail_phases = tail(backend, sequences, prefixes, owners, states, suffixes, native_ids)
    for name, count in tail_work.items():
        work[name] += count
    phases.update(tail_phases)
    phase = time.perf_counter()
    valid = torch.arange(len(labels), device=device)[None] < torch.tensor(counts, device=device)[:, None]
    masked = logits.masked_fill(~valid, -torch.inf)
    probabilities = masked.softmax(-1)
    ranked = masked.argsort(dim=-1, descending=True, stable=True)
    chosen = ranked[:, 0].tolist()
    ranked_rows = ranked.tolist()
    probability_rows, logit_rows = probabilities.tolist(), logits.tolist()
    answers = [{'id': request.field['id'], 'choice': request.field['options'][choice]['id'],
                'probabilities': probability[:count], 'option_logits': row[:count],
                'option_ids': [option['id'] for option in request.field['options']],
                'ranked_option_ids': [request.field['options'][index]['id'] for index in ranking[:count]],
                'input_tokens': len(sequence)}
               for request, choice, probability, row, ranking, count, sequence in
               zip(requests, chosen, probability_rows, logit_rows, ranked_rows, counts, sequences, strict=True)]
    phases['readout_seconds'] += time.perf_counter() - phase
    answers = answers[:active_count]
    host_timings = dict(phases)
    for name in ('suffix_pack_seconds', 'suffix_forward_seconds', 'suffix_split_seconds'):
        host_timings[name] = work.pop(name)
    return {'groups': [answers], 'device_probabilities': probabilities[:active_count],
            'root_batch_size': len(groups), 'group_sizes': list(map(len, groups.values())),
            'batch_size': active_count, 'physical_batch_size': len(requests),
            'logical_field_count': active_count, 'timings': {'capture_seconds': phases['capture_seconds']},
            'unfenced_host_timings': host_timings,
            'elapsed_seconds': time.perf_counter() - started, 'logical_input_tokens': sum(map(len, sequences)),
            **work, 'prefix_tokens': list(map(len, prefixes)), 'suffix_tokens': list(map(len, suffixes)),
            'root_prefix_tokens': list(map(len, bases)), 'persistent_prefix_scope': 'task_root',
            'persistent_prefix_hits': hits[:active_count], 'option_counts': counts[:active_count],
            'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def extend_prefixes(backend, states, prefixes, bases, work, extend_states):
    return extend_states(backend, states,
        [prefix[len(base):] for prefix, base in zip(prefixes, bases, strict=True)], work)


def eager_finite_tail(backend, sequences, prefixes, owners, states, suffixes, native_ids):
    device, tokenizer = backend.device, backend.tokenizer
    phase = time.perf_counter()
    cache, prefix_mask = pack_native_caches([states[owners[index]] for index in range(len(sequences))])
    ids, mask = padded(suffixes, tokenizer.pad_token_id, device, 'right')
    full_mask = torch.cat([prefix_mask, mask], dim=-1)
    positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
    output = backend.model.model(input_ids=ids, attention_mask=full_mask,
        position_ids=positions, past_key_values=cache, use_cache=True)
    ends = torch.tensor(list(map(len, suffixes)), device=device) - 1
    hidden = output.last_hidden_state[torch.arange(len(sequences), device=device), ends]
    work = {'computed_input_tokens': sum(map(len, suffixes)), 'padded_input_tokens': ids.numel()}
    torch.cuda.synchronize(device)
    phases = {'tiles_seconds': time.perf_counter() - phase}
    phase = time.perf_counter()
    weight = backend.finite_output_weights[:len(native_ids)]
    logits = F.linear(hidden.float(), weight)
    torch.cuda.synchronize(device)
    phases['readout_seconds'] = time.perf_counter() - phase
    return logits, work, phases
