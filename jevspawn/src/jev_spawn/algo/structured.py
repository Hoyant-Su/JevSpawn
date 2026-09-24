from copy import deepcopy
import time

import torch
import torch.nn.functional as F

from jev_spawn.schema import CONTROLLER, controller_prompts
from jev_spawn.runtime.streaming import streamed_hidden
from jev_spawn.infra.readout_labels import validate_admitted_boundaries, validate_boundaries
from jev_spawn.infra.configuration import CORE, EXECUTION_POLICY


def padded(sequences, pad_id, device, side):
    width = max(map(len, sequences))
    ids, masks = [], []
    for sequence in sequences:
        padding = width - len(sequence)
        if side == 'left':
            ids.append([pad_id] * padding + sequence)
            masks.append([0] * padding + [1] * len(sequence))
        else:
            ids.append(sequence + [pad_id] * padding)
            masks.append([1] * len(sequence) + [0] * padding)
    return (torch.tensor(ids, device=device), torch.tensor(masks, device=device))


def common_prefix(sequences):
    length = 0
    for tokens in zip(*sequences):
        if len(set(tokens)) != 1:
            break
        length += 1
    return min(length, min(map(len, sequences)) - 1)


@torch.inference_mode()
def score_fields(backend, states, fields, mode, *, field_states=None, prefix_cache=None,
                 prefix_lengths_override=None, base_prefix_cache=None, base_prefix_length=None,
                 admitted_prompts=None):
    assert mode in EXECUTION_POLICY['structured']['modes']
    assert states and fields
    schemas = [fields] * len(states) if isinstance(fields, dict) else fields
    assert len(schemas) == len(states)
    fields = schemas[0]
    names, definitions = list(fields), list(fields.values())
    signature = lambda schema: [(name, [option['id'] for option in field['options']]) for name, field in schema.items()]
    assert all(signature(schema) == signature(fields) for schema in schemas)
    counts = [len(field['options']) for field in definitions]
    assert all(CORE['controller']['minimum_options'] <= count <= len(backend.answer_labels) for count in counts)
    batch, questions = len(states), len(fields)
    prompt_states = [[state] * questions for state in states] if field_states is None else field_states
    assert len(prompt_states) == batch and all(len(group) == questions for group in prompt_states)
    device, tokenizer = backend.device, backend.tokenizer
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    labels = list(backend.answer_labels[:max(counts)])
    native_ids = backend.answer_label_ids[:max(counts)]
    label_ids = [[token] for token in native_ids]
    if admitted_prompts is None:
        prompts = [
            controller_prompts([state], field['question'], field['options'], labels[:count],
                               CONTROLLER['output_instruction'],
                               contexts=[field.get('context', '')])[0]
            for group, schema in zip(prompt_states, schemas)
            for state, field, count in zip(group, schema.values(), counts)
        ]
        rendered = backend._render(prompts, CONTROLLER['system'])
        sequences = tokenizer(rendered, add_special_tokens=False)['input_ids']
        validate_boundaries(tokenizer, rendered, sequences, labels, native_ids, counts * batch)
    else:
        assert len(admitted_prompts) == batch * questions
        sequences = [list(prompt.tokens) for prompt in admitted_prompts]
        validate_admitted_boundaries(tokenizer, admitted_prompts, labels, native_ids, counts * batch,
                                     backend.answer_boundary_cache)
    lengths = [len(sequence) for sequence in sequences]
    assert max(lengths) <= backend.config['max_input_tokens'], 'Full field prompt exceeds max_input_tokens; no truncation.'
    prefix_lengths = [common_prefix(sequences[i:i + questions]) for i in range(0, len(sequences), questions)]
    if prefix_lengths_override is not None:
        assert len(prefix_lengths_override) == batch
        assert all(0 < length <= actual for length, actual in zip(prefix_lengths_override, prefix_lengths))
        prefix_lengths = list(prefix_lengths_override)
    assert prefix_cache is None or mode == 'tiled_shared'
    assert (base_prefix_cache is None) == (base_prefix_length is None)
    assert base_prefix_cache is None or mode == 'tiled_shared'
    assert min(prefix_lengths) > 0, 'Fields must share a nonempty exact token prefix.'
    prefixes = [sequences[i * questions][:length] for i, length in enumerate(prefix_lengths)]
    suffixes = [sequence[prefix_lengths[index // questions]:] for index, sequence in enumerate(sequences)]
    suffix_lengths = [len(sequence) for sequence in suffixes]
    timings = {'prepare_seconds': time.perf_counter() - started}
    phase = time.perf_counter()
    tiles = []
    memory_checkpoints = {'before_execution': {'allocated_bytes': torch.cuda.memory_allocated(device),
                                               'reserved_bytes': torch.cuda.memory_reserved(device)}}
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    if mode in EXECUTION_POLICY['structured']['prefix_batch_modes']:
        prefix_ids, prefix_mask = padded(prefixes, tokenizer.pad_token_id, device, 'left')
        branches = torch.arange(batch, device=device).repeat_interleave(questions)
        suffix_ids, suffix_mask = padded(suffixes, tokenizer.pad_token_id, device, 'right')
        computed_tokens = sum(prefix_lengths) + sum(suffix_lengths)
        padded_tokens = prefix_ids.numel() + suffix_ids.numel()
        prefix_padded, suffix_padded = prefix_ids.numel(), suffix_ids.numel()
    if mode in EXECUTION_POLICY['structured']['tiled_modes']:
        tile_size = backend.config['branch_batch_size']
        assert tile_size > 0
        hidden_tiles = []
        if mode == 'tiled_shared':
            prefix_ids, prefix_mask = padded(prefixes, tokenizer.pad_token_id, device, 'left')
            prefix_computed_tokens, prefix_computed_padded = 0, 0
            base_prefix_hit = None

            def prefill_prefix():
                nonlocal prefix_computed_tokens, prefix_computed_padded, base_prefix_hit
                if base_prefix_cache is not None:
                    root_sequence, = prefixes
                    assert 0 < base_prefix_length < len(root_sequence)
                    root_ids = prefix_ids[:, :base_prefix_length]
                    root_mask = prefix_mask[:, :base_prefix_length]

                    def prefill_root():
                        return backend.model.model(input_ids=root_ids, attention_mask=root_mask,
                            position_ids=(root_mask.cumsum(-1) - 1).clamp_min(0),
                            use_cache=True).past_key_values

                    root_state, base_prefix_hit = base_prefix_cache.get(
                        [root_sequence[:base_prefix_length]], prefill_root)
                    extension = prefix_ids[:, base_prefix_length:]
                    positions = (prefix_mask.cumsum(-1) - 1).clamp_min(0)[:, base_prefix_length:]
                    output = backend.model.model(input_ids=extension, attention_mask=prefix_mask,
                        position_ids=positions, past_key_values=deepcopy(root_state), use_cache=True)
                    prefix_computed_tokens = len(root_sequence) - (base_prefix_length if base_prefix_hit else 0)
                    prefix_computed_padded = prefix_computed_tokens
                    return output.past_key_values
                output = backend.model.model(
                    input_ids=prefix_ids, attention_mask=prefix_mask,
                    position_ids=(prefix_mask.cumsum(-1) - 1).clamp_min(0), use_cache=True,
                )
                prefix_computed_tokens = sum(prefix_lengths)
                prefix_computed_padded = prefix_ids.numel()
                return output.past_key_values

            if prefix_cache is None:
                prefix_state, prefix_hit = prefill_prefix(), False
            else:
                prefix_state, prefix_hit = prefix_cache.get(prefixes, prefill_prefix)
            computed_tokens = prefix_computed_tokens + sum(suffix_lengths)
            prefix_padded, suffix_padded = prefix_computed_padded, 0
            padded_tokens = prefix_padded
            torch.cuda.synchronize(device)
            timings['prefix_seconds'] = time.perf_counter() - phase
            memory_checkpoints['after_prefix'] = {'allocated_bytes': torch.cuda.memory_allocated(device),
                                                   'reserved_bytes': torch.cuda.memory_reserved(device)}
        else:
            computed_tokens, padded_tokens = sum(lengths), 0
            prefix_padded, suffix_padded = None, None
        phase = time.perf_counter()
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        for start in range(0, len(sequences), tile_size):
            stop = min(start + tile_size, len(sequences))
            allocated_before = torch.cuda.memory_allocated(device)
            reserved_before = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            if mode == 'tiled_shared':
                ids, mask = padded(suffixes[start:stop], tokenizer.pad_token_id, device, 'right')
                branches = torch.arange(start, stop, device=device) // questions
                # Qwen attention, recurrent and convolution states all belong to this fork.
                fork = deepcopy(prefix_state)
                fork.reorder_cache(branches)
                full_mask = torch.cat([prefix_mask.index_select(0, branches), mask], dim=1)
                positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
                output = backend.model.model(input_ids=ids, attention_mask=full_mask,
                                             position_ids=positions, past_key_values=fork, use_cache=True)
                ends = torch.tensor(suffix_lengths[start:stop], device=device) - 1
                final = output.last_hidden_state[torch.arange(stop - start, device=device), ends]
                suffix_padded += ids.numel()
                del fork, full_mask, positions, branches, ends
            else:
                ids, mask = padded(sequences[start:stop], tokenizer.pad_token_id, device, 'left')
                output = backend.model.model(input_ids=ids, attention_mask=mask,
                                             position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)
                final = output.last_hidden_state[:, -1]
            # A view of the last position would otherwise retain every token activation.
            hidden_tiles.append(final.clone())
            padded_tokens += ids.numel()
            token_width = ids.shape[1]
            del final, output, ids, mask
            tile_peak_allocated = torch.cuda.max_memory_allocated(device)
            tile_peak_reserved = torch.cuda.max_memory_reserved(device)
            peak_allocated = max(peak_allocated, tile_peak_allocated)
            peak_reserved = max(peak_reserved, tile_peak_reserved)
            tiles.append({'start': start, 'stop': stop, 'field_batch_size': stop - start,
                          'token_width': token_width,
                          'allocated_before_bytes': allocated_before, 'reserved_before_bytes': reserved_before,
                          'peak_allocated_bytes': tile_peak_allocated, 'peak_reserved_bytes': tile_peak_reserved,
                          'allocated_after_release_bytes': torch.cuda.memory_allocated(device),
                          'reserved_after_release_bytes': torch.cuda.memory_reserved(device)})
        if mode == 'tiled_shared':
            del prefix_state, prefix_ids, prefix_mask
        hidden = torch.cat(hidden_tiles, dim=0)
        del hidden_tiles
        torch.cuda.synchronize(device)
        timings['tiles_seconds'] = time.perf_counter() - phase
    elif mode == 'streamed':
        branch_batch_size = backend.config['branch_batch_size']
        assert branch_batch_size > 0
        output = streamed_hidden(backend.model, prefix_ids, prefix_mask, suffix_ids, suffix_mask,
                                 branches, branch_batch_size)
        ends = torch.tensor(suffix_lengths, device=device) - 1
        hidden = output[torch.arange(batch * questions, device=device), ends]
        del output
        torch.cuda.synchronize(device)
        timings['streamed_layers_seconds'] = time.perf_counter() - phase
    elif mode == 'shared':
        output = backend.model.model(
            input_ids=prefix_ids, attention_mask=prefix_mask,
            position_ids=(prefix_mask.cumsum(-1) - 1).clamp_min(0), use_cache=True,
        )
        cache = output.past_key_values
        del output
        torch.cuda.synchronize(device)
        timings['prefix_seconds'] = time.perf_counter() - phase
        phase = time.perf_counter()
        cache.reorder_cache(branches)
        mask = torch.cat([prefix_mask.index_select(0, branches), suffix_mask], dim=1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)[:, -suffix_ids.shape[1]:]
        torch.cuda.synchronize(device)
        timings['branch_setup_seconds'] = time.perf_counter() - phase
        phase = time.perf_counter()
        output = backend.model.model(
            input_ids=suffix_ids, attention_mask=mask, position_ids=positions,
            past_key_values=cache, use_cache=True,
        )
        ends = torch.tensor(suffix_lengths, device=device) - 1
        hidden = output.last_hidden_state[torch.arange(batch * questions, device=device), ends]
        del output, cache
        torch.cuda.synchronize(device)
        timings['suffix_seconds'] = time.perf_counter() - phase
    else:
        ids, mask = padded(sequences, tokenizer.pad_token_id, device, 'left')
        output = backend.model.model(
            input_ids=ids, attention_mask=mask,
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False,
        )
        hidden = output.last_hidden_state[:, -1]
        computed_tokens, padded_tokens = sum(lengths), ids.numel()
        prefix_padded, suffix_padded = None, None
        del output
        torch.cuda.synchronize(device)
        timings['independent_seconds'] = time.perf_counter() - phase
    phase = time.perf_counter()
    # These are frozen vocabulary rows, not an added or trained classifier.
    weight = backend.finite_output_weights[:len(labels)]
    logits = F.linear(hidden.float(), weight).reshape(batch, questions, len(labels))
    valid = torch.arange(len(labels), device=device)[None, :] < torch.tensor(counts, device=device)[:, None]
    masked = logits.masked_fill(~valid[None], -torch.inf)
    probabilities = masked.softmax(dim=-1)
    chosen = masked.argmax(dim=-1).tolist()
    probability_rows, logit_rows = probabilities.tolist(), logits.tolist()
    answers = {}
    for index, (name, field, count) in enumerate(zip(names, definitions, counts)):
        answers[name] = {
            'choices': [field['options'][row[index]]['id'] for row in chosen],
            'probabilities': [row[index][:count] for row in probability_rows],
            'option_logits': [row[index][:count] for row in logit_rows],
            'option_ids': [option['id'] for option in field['options']],
            'input_tokens': lengths[index::questions], 'batch_size': batch,
        }
    torch.cuda.synchronize(device)
    timings['readout_seconds'] = time.perf_counter() - phase
    return {
        'fields': answers, 'mode': mode, 'batch_size': batch, 'field_count': questions,
        'elapsed_seconds': time.perf_counter() - started, 'timings': timings,
        'input_tokens': [sum(lengths[i:i + questions]) for i in range(0, len(lengths), questions)],
        'logical_input_tokens': sum(lengths), 'computed_input_tokens': computed_tokens,
        'padded_input_tokens': padded_tokens, 'prefix_tokens': prefix_lengths,
        'suffix_tokens': [suffix_lengths[i:i + questions] for i in range(0, len(suffix_lengths), questions)],
        'prefix_padded_tokens': prefix_padded, 'suffix_padded_tokens': suffix_padded,
        'persistent_prefix_hit': prefix_hit if mode == 'tiled_shared' else False,
        'base_prefix_hit': base_prefix_hit if mode == 'tiled_shared' else None,
        'base_prefix_length': base_prefix_length,
        'tiles': tiles,
        'memory_checkpoints': memory_checkpoints,
        'output_tokens': [0] * batch,
        'peak_cuda_memory_bytes': max(peak_allocated, torch.cuda.max_memory_allocated(device)),
        'peak_cuda_reserved_bytes': max(peak_reserved, torch.cuda.max_memory_reserved(device)),
    }
