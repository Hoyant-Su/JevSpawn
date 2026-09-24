from statistics import median

import torch

from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache_at
from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from tests.infra.native_suffix_graph.qualify import compare, measure, next_logits, tensors


def qualify(backend, shared, settings, recorded, matched, root_states, checkpoint_states, offsets):
    rows = [row for row, offset in enumerate(offsets) if offset is not None]
    assert len(rows) == len(matched) == len(checkpoint_states)
    originals = [root_states[row] for row in rows]
    sequences = [request['input_ids'] for request in matched]
    for row, request, snapshot in zip(rows, matched, checkpoint_states, strict=True):
        previous = recorded[row]
        boundary = len(previous['root_tokens']) + offsets[row]
        assert previous['task_id'] == request['task_id']
        assert previous['input_ids'][:boundary] == request['input_ids'][:boundary]
        assert snapshot.get_seq_length() == boundary
        rendered = backend.tokenizer.apply_chat_template(
            request['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert backend.tokenizer(rendered, add_special_tokens=False)['input_ids'] == request['input_ids']
    source_tensors = [tensor for cache in [*originals, *checkpoint_states] for tensor in tensors(cache)]
    source_snapshot = [tensor.clone() for tensor in source_tensors]

    def operation(states):
        lengths = [state.get_seq_length() for state in states]
        tails = [sequence[length:-1] for sequence, length in zip(sequences, lengths, strict=True)]
        assert all(tails)
        cache, prefix_mask = pack_native_caches(states)
        ids, mask = padded(tails, backend.tokenizer.pad_token_id, backend.device, 'right')
        full_mask = torch.cat((prefix_mask, mask), dim=-1)
        positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        suffix_lengths = list(map(len, tails))
        descriptor = RaggedSuffix(suffix_lengths, backend.device)
        attention = RaggedCacheAttention(descriptor, lengths)
        output = backend.model.model(input_ids=ids, attention_mask=full_mask,
            position_ids=positions, past_key_values=cache, use_cache=True,
            ragged_suffix=descriptor, decode_attention=attention)
        final_lengths = [prefix + suffix for prefix, suffix in zip(lengths, suffix_lengths, strict=True)]
        stops = [prefix_mask.shape[1] + length for length in suffix_lengths]
        compact = split_native_cache_at(output.past_key_values, final_lengths, stops)
        logits = next_logits(backend, output.past_key_values, lengths, suffix_lengths,
                             prefix_mask.shape[1], sequences)
        return logits, compact

    outputs, timing = {}, {}
    for name, states in zip(settings['arms'], (originals, checkpoint_states), strict=True):
        logits, caches = operation(states)
        outputs[name] = {'logits': logits, 'cache': [tensor for cache in caches for tensor in tensors(cache)]}
        for _ in range(shared.runtime.graph_warmup_steps):
            operation(states)
        timing[name] = measure(lambda: operation(states), backend.device, settings['repetitions'])
    reference, retained = (outputs[name] for name in settings['arms'])
    counts = torch.tensor([len(request['field']['options']) for request in matched], device=backend.device)
    valid = torch.arange(reference['logits'].shape[1], device=backend.device)[None] < counts[:, None]
    choices = {name: output['logits'].masked_fill(~valid, -torch.inf).argmax(-1).tolist()
               for name, output in outputs.items()}
    medians = {name: median(row['wall_seconds'] for row in samples) for name, samples in timing.items()}
    checks = {'logits': compare([reference['logits']], [retained['logits']]),
              'final_cache': compare(reference['cache'], retained['cache']),
              'choices_equal': choices[settings['arms'][0]] == choices[settings['arms'][1]],
              'source_unchanged': compare(source_snapshot, source_tensors)}
    return {'batch_size': len(matched), 'task_ids': [request['task_id'] for request in matched],
        'source_rows': rows, 'checkpoint_offsets': [offsets[row] for row in rows],
        'logical_suffix_tokens': {name: sum(len(sequence) - state.get_seq_length() - 1
            for sequence, state in zip(sequences, states, strict=True))
            for name, states in zip(settings['arms'], (originals, checkpoint_states), strict=True)},
        'checks': checks, 'choices': choices,
        'selected_logits': {name: output['logits'].tolist() for name, output in outputs.items()},
        'timings': timing, 'median_wall_seconds': medians,
        'speedup': medians[settings['arms'][0]] / medians[settings['arms'][1]],
        'scope': 'Actual later recorded decisions using identical prompts, roots, model and domains. '
                 'Both arms include cache packing, native suffix forward, compact final states and next-token readout. '
                 'Initial checkpoint capture cost is reported separately; this is not full-rollout speed.'}
