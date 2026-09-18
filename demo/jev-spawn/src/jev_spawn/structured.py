import string
import time

import torch
import torch.nn.functional as F

from jev_spawn.schema import CONTROLLER, controller_prompts
from jev_spawn.streaming import streamed_hidden


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
def score_fields(backend, states, fields, mode):
    assert mode in {'shared', 'independent', 'streamed'}
    assert states and fields
    schemas = [fields] * len(states) if isinstance(fields, dict) else fields
    assert len(schemas) == len(states)
    fields = schemas[0]
    names, definitions = list(fields), list(fields.values())
    signature = lambda schema: [(name, [option['id'] for option in field['options']]) for name, field in schema.items()]
    assert all(signature(schema) == signature(fields) for schema in schemas)
    counts = [len(field['options']) for field in definitions]
    assert all(2 <= count <= len(string.ascii_uppercase) for count in counts)
    batch, questions = len(states), len(fields)
    device, tokenizer = backend.device, backend.tokenizer
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    labels = list(string.ascii_uppercase[:max(counts)])
    label_ids = tokenizer(labels, add_special_tokens=False)['input_ids']
    assert all(len(ids) == 1 for ids in label_ids), 'Answer labels must be single tokens.'
    assert len({ids[0] for ids in label_ids}) == len(labels), 'Answer labels must be distinct.'
    prompts = [
        controller_prompts([state], field['question'], field['options'], labels[:count],
                           CONTROLLER['output_instruction'])[0]
        for state, schema in zip(states, schemas) for field, count in zip(schema.values(), counts)
    ]
    rendered = backend._render(prompts, CONTROLLER['system'])
    sequences = tokenizer(rendered, add_special_tokens=False)['input_ids']
    lengths = [len(sequence) for sequence in sequences]
    assert max(lengths) <= backend.config['max_input_tokens'], 'Full field prompt exceeds max_input_tokens; no truncation.'
    extended = tokenizer(
        [text + label for index, text in enumerate(rendered) for label in labels[:counts[index % questions]]],
        add_special_tokens=False,
    )['input_ids']
    expected = [sequence + ids for index, sequence in enumerate(sequences)
                for ids in label_ids[:counts[index % questions]]]
    assert extended == expected, 'Answer-label tokenization changes at the prompt boundary.'
    prefix_lengths = [common_prefix(sequences[i:i + questions]) for i in range(0, len(sequences), questions)]
    assert min(prefix_lengths) > 0, 'Fields must share a nonempty exact token prefix.'
    prefixes = [sequences[i * questions][:length] for i, length in enumerate(prefix_lengths)]
    suffixes = [sequence[prefix_lengths[index // questions]:] for index, sequence in enumerate(sequences)]
    suffix_lengths = [len(sequence) for sequence in suffixes]
    timings = {'prepare_seconds': time.perf_counter() - started}
    phase = time.perf_counter()
    if mode in {'shared', 'streamed'}:
        prefix_ids, prefix_mask = padded(prefixes, tokenizer.pad_token_id, device, 'left')
        branches = torch.arange(batch, device=device).repeat_interleave(questions)
        suffix_ids, suffix_mask = padded(suffixes, tokenizer.pad_token_id, device, 'right')
        computed_tokens = sum(prefix_lengths) + sum(suffix_lengths)
        padded_tokens = prefix_ids.numel() + suffix_ids.numel()
        prefix_padded, suffix_padded = prefix_ids.numel(), suffix_ids.numel()
    if mode == 'streamed':
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
    weight = backend.model.lm_head.weight.index_select(0, torch.tensor([ids[0] for ids in label_ids], device=device))
    logits = F.linear(hidden.float(), weight.float()).reshape(batch, questions, len(labels))
    answers = {}
    for index, (name, field, count) in enumerate(zip(names, definitions, counts)):
        option_logits = logits[:, index, :count]
        probabilities = option_logits.softmax(dim=-1)
        chosen = probabilities.argmax(dim=-1).tolist()
        answers[name] = {
            'choices': [field['options'][choice]['id'] for choice in chosen],
            'probabilities': probabilities.tolist(), 'option_logits': option_logits.tolist(),
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
        'output_tokens': [0] * batch,
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(device),
    }
