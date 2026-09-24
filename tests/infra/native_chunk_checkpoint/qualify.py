import argparse
from importlib import import_module
import json
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.infra.qwen35 import gdn
from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache
from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from tests.infra.native_chunk_checkpoint.continuation import qualify as qualify_continuation
from tests.infra.native_chunk_checkpoint.integration import CheckpointRecorder
from tests.infra.native_suffix_graph.qualify import compare, measure, next_logits, tensors, wrapper


FLA_CHUNK = import_module('fla.ops.gated_delta_rule.chunk')


def match_requests(recorded, following, chunk_size):
    matches, offsets, evidence = [], [], []
    for row, request in enumerate(recorded):
        candidates = [(index, other) for index, other in enumerate(following)
                      if other['task_id'] == request['task_id']]
        assert len(candidates) <= 1
        if not candidates:
            offsets.append(None)
            continue
        index, other = candidates[0]
        left, right = request['input_ids'], other['input_ids']
        common = next((position for position, pair in enumerate(zip(left, right))
                       if pair[0] != pair[1]), min(len(left), len(right)))
        root = len(request['root_tokens'])
        suffix_length = len(left) - root - 1
        offset = min(common - root, suffix_length - 1) // chunk_size * chunk_size
        assert 0 < offset < suffix_length
        assert left[:root + offset] == right[:root + offset]
        offsets.append(offset)
        matches.append(other)
        evidence.append({'source_row': row, 'following_row': index, 'task_id': request['task_id'],
                         'common_prefix_tokens': common, 'root_tokens': root, 'offset': offset})
    return matches, offsets, evidence


@torch.inference_mode()
def qualify(backend, shared, settings, case):
    recorded = json.loads(Path(case['source']).read_text())['requests']
    following = json.loads(Path(case['following']).read_text())['requests']
    for request in [*recorded, *following]:
        rendered = backend.tokenizer.apply_chat_template(request['messages'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        assert backend.tokenizer(rendered, add_special_tokens=False)['input_ids'] == request['input_ids']
    matched, offsets, matching = match_requests(recorded, following, settings['chunk_size'])
    roots = [request['root_tokens'] for request in recorded]
    sequences = [request['input_ids'] for request in recorded]
    assert all(sequence[:len(root)] == root for sequence, root in zip(sequences, roots, strict=True))
    tails = [sequence[len(root):-1] for sequence, root in zip(sequences, roots, strict=True)]
    assert all(tails)
    root_lengths, lengths = list(map(len, roots)), list(map(len, tails))
    root_ids, root_mask = padded(roots, backend.tokenizer.pad_token_id, backend.device, 'left')
    root_output = backend.model.model(input_ids=root_ids, attention_mask=root_mask,
        position_ids=(root_mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
    root_states = split_native_cache(root_output.past_key_values, root_lengths)
    immutable, prefix_mask = pack_native_caches(root_states)
    static, static_mask = pack_native_caches(root_states)
    assert torch.equal(prefix_mask, static_mask)
    source_snapshot = [tensor.clone() for tensor in tensors(immutable)]
    reset_pairs = list(zip(tensors(static), tensors(immutable), strict=True))
    assert not {tensor.data_ptr() for tensor in tensors(static)} & {tensor.data_ptr() for tensor in tensors(immutable)}
    copy_settings = json.loads(Path(settings['state_copy_settings']).read_text())
    ids, mask = padded(tails, backend.tokenizer.pad_token_id, backend.device, 'right')
    full_mask = torch.cat((prefix_mask, mask), dim=-1)
    positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
    descriptor = RaggedSuffix(lengths, backend.device)
    attention = RaggedCacheAttention(descriptor, root_lengths)
    recorder = CheckpointRecorder(lengths, offsets, backend.device, {
        'chunk_size': settings['chunk_size'],
        'capture': json.loads(Path(settings['capture_settings']).read_text())})
    original_forward = gdn.ragged_gdn_forward
    original_capture = FLA_CHUNK.chunk_gated_delta_rule_fwd_h

    def forward():
        copy_states(reset_pairs, copy_settings)
        return backend.model.model(input_ids=ids, attention_mask=full_mask,
            position_ids=positions, past_key_values=wrapper(static), use_cache=True,
            ragged_suffix=descriptor, decode_attention=attention)

    def captured():
        gdn.ragged_gdn_forward = recorder.patched_forward
        FLA_CHUNK.chunk_gated_delta_rule_fwd_h = recorder.capture_native
        output = forward()
        gdn.ragged_gdn_forward = original_forward
        FLA_CHUNK.chunk_gated_delta_rule_fwd_h = original_capture
        snapshots = recorder.compact(output.past_key_values, root_lengths, prefix_mask.shape[1], copy_settings)
        return output, snapshots

    reference = forward()
    reference_hidden = reference.last_hidden_state.clone()
    reference_cache = [tensor.clone() for tensor in tensors(reference.past_key_values)]
    reference_logits = next_logits(backend, reference.past_key_values, root_lengths, lengths,
                                   prefix_mask.shape[1], sequences)
    candidate, checkpoint_states = captured()
    candidate_logits = next_logits(backend, candidate.past_key_values, root_lengths, lengths,
                                   prefix_mask.shape[1], sequences)
    counts = torch.tensor([len(request['field']['options']) for request in recorded], device=backend.device)
    valid = torch.arange(reference_logits.shape[1], device=backend.device)[None] < counts[:, None]
    checks = {
        'hidden': compare([reference_hidden], [candidate.last_hidden_state]),
        'cache': compare(reference_cache, tensors(candidate.past_key_values)),
        'next_logits': compare([reference_logits], [candidate_logits]),
        'next_choices_equal': torch.equal(reference_logits.masked_fill(~valid, -torch.inf).argmax(-1),
                                         candidate_logits.masked_fill(~valid, -torch.inf).argmax(-1)),
        'source_unchanged': compare(source_snapshot, tensors(immutable))}
    passed = all(checks[name]['equal'] for name in ('hidden', 'cache', 'next_logits', 'source_unchanged'))
    passed = passed and checks['next_choices_equal']
    agreement = torch.tensor(passed, dtype=torch.int32, device=backend.device)
    dist.all_reduce(agreement, op=dist.ReduceOp.MIN)
    passed = bool(agreement.item())
    measurements = {}
    if passed:
        for name, operation in [('reference', forward), ('capture_and_compact', captured)]:
            for _ in range(shared.runtime.graph_warmup_steps):
                operation()
            torch.cuda.reset_peak_memory_stats(backend.device)
            samples = measure(operation, backend.device, settings['repetitions'])
            measurements[name] = {'samples': samples,
                'median_wall_seconds': median(sample['wall_seconds'] for sample in samples),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device)}
    row = {'case': case, 'batch_size': len(recorded), 'task_ids': [request['task_id'] for request in recorded],
        'root_lengths': root_lengths, 'suffix_lengths': lengths, 'input_shape': list(ids.shape),
        'logical_suffix_tokens': sum(lengths), 'physical_suffix_tokens': ids.numel(),
        'checkpoint_rows': recorder.rows, 'checkpoint_offsets': offsets, 'following_matches': matching,
        'checkpoint_cache_lengths': [state.get_seq_length() for state in checkpoint_states],
        'checks': checks, 'exact_forward_passed': passed, 'measurements': measurements,
        'scope': 'Actual complete recorded B1/B8 suffixes, with source-cache restoration included. Capture timing includes compact checkpoint copies. Root prefill and next-token validation are excluded. No continuation or complete-task result is claimed.'}
    context = {'recorded': recorded, 'matched': matched, 'root_states': root_states,
               'checkpoint_states': checkpoint_states, 'offsets': offsets}
    return row, context


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(
        shared, json.loads(Path(settings['parallel_settings']).read_text()))
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    rows, contexts = [], []
    for case in settings['cases']:
        row, context = qualify(backend, shared, settings, case)
        rows.append(row)
        contexts.append(context)
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(
            {'settings': settings, 'startup': startup, 'rows': rows,
             'all_exact_so_far': all(item['exact_forward_passed'] for item in rows)}, indent=2) + '\n')
    if all(row['exact_forward_passed'] for row in rows):
        for row, context in zip(rows, contexts, strict=True):
            row['continuation'] = qualify_continuation(backend, shared, settings, **context)
            (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(
                {'settings': settings, 'startup': startup, 'rows': rows,
                 'all_exact_so_far': True}, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
