import argparse
from copy import copy
import json
import math
from pathlib import Path
from statistics import median
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers.cache_utils import DynamicLayer

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import initialize_parallel
from jev_spawn.algo.structured import padded
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.runtime.batched_state_copy import copy_states
from jev_spawn.runtime.native_cache_batch import (
    _metadata, _with_layers, pack_native_caches, split_native_cache,
    split_native_cache_at,
)
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


def tensors(cache):
    return [tensor for layer in cache.layers for tensor in
            ([layer.keys, layer.values] if type(layer) is DynamicLayer else
             [*layer.conv_states.values(), *layer.recurrent_states.values()])]


def wrapper(source):
    layers = []
    for original in source.layers:
        names = ('keys', 'values') if type(original) is DynamicLayer else ('conv_states', 'recurrent_states')
        layer = copy(original)
        layer.__dict__ = _metadata(original, names)
        for name in names:
            value = getattr(original, name)
            setattr(layer, name, value if type(original) is DynamicLayer else dict(value))
        layers.append(layer)
    return _with_layers(source, layers)


def compare(left, right):
    pairs = list(zip(left, right, strict=True))
    return {'equal': all(torch.equal(a, b) for a, b in pairs),
            'max_absolute_error': max((a.float() - b.float()).abs().max().item() for a, b in pairs)}


def measure(operation, device, repetitions):
    rows = []
    for _ in range(repetitions):
        torch.cuda.synchronize(device)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.perf_counter()
        begin.record()
        output = operation()
        end.record()
        end.synchronize()
        rows.append({'wall_seconds': time.perf_counter() - started,
                     'cuda_seconds': begin.elapsed_time(end) / 1000})
        del output
    return rows


def next_logits(backend, cache, prefix_lengths, suffix_lengths, prefix_width, sequences):
    lengths = [a + b for a, b in zip(prefix_lengths, suffix_lengths, strict=True)]
    stops = [prefix_width + length for length in suffix_lengths]
    rows = split_native_cache_at(cache, lengths, stops)
    packed, mask = pack_native_caches(rows)
    ids = torch.tensor([sequence[-1] for sequence in sequences], device=backend.device)[:, None]
    output = backend.model.model(input_ids=ids, attention_mask=torch.cat((mask, torch.ones_like(ids)), dim=-1),
        position_ids=torch.tensor(lengths, device=backend.device)[:, None],
        past_key_values=packed, use_cache=True)
    return F.linear(output.last_hidden_state[:, -1].float(), backend.finite_output_weights)


@torch.inference_mode()
def qualify(backend, shared, settings, source, save_phase):
    requests = json.loads(Path(source).read_text())['requests']
    assert len({(request['task_id'], request['field']['context']) for request in requests}) == len(requests)
    roots = [request['root_tokens'] for request in requests]
    sequences = [request['input_ids'] for request in requests]
    assert all(sequence[:len(root)] == root for sequence, root in zip(sequences, roots, strict=True))
    tails = [sequence[len(root):-1] for sequence, root in zip(sequences, roots, strict=True)]
    assert all(tails)
    root_ids, root_mask = padded(roots, backend.tokenizer.pad_token_id, backend.device, 'left')
    root_output = backend.model.model(input_ids=root_ids, attention_mask=root_mask,
        position_ids=(root_mask.cumsum(-1) - 1).clamp_min(0), use_cache=True)
    root_states = split_native_cache(root_output.past_key_values, list(map(len, roots)))
    immutable, prefix_mask = pack_native_caches(root_states)
    static, static_mask = pack_native_caches(root_states)
    assert torch.equal(prefix_mask, static_mask)
    del root_output, root_states
    source_snapshot = [tensor.clone() for tensor in tensors(immutable)]
    pairs = list(zip(tensors(static), tensors(immutable), strict=True))
    assert not {tensor.data_ptr() for tensor in tensors(static)} & {tensor.data_ptr() for tensor in tensors(immutable)}
    copy_settings = json.loads(Path(settings['state_copy_settings']).read_text())
    ids, mask = padded(tails, backend.tokenizer.pad_token_id, backend.device, 'right')
    full_mask = torch.cat((prefix_mask, mask), dim=-1)
    positions = (full_mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
    lengths = list(map(len, tails))
    descriptor = RaggedSuffix(lengths, backend.device)
    attention = RaggedCacheAttention(descriptor, list(map(len, roots)))

    def reset():
        copy_states(pairs, copy_settings)

    def forward(cache):
        return backend.model.model(input_ids=ids, attention_mask=full_mask,
            position_ids=positions, past_key_values=cache, use_cache=True,
            ragged_suffix=descriptor, decode_attention=attention)

    def eager():
        reset()
        return forward(wrapper(static))

    reset()
    reference = forward(wrapper(static))
    reference_hidden = reference.last_hidden_state.clone()
    reference_cache = [tensor.clone() for tensor in tensors(reference.past_key_values)]
    reference_logits = next_logits(backend, reference.past_key_values, list(map(len, roots)),
                                   lengths, prefix_mask.shape[1], sequences)
    del reference
    stream = torch.cuda.Stream(device=backend.device)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(shared.runtime.graph_warmup_steps):
            output = eager()
            del output
        reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize(backend.device)
    save_phase(source, 'capturing', {'batch_size': len(requests), 'suffix_lengths': lengths})
    capture_cache = wrapper(static)
    graph = torch.cuda.CUDAGraph()
    before_capture = torch.cuda.memory_allocated(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    started = time.perf_counter()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward(capture_cache)
    torch.cuda.synchronize(backend.device)
    capture_seconds = time.perf_counter() - started
    capture_memory = {'before_allocated_bytes': before_capture,
                      'after_allocated_bytes': torch.cuda.memory_allocated(backend.device),
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
                      'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device)}

    def replay():
        reset()
        graph.replay()
        return captured

    replay()
    checks = {'hidden': compare([reference_hidden], [captured.last_hidden_state]),
              'cache': compare(reference_cache, tensors(captured.past_key_values))}
    replay_logits = next_logits(backend, captured.past_key_values, list(map(len, roots)),
                                lengths, prefix_mask.shape[1], sequences)
    counts = torch.tensor([len(request['field']['options']) for request in requests], device=backend.device)
    valid = torch.arange(reference_logits.shape[1], device=backend.device)[None] < counts[:, None]
    checks['next_logits'] = compare([reference_logits], [replay_logits])
    checks['next_choices_equal'] = torch.equal(reference_logits.masked_fill(~valid, -torch.inf).argmax(-1),
                                               replay_logits.masked_fill(~valid, -torch.inf).argmax(-1))
    checks['source_unchanged'] = compare(source_snapshot, tensors(immutable))
    save_phase(source, 'numerical_checks', checks)
    assert all(checks[name]['equal'] for name in ('hidden', 'cache', 'next_logits', 'source_unchanged'))
    assert checks['next_choices_equal']
    measurements = {}
    for name, operation in [('eager_reset_forward', eager), ('graph_reset_replay', replay)]:
        torch.cuda.reset_peak_memory_stats(backend.device)
        measurements[name] = {'samples': measure(operation, backend.device, settings['repetitions']),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device)}
    replay()
    checks['after_timing_cache'] = compare(reference_cache, tensors(captured.past_key_values))
    checks['after_timing_source_unchanged'] = compare(source_snapshot, tensors(immutable))
    eager_seconds = median(row['wall_seconds'] for row in measurements['eager_reset_forward']['samples'])
    replay_seconds = median(row['wall_seconds'] for row in measurements['graph_reset_replay']['samples'])
    saved_seconds = eager_seconds - replay_seconds
    row = {'source': source, 'task_ids': [request['task_id'] for request in requests],
        'batch_size': len(requests), 'root_lengths': list(map(len, roots)), 'suffix_lengths': lengths,
        'input_shape': list(ids.shape), 'mask_shape': list(full_mask.shape),
        'logical_suffix_tokens': sum(lengths), 'physical_suffix_tokens': ids.numel(),
        'checks': checks, 'measurements': measurements, 'capture_seconds': capture_seconds,
        'capture_memory': capture_memory, 'warmup_steps': shared.runtime.graph_warmup_steps,
        'median_wall_speedup': eager_seconds / replay_seconds,
        'capture_break_even_replays': math.ceil(capture_seconds / saved_seconds) if saved_seconds > 0 else None,
        'scope': 'Same real ragged suffix forward at fixed actual shape, with native input-cache restoration included. Root prefill, packing, splitting and next-token scoring are outside timing. Graph replay is not a generic variable-shape or end-to-end speedup.'}
    save_phase(source, 'complete', row)
    assert checks['after_timing_cache']['equal'] and checks['after_timing_source_unchanged']['equal']
    return row


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    backend, commands, startup = initialize_parallel(shared, json.loads(Path(settings['parallel_settings']).read_text()))
    destination = Path(settings['output'])
    destination.mkdir(parents=True, exist_ok=True)
    rows = []

    def save_phase(source, phase, evidence):
        (destination / f'rank-{dist.get_rank()}-progress.json').write_text(json.dumps(
            {'source': source, 'phase': phase, 'evidence': evidence}, indent=2) + '\n')

    for source in settings['sources']:
        rows.append(qualify(backend, shared, settings, source, save_phase))
        (destination / f'rank-{dist.get_rank()}.json').write_text(json.dumps(
            {'settings': settings, 'startup': startup, 'rows': rows}, indent=2) + '\n')
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
