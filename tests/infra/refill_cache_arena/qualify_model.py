import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch
from transformers.cache_utils import LinearAttentionCacheLayerMixin

from baselines.common.config import SharedConfig
from baselines.common.stops import StopCache
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.refill_cache_arena import RefillCacheArena
from jev_spawn.runtime.refill_state import RefillState
from jev_spawn.runtime.rolling_decode import RollingDecode


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def native_tensors(decoder):
    tensors = {name: getattr(decoder, name) for name in ('ids', 'positions', 'key_valid', 'logits')}
    for index, layer in enumerate(decoder.cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin):
            for name in ('conv_states', 'recurrent_states'):
                tensors.update({f'{index}/{name}/{key}': value for key, value in getattr(layer, name).items()})
        else:
            tensors.update({f'{index}/{name}': getattr(layer, name) for name in ('keys', 'values')})
            tensors[f'{index}/length'] = layer.cumulative_length.expand(decoder.ids.shape[0])
    return tensors


def compare(expected, actual, boundary, output):
    assert set(expected) == set(actual)
    for name in expected:
        left, right = expected[name], actual[name]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
            error = {'boundary': boundary, 'tensor': name, 'expected_shape': list(left.shape),
                     'actual_shape': list(right.shape), 'expected_dtype': str(left.dtype),
                     'actual_dtype': str(right.dtype)}
            if left.shape == right.shape:
                error['different_elements'] = int((left != right).sum())
                error['maximum_absolute_difference'] = float((left.float() - right.float()).abs().max())
            save(output / 'mismatch.json', error)
            raise AssertionError(f'Native state mismatch at {boundary}: {name}')


def capture(decoder, warmup, output, boundary):
    expected = {name: tensor.clone() for name, tensor in native_tensors(decoder).items()}
    decoder.capture(warmup)
    compare(expected, native_tensors(decoder), boundary, output)


def stop_causes(decoder, history, counts, requests, deadlines, now, backend, stop_cache, codes):
    tokens = decoder.ids[:, 0]
    eos = torch.isin(tokens, torch.tensor(backend.eos_ids, device=backend.device))
    stopped = torch.zeros_like(eos)
    for pattern in dict.fromkeys(request.stop for request in requests):
        if pattern:
            criteria = stop_cache.get(pattern)
            offsets = torch.arange(criteria.maximum_token_len, device=backend.device)
            positions = counts[:, None] - len(offsets) + offsets
            window = history.gather(1, positions.clamp_min(0))
            window.masked_fill_(positions < 0, backend.tokenizer.pad_token_id)
            mask = torch.tensor([request.stop == pattern for request in requests], device=backend.device)
            stopped |= criteria(window, None) & mask
    budgets = torch.tensor([request.max_tokens for request in requests], device=backend.device)
    expired = deadlines <= now
    exhausted = counts >= budgets
    causes = torch.where(expired, codes['deadline'], torch.where(eos | stopped, codes['stop'],
                         torch.where(exhausted, codes['budget'], codes['continue'])))
    return causes, {'eos': eos, 'stop': stopped, 'budget': exhausted, 'deadline': expired}


@torch.inference_mode()
def qualify(specification):
    settings = json.loads(specification.read_text())
    shared = SharedConfig.load(settings['shared_config'])
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads(Path(settings['source_batches']).read_text())[settings['source_batch']]
    assert shared.generation.temperature == 0
    assert len(source['messages']) == shared.runtime.batch_size
    save(output / 'protocol.json', {'specification': settings, 'shared_config_text': Path(settings['shared_config']).read_text(),
        'source_messages': source['messages'], 'scope': 'Real native model operator equivalence; no benchmark accuracy or isolated performance claim.'})
    backend = Backend(shared.backend())
    torch.cuda.reset_peak_memory_stats(backend.device)
    arena = RefillCacheArena(backend.model.config, shared.runtime.batch_size, settings['capacity'],
                             backend.model.lm_head.weight.dtype, backend.device)
    state = RefillState(arena, settings['max_new_tokens'], backend.tokenizer.pad_token_id)
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=backend.device)
    stop_cache = StopCache(backend.tokenizer, backend.device, shared.runtime.stop_engine)
    reference, candidate = None, None
    ref_history = state.history[:0].clone()
    ref_counts = state.counts[:0].clone()
    requests, ends, pending, boundaries, finished = [], [], list(settings['requests']), [], []
    events, refills = Counter(), 0
    started = time.perf_counter()
    while pending or requests:
        incoming_count = min(shared.runtime.batch_size - len(requests), len(pending))
        if incoming_count:
            selected, pending = pending[:incoming_count], pending[incoming_count:]
            incoming = [SimpleNamespace(task_id=row['request_id'], max_tokens=row['max_tokens'], stop=tuple(row['stop']))
                        for row in selected]
            now = time.perf_counter()
            incoming_ends = [now + row['deadline_seconds'] for row in selected]
            messages = [source['messages'][row['source_row']] for row in selected]
            text = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = backend.tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False).to(backend.device)
            assert inputs['input_ids'].shape[1] + max(request.max_tokens for request in incoming) <= settings['capacity']
            newcomer = CapturedDecode(backend, incoming_count, settings['capacity'], graph_pool=pool, graph_stream=stream)
            newcomer.prefill(inputs)
            actual, rows = state.prefill(backend, incoming, incoming_ends, inputs, pool, stream)
            compare(native_tensors(newcomer), native_tensors(actual), 'incoming_prefill', output)
            sources = [] if reference is None else [(reference, torch.arange(len(requests), device=backend.device))]
            sources.append((newcomer, torch.arange(incoming_count, device=backend.device)))
            merged = RollingDecode(backend, len(requests) + incoming_count, settings['capacity'], newcomer)
            merged.graph_pool, merged.graph_stream = pool, stream
            merged.load(sources)
            ref_history = torch.cat([ref_history, torch.full((incoming_count, settings['max_new_tokens']),
                                    backend.tokenizer.pad_token_id, dtype=torch.long, device=backend.device)])
            ref_counts = torch.cat([ref_counts, torch.zeros(incoming_count, dtype=torch.long, device=backend.device)])
            reference = merged
            refills += bool(requests)
            requests.extend(incoming)
            ends.extend(incoming_ends)
            candidate = state.decode(backend, pool, stream)
            compare(native_tensors(reference), native_tensors(candidate), 'refill', output)
            capture(reference, shared.runtime.graph_warmup_steps, output, 'reference_capture')
            capture(candidate, shared.runtime.graph_warmup_steps, output, 'candidate_capture')
            ref_history[rows.start:rows.stop].scatter_(1, ref_counts[rows.start:rows.stop, None],
                                                      reference.ids[rows.start:rows.stop])
            ref_counts[rows.start:rows.stop].add_(1)
            state.emit(rows.start, rows.stop)
            del newcomer, actual, inputs, merged, sources
        else:
            reference.graph.replay()
            candidate.graph.replay()
            compare(native_tensors(reference), native_tensors(candidate), 'decode_step', output)
            ref_history.scatter_(1, ref_counts[:, None], reference.ids)
            ref_counts.add_(1)
            state.emit(0, len(requests))
        compare({'history': ref_history, 'counts': ref_counts,
                 'budgets': torch.tensor([r.max_tokens for r in requests], device=backend.device),
                 'deadlines': torch.tensor(ends, device=backend.device, dtype=torch.float64)},
                {'history': state.history[:len(requests)], 'counts': state.counts[:len(requests)],
                 'budgets': state.budgets[:len(requests)], 'deadlines': state.deadlines[:len(requests)]}, 'request_alignment', output)
        assert all(left is right for left, right in zip(requests, state.requests, strict=True))
        now = time.perf_counter()
        expected_causes, observed = stop_causes(reference, ref_history, ref_counts, requests,
                                                torch.tensor(ends, device=backend.device, dtype=torch.float64),
                                                now, backend, stop_cache, settings['reason_codes'])
        actual_causes, _ = stop_causes(candidate, state.history[:len(requests)], state.counts[:len(requests)],
                                       state.requests, state.deadlines[:len(requests)], now,
                                       backend, stop_cache, settings['reason_codes'])
        compare({'causes': expected_causes}, {'causes': actual_causes}, 'termination_alignment', output)
        causes = expected_causes.tolist()
        keep = [index for index, cause in enumerate(causes) if cause == settings['reason_codes']['continue']]
        for index, cause in enumerate(causes):
            if cause != settings['reason_codes']['continue']:
                for name, values in observed.items():
                    events[name] += bool(values[index])
                finished.append({'request_id': requests[index].task_id, 'cause': cause,
                                 'token_ids': ref_history[index, :int(ref_counts[index])].tolist()})
        boundaries.append({'live_requests': [request.task_id for request in requests], 'causes': causes,
                           'exact_native_states_logits_tokens': True, 'elapsed_seconds': time.perf_counter() - started})
        save(output / 'progress.json', {'boundaries': boundaries, 'finished': finished, 'termination_events': dict(events)})
        if len(keep) != len(requests):
            state.compact(keep)
            indices = torch.tensor(keep, device=backend.device, dtype=torch.long)
            ref_history, ref_counts = ref_history.index_select(0, indices), ref_counts.index_select(0, indices)
            requests, ends = [requests[i] for i in keep], [ends[i] for i in keep]
            if keep:
                compacted = RollingDecode(backend, len(keep), settings['capacity'], reference)
                compacted.graph_pool, compacted.graph_stream = pool, stream
                compacted.load([(reference, indices)])
                reference = compacted
                del compacted
                candidate = state.decode(backend, pool, stream)
                compare(native_tensors(reference), native_tensors(candidate), 'departure', output)
                capture(reference, shared.runtime.graph_warmup_steps, output, 'departure_reference_capture')
                capture(candidate, shared.runtime.graph_warmup_steps, output, 'departure_candidate_capture')
            else:
                reference, candidate = None, None
    assert len(finished) == len(settings['requests'])
    assert refills >= settings['minimum_refills']
    assert all(events[name] for name in settings['required_termination_events'])
    result = {'passed': True, 'model': shared.model.path, 'requests': len(finished), 'refills': refills,
              'termination_events': dict(events), 'exact_native_states_logits_tokens': True,
              'arena_and_request_state_bytes': state.nbytes, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'elapsed_seconds': time.perf_counter() - started,
              'scope': 'Serialized reference/candidate diagnostic includes full-state GPU snapshots and comparisons. No throughput or full-capacity qualification claim.'}
    save(output / 'completion.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    qualify(parser.parse_args().specification)
