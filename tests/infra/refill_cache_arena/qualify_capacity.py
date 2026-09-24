import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.refill_cache_arena import RefillCacheArena
from jev_spawn.runtime.refill_state import RefillState


@torch.inference_mode()
def qualify(specification):
    settings = json.loads(specification.read_text())
    shared = SharedConfig.load(settings['shared_config'])
    assert settings['input_tokens'] == shared.model.max_input_tokens
    assert settings['generation_tokens'] == shared.generation.max_new_tokens
    assert settings['construction'] == 'repeat_recorded_prompt_tokens_to_explicit_width'
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(settings, indent=2) + '\n')
    backend = Backend(shared.backend())
    torch.cuda.reset_peak_memory_stats(backend.device)
    phases = []
    started = time.perf_counter()

    def record(phase, **measurements):
        torch.cuda.synchronize(backend.device)
        phases.append({'phase': phase, 'elapsed_seconds': time.perf_counter() - started,
                       'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved(),
                       'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'peak_reserved_bytes': torch.cuda.max_memory_reserved(), **measurements})
        (output / 'phases.json').write_text(json.dumps(phases, indent=2) + '\n')
        print(json.dumps(phases[-1]), flush=True)

    capacity = settings['input_tokens'] + settings['generation_tokens']
    arena = RefillCacheArena(backend.model.config, shared.runtime.batch_size, capacity,
                             backend.model.lm_head.weight.dtype, backend.device)
    state = RefillState(arena, settings['generation_tokens'], backend.tokenizer.pad_token_id)
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=backend.device)
    record('arena_allocated', arena_and_request_state_bytes=state.nbytes,
           compaction_tensor_scratch_bytes=arena.compaction_scratch_bytes)
    source = json.loads(Path(settings['source_batches']).read_text())[settings['source_batch']]['messages']
    assert len(source) == shared.runtime.batch_size
    text = backend.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = backend.tokenizer(text, add_special_tokens=False)['input_ids']
    width = settings['input_tokens']
    ids = torch.tensor([(tokens * ((width + len(tokens) - 1) // len(tokens)))[:width]
                        for tokens in encoded], device=backend.device, dtype=torch.long)

    def prefill(indices, label):
        selected = ids.index_select(0, torch.tensor(indices, device=backend.device))
        requests = [SimpleNamespace(task_id=f'{label}/{index}', max_tokens=settings['generation_tokens'], stop=())
                    for index in indices]
        ends = [time.perf_counter() + settings['request_deadline_seconds'] for _ in requests]
        before = time.perf_counter()
        decoder, rows = state.prefill(backend, requests, ends,
            {'input_ids': selected, 'attention_mask': torch.ones_like(selected)}, pool, stream)
        torch.cuda.synchronize(backend.device)
        record(label, input_shape=list(selected.shape), input_tokens=selected.numel(),
               prefill_seconds=time.perf_counter() - before, scalar_view_metadata_bytes=rows.scalar_metadata_bytes)
        return decoder, rows

    prefill(list(range(shared.runtime.batch_size)), 'initial_prefill_only_probe')
    state.compact(settings['retained_rows'])
    record('survivors_compacted', retained_rows=settings['retained_rows'], live_rows=arena.live_count)
    prefill(settings['new_source_rows'], 'newcomer_prefill')
    assert arena.live_count == shared.runtime.batch_size
    decoder = state.decode(backend, pool, stream)
    before = time.perf_counter()
    decoder.capture(shared.runtime.graph_warmup_steps)
    record('graph_captured', capture_seconds=time.perf_counter() - before,
           capture_snapshot=decoder.capture_memory, decode_input_shape=list(decoder.ids.shape))
    events = []
    for step in range(settings['generation_tokens']):
        if step:
            decoder.graph.replay()
        state.emit(0, arena.live_count)
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        events.append(event)
        if (step + 1) % settings['progress_interval'] == 0:
            record('decode_progress', emitted_tokens_per_row=step + 1,
                   real_graph_replays=step, executed_token_slots=(step + 1) * arena.live_count)
    torch.cuda.synchronize(backend.device)
    assert bool((state.counts == settings['generation_tokens']).all())
    intervals = [left.elapsed_time(right) for left, right in zip(events, events[1:])]
    result = {'passed': True, 'model': shared.model.path, 'scope': settings['scope'],
              'generated_shape': list(state.history.shape), 'generated_tokens': int(state.counts.sum()),
              'real_graph_replays': len(events) - 1, 'input_tokens_per_row': width,
              'arena_and_request_state_bytes': state.nbytes, 'capture_snapshot': decoder.capture_memory,
              'compaction_tensor_scratch_bytes': arena.compaction_scratch_bytes,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'replay_inter_token_ms_mean': sum(intervals) / len(intervals),
              'replay_inter_token_ms_median': sorted(intervals)[len(intervals) // 2],
              'timing_scope': 'Intervals include token bookkeeping and configured progress synchronization; initial prefill and capture are reported separately.',
              'elapsed_seconds': time.perf_counter() - started}
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    qualify(parser.parse_args().specification)
