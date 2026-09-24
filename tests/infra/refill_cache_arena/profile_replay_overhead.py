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
def run(specification, output):
    settings = json.loads(specification.read_text())
    shared = SharedConfig.load(settings['shared_config'])
    backend = Backend(shared.backend())
    source = json.loads(Path(settings['source_batches']).read_text())[settings['source_batch']]['messages']
    text = backend.tokenizer.apply_chat_template(source, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    encoded = backend.tokenizer(text, add_special_tokens=False)['input_ids']
    width = settings['input_tokens']
    ids = torch.tensor([(tokens * ((width + len(tokens) - 1) // len(tokens)))[:width]
                        for tokens in encoded], device=backend.device, dtype=torch.long)
    arena = RefillCacheArena(backend.model.config, shared.runtime.batch_size,
                             width + settings['generation_tokens'], backend.model.lm_head.weight.dtype,
                             backend.device)
    state = RefillState(arena, settings['generation_tokens'], backend.tokenizer.pad_token_id)
    requests = [SimpleNamespace(task_id=str(index), max_tokens=settings['generation_tokens'], stop=())
                for index in range(shared.runtime.batch_size)]
    ends = [time.perf_counter() + settings['deadline_seconds'] for _ in requests]
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=backend.device)
    decoder, rows = state.prefill(backend, requests, ends,
                                  {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, pool, stream)
    decoder.capture(shared.runtime.graph_warmup_steps)
    modes = settings['modes']
    measurements = []
    for mode in modes:
        state.counts.zero_()
        torch.cuda.synchronize(backend.device)
        start = time.perf_counter()
        events = []
        for _ in range(settings['profile_steps']):
            decoder.graph.replay()
            if mode in {'replay_emit', 'replay_emit_event'}:
                state.emit(0, arena.live_count)
            if mode in {'replay_event', 'replay_emit_event'}:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                events.append(event)
        torch.cuda.synchronize(backend.device)
        elapsed = time.perf_counter() - start
        measurements.append({'mode': mode, 'steps': settings['profile_steps'],
                             'wall_ms_per_step': elapsed * 1000 / settings['profile_steps'],
                             'cuda_event_ms_per_step': (sum(left.elapsed_time(right)
                                for left, right in zip(events[:-1], events[1:], strict=True)) /
                                len(events[1:])) if len(events) > 1 else None})
    output.write_text(json.dumps({'settings': settings, 'measurements': measurements}, indent=2) + '\n')
    print(json.dumps({'measurements': measurements}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.specification, args.output)
