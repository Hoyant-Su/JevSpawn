import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.decoding import CapturedDecode


class ReadoutDecode(CapturedDecode):
    def step(self):
        valid = self.key_valid & (self.key_positions <= self.cache.get_seq_length())
        output = self.trunk(input_ids=self.ids, position_ids=self.positions,
                            attention_mask={'full_attention': valid[:, None, None, :], 'linear_attention': None},
                            past_key_values=self.cache, use_cache=True)
        self.hidden.copy_(output.last_hidden_state[:, -1])
        self.candidates.copy_(F.linear(self.hidden.float(), self.weights.float()))


def timed(function, device):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize(device)
    before = time.perf_counter()
    start.record()
    function()
    end.record()
    end.synchronize()
    return {'wall_ms': (time.perf_counter() - before) * 1000,
            'cuda_event_ms': start.elapsed_time(end)}


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['source_inputs']).read_text())
    assert settings['batch_size'] == 1
    tokens = source['input_ids'][settings['source_row']]
    field = source['fields'][settings['source_row']]
    backend = Backend(shared.backend())
    ids = torch.tensor([tokens], device=backend.device)
    inputs = {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}
    capacity = len(tokens) + shared.generation.max_new_tokens
    assert settings['steps'] < shared.generation.max_new_tokens
    decoder = CapturedDecode(backend, settings['batch_size'], capacity)
    decoder.prefill(inputs)
    snapshot = decoder.capture_snapshot()
    decoder.step()
    reference_logits, reference_token = decoder.logits.clone(), decoder.ids.clone()
    decoder.restore_capture_snapshot(snapshot)
    decoder.capture(shared.runtime.graph_warmup_steps)
    decoder.graph.replay()
    assert torch.equal(reference_logits, decoder.logits)
    assert torch.equal(reference_token, decoder.ids)
    decoder.restore_capture_snapshot(snapshot)
    events = [torch.cuda.Event(enable_timing=True) for _ in range(settings['steps'] + 1)]
    torch.cuda.synchronize(backend.device)
    started = time.perf_counter()
    events[0].record()
    for index in range(settings['steps']):
        decoder.graph.replay()
        events[index + 1].record()
    torch.cuda.synchronize(backend.device)
    wall = time.perf_counter() - started
    intervals = [left.elapsed_time(right) for left, right in zip(events, events[1:])]
    decoder.restore_capture_snapshot(snapshot)
    candidate = ReadoutDecode(backend, settings['batch_size'], capacity)
    candidate.cache = decoder.cache
    candidate.ids.copy_(decoder.ids)
    candidate.positions.copy_(decoder.positions)
    candidate.key_valid.copy_(decoder.key_valid)
    label_ids = torch.tensor(backend.answer_label_ids[:len(field['options'])], device=backend.device)
    candidate.weights = backend.model.lm_head.weight.index_select(0, label_ids)
    candidate.hidden = torch.empty((settings['batch_size'], candidate.weights.shape[1]),
                                   device=backend.device, dtype=candidate.weights.dtype)
    candidate.candidates = torch.empty((settings['batch_size'], len(label_ids)),
                                      device=backend.device, dtype=torch.float32)
    candidate_snapshot = candidate.capture_snapshot()
    candidate.step()
    expected = candidate.candidates.clone()
    candidate.restore_capture_snapshot(candidate_snapshot)
    candidate.capture(shared.runtime.graph_warmup_steps)
    candidate.graph.replay()
    assert torch.equal(expected, candidate.candidates)
    candidate.restore_capture_snapshot(candidate_snapshot)
    rows = []
    for repeat in range(settings['paired_repeats']):
        decoder.restore_capture_snapshot(snapshot)
        full = timed(decoder.graph.replay, backend.device)
        candidate.restore_capture_snapshot(candidate_snapshot)
        finite = timed(candidate.graph.replay, backend.device)
        selected_logits = decoder.logits.index_select(-1, label_ids).float()
        rows.append({'repeat': repeat, 'full': full, 'finite': finite,
                     'same_candidate_choice': torch.equal(selected_logits.argmax(-1), candidate.candidates.argmax(-1)),
                     'max_candidate_logit_difference': (selected_logits - candidate.candidates).abs().max().item()})
    hidden = candidate.hidden.clone()
    readouts = []
    for repeat in range(settings['paired_repeats']):
        readouts.append({'full': timed(lambda: backend.model.lm_head(hidden), backend.device),
                         'finite': timed(lambda: F.linear(hidden.float(), candidate.weights.float()), backend.device)})
    result = {'settings': settings, 'backend': {k: backend.metadata[k] for k in
              ('model_path', 'dtype', 'attention', 'kernel', 'device')},
              'input_tokens': len(tokens), 'cache_capacity': capacity,
              'candidate_count': len(label_ids), 'batch_size': settings['batch_size'],
              'eager_graph_logits_and_token_exact': True,
              'finite_eager_graph_exact': True,
              'decode': {'steps': settings['steps'], 'wall_ms_per_step': wall * 1000 / settings['steps'],
                         'cuda_interval_ms': intervals, 'median_ms': statistics.median(intervals),
                         'max_ms': max(intervals)},
              'paired_steps': rows, 'readout_only': readouts}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'decode_median_ms': result['decode']['median_ms'],
                      'paired_full_ms': statistics.median(r['full']['cuda_event_ms'] for r in rows),
                      'paired_finite_ms': statistics.median(r['finite']['cuda_event_ms'] for r in rows),
                      'full_head_ms': statistics.median(r['full']['cuda_event_ms'] for r in readouts),
                      'finite_head_ms': statistics.median(r['finite']['cuda_event_ms'] for r in readouts)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
