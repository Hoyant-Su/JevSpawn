import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from tests.infra.b1_decode_readout import ReadoutDecode, timed


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['source_inputs']).read_text())
    tokens = source['input_ids'][settings['source_row']]
    field = source['fields'][settings['source_row']]
    backend = Backend(shared.backend())
    ids = torch.tensor([tokens], device=backend.device)
    capacity = len(tokens) + shared.generation.max_new_tokens
    decoder = ReadoutDecode(backend, settings['batch_size'], capacity)
    decoder.prefill({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
    labels = torch.tensor(backend.answer_label_ids[:len(field['options'])], device=backend.device)
    decoder.weights = backend.model.lm_head.weight.index_select(0, labels)
    decoder.hidden = torch.empty((settings['batch_size'], decoder.weights.shape[1]),
                                device=backend.device, dtype=decoder.weights.dtype)
    decoder.candidates = torch.empty((settings['batch_size'], len(labels)),
                                    device=backend.device, dtype=torch.float32)
    snapshot = decoder.capture_snapshot()
    decoder.step()
    expected = decoder.candidates.clone()
    decoder.restore_capture_snapshot(snapshot)
    decoder.capture(shared.runtime.graph_warmup_steps)
    decoder.graph.replay()
    assert torch.equal(expected, decoder.candidates)
    controls = []
    for _ in range(settings['control_repeats']):
        decoder.restore_capture_snapshot(snapshot)
        controls.append(timed(decoder.graph.replay, backend.device))
    decoder.restore_capture_snapshot(snapshot)
    torch.cuda.synchronize(backend.device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.profiler.record_function('finite_decode_graph'):
            decoder.graph.replay()
            torch.cuda.synchronize(backend.device)
    profile.export_chrome_trace(settings['trace'])
    trace = json.loads(Path(settings['trace']).read_text())
    kernels = [event for event in trace['traceEvents'] if event.get('cat') == 'kernel']
    grouped = defaultdict(list)
    for kernel in kernels:
        grouped[kernel['name']].append(kernel['dur'])
    rows = [{'kernel': name, 'count': len(durations), 'total_us': sum(durations),
             'median_us': statistics.median(durations)} for name, durations in grouped.items()]
    rows.sort(key=lambda row: row['total_us'], reverse=True)
    elapsed = max(event['ts'] + event['dur'] for event in kernels) - min(event['ts'] for event in kernels)
    result = {'settings': settings, 'input_tokens': len(tokens), 'cache_capacity': capacity,
              'batch_size': settings['batch_size'], 'candidate_count': len(labels),
              'eager_graph_exact': True, 'control': controls,
              'control_median_ms': statistics.median(row['cuda_event_ms'] for row in controls),
              'kernel_count': len(kernels), 'kernel_sum_us': sum(event['dur'] for event in kernels),
              'kernel_span_us': elapsed, 'kernels': rows}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('control', 'kernels')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
