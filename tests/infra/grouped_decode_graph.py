import argparse
import json
from pathlib import Path
import statistics

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from baselines.common.config import SharedConfig
from jev_spawn.algo.structured import padded
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.cached_attention import GroupedDecodeAttention
from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.rolling_decode import RollingDecode
from tests.infra.b1_decode_readout import timed


def metadata(decoder):
    batch = decoder.ids.shape[0]
    lengths = decoder.cache.get_seq_length().expand(batch)
    positions = decoder.key_positions[None]
    leftpad = decoder.key_valid.to(torch.int32).argmax(-1).to(torch.int32)
    expected = (positions >= leftpad[:, None]) & (positions < lengths[:, None])
    actual = decoder.key_valid & (positions < lengths[:, None])
    assert torch.equal(expected, actual), 'Grouped cache attention requires a contiguous valid interval per row.'
    return leftpad, {'lengths': lengths.tolist(), 'positions': decoder.positions[:, 0].tolist(),
                     'leftpad': leftpad.tolist(), 'masks_equal': True}


def qualify(decoder, identities, backend, settings, shared):
    leftpad, initial = metadata(decoder)
    decoder.capture(shared.runtime.graph_warmup_steps)
    reference_graph = decoder.graph
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    candidate = GroupedDecodeAttention(decoder.cache, leftpad, settings['num_splits'])
    ALL_ATTENTION_FUNCTIONS.register('sdpa', candidate)
    try:
        before = decoder.capture_snapshot()
        decoder.step()
        eager_logits, eager_ids = decoder.logits.clone(), decoder.ids.clone()
        decoder.restore_capture_snapshot(before)
        decoder.capture(shared.runtime.graph_warmup_steps)
        decoder.graph.replay()
        assert torch.equal(eager_logits, decoder.logits) and torch.equal(eager_ids, decoder.ids)
        decoder.restore_capture_snapshot(before)
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original)
    candidate_graph = decoder.graph
    rows = []
    for step in range(settings['steps_per_phase']):
        before = decoder.capture_snapshot()
        reference_time = timed(reference_graph.replay, backend.device)
        expected_logits, expected_ids, expected_positions = decoder.logits.clone(), decoder.ids.clone(), decoder.positions.clone()
        after = decoder.capture_snapshot()
        after = (*after[:2], [(target, indices, target.gather(2, indices)) for target, indices, _ in before[2]])
        decoder.restore_capture_snapshot(before)
        candidate_time = timed(candidate_graph.replay, backend.device)
        assert torch.equal(expected_positions, decoder.positions)
        _, current = metadata(decoder)
        delta = (expected_logits.float() - decoder.logits.float()).abs()
        rows.append({'step': step, 'task_ids': identities, 'reference': reference_time, 'candidate': candidate_time,
            'reference_token_ids': expected_ids[:, 0].tolist(), 'candidate_token_ids': decoder.ids[:, 0].tolist(),
            'same_token': (expected_ids[:, 0] == decoder.ids[:, 0]).tolist(),
            'maximum_logit_difference': delta.amax(-1).tolist(), 'mean_logit_difference': delta.mean(-1).tolist(),
            'metadata': current})
        decoder.restore_capture_snapshot(after)
    return {'task_ids': identities, 'initial': initial, 'graph_eager_exact': True,
        'reference_median_ms': statistics.median(row['reference']['cuda_event_ms'] for row in rows),
        'candidate_median_ms': statistics.median(row['candidate']['cuda_event_ms'] for row in rows),
        'same_token_count': sum(sum(row['same_token']) for row in rows),
        'token_comparisons': len(rows) * len(identities), 'rows': rows}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['source_inputs']).read_text())
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'protocol.json').write_text(json.dumps(settings, indent=2) + '\n')
    backend = Backend(shared.backend())
    capacity = max(len(tokens) for tokens in source['input_ids']) + shared.generation.max_new_tokens
    pool, stream = torch.cuda.graph_pool_handle(), torch.cuda.Stream(device=backend.device)

    def prefill(indices):
        sequences = [source['input_ids'][index] for index in indices]
        ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, settings['padding_side'])
        decoder = CapturedDecode(backend, len(indices), capacity, graph_pool=pool, graph_stream=stream)
        decoder.prefill({'input_ids': ids, 'attention_mask': mask})
        return decoder

    decoder = prefill(settings['initial_rows'])
    identities = [source['fields'][index]['id'] for index in settings['initial_rows']]
    phases = [qualify(decoder, identities, backend, settings, shared)]
    (output / 'initial.json').write_text(json.dumps(phases[-1], indent=2) + '\n')
    incoming = prefill(settings['incoming_rows'])
    survivor_rows = torch.tensor(settings['survivor_rows'], device=backend.device)
    incoming_rows = torch.arange(len(settings['incoming_rows']), device=backend.device)
    merged = RollingDecode(backend, len(survivor_rows) + len(incoming_rows), capacity, incoming)
    merged.graph_pool, merged.graph_stream = pool, stream
    merged.load([(decoder, survivor_rows), (incoming, incoming_rows)])
    identities = [identities[index] for index in settings['survivor_rows']] + [
        source['fields'][index]['id'] for index in settings['incoming_rows']]
    decoder = merged
    del incoming, merged
    phases.append(qualify(decoder, identities, backend, settings, shared))
    result = {'settings': settings, 'phases': phases, 'production_promoted': False,
              'positions_masks_and_row_identities_preserved': True,
              'all_grouped_eager_graph_exact': all(phase['graph_eager_exact'] for phase in phases)}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'phases'}), flush=True)


if __name__ == '__main__':
    main()
