import argparse
import json
from pathlib import Path
import statistics

import torch

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.fused_norm import install, restore
from jev_spawn.runtime.decoding import CapturedDecode
from tests.infra.b1_decode_readout import timed


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['source_inputs']).read_text())
    tokens = source['input_ids'][settings['source_row']]
    backend = Backend(shared.backend())
    ids = torch.tensor([tokens], device=backend.device)
    decoder = CapturedDecode(backend, settings['batch_size'], len(tokens) + shared.generation.max_new_tokens)
    decoder.prefill({'input_ids': ids, 'attention_mask': torch.ones_like(ids)})
    decoder.capture(shared.runtime.graph_warmup_steps)
    reference = decoder.graph
    originals = install(backend.model.model.language_model)
    decoder.capture(shared.runtime.graph_warmup_steps)
    candidate = decoder.graph
    labels = torch.tensor(backend.answer_label_ids[:len(source['fields'][settings['source_row']]['options'])],
                          device=backend.device)
    rows = []
    for step in range(settings['steps']):
        before = decoder.capture_snapshot()
        reference_time = timed(reference.replay, backend.device)
        expected_logits, expected_ids = decoder.logits.clone(), decoder.ids.clone()
        after = decoder.capture_snapshot()
        after = (*after[:2], [(destination, indices, destination.gather(2, indices))
                              for destination, indices, _ in before[2]])
        decoder.restore_capture_snapshot(before)
        candidate_time = timed(candidate.replay, backend.device)
        rows.append({'step': step, 'reference': reference_time, 'candidate': candidate_time,
                     'same_token': torch.equal(expected_ids, decoder.ids),
                     'same_finite_choice': torch.equal(expected_logits.index_select(-1, labels).argmax(-1),
                                                      decoder.logits.index_select(-1, labels).argmax(-1)),
                     'maximum_logit_difference': (expected_logits.float() - decoder.logits.float()).abs().max().item()})
        decoder.restore_capture_snapshot(after)
    restore(originals)
    result = {'settings': settings, 'input_tokens': len(tokens), 'cache_capacity': decoder.capacity,
              'fused_modules': [name for name, _, _ in originals],
              'reference_median_ms': statistics.median(row['reference']['cuda_event_ms'] for row in rows),
              'candidate_median_ms': statistics.median(row['candidate']['cuda_event_ms'] for row in rows),
              'same_token_count': sum(row['same_token'] for row in rows),
              'same_finite_choice_count': sum(row['same_finite_choice'] for row in rows), 'rows': rows}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('rows', 'fused_modules')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
