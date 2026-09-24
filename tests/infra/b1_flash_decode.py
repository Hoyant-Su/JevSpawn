import argparse
import json
from pathlib import Path
import statistics

import torch
from flash_attn import flash_attn_with_kvcache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
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
    assert decoder.ids.shape[0] == 1 and bool(decoder.key_valid.all())
    decoder.capture(shared.runtime.graph_warmup_steps)
    reference_graph = decoder.graph

    def flash_forward(module, query, key, value, attention_mask, **kwargs):
        lengths = decoder.cache.layers[module.layer_idx].cumulative_length.expand(query.shape[0]).to(torch.int32)
        output = flash_attn_with_kvcache(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            cache_seqlens=lengths, softmax_scale=kwargs['scaling'],
            causal=settings['causal'], num_splits=settings['num_splits'],
        )
        return output, None

    ALL_ATTENTION_FUNCTIONS.register('sdpa', flash_forward)
    decoder.capture(shared.runtime.graph_warmup_steps)
    flash_graph = decoder.graph
    labels = torch.tensor(backend.answer_label_ids[:len(source['fields'][settings['source_row']]['options'])],
                          device=backend.device)
    rows = []
    for step in range(settings['steps']):
        before = decoder.capture_snapshot()
        reference_time = timed(reference_graph.replay, backend.device)
        expected_logits, expected_ids = decoder.logits.clone(), decoder.ids.clone()
        after_reference = decoder.capture_snapshot()
        # Restore the KV positions just written, rather than the next decode positions.
        after_reference = (*after_reference[:2], [(destination, indices, destination.gather(2, indices))
                                                  for destination, indices, _ in before[2]])
        decoder.restore_capture_snapshot(before)
        flash_time = timed(flash_graph.replay, backend.device)
        rows.append({'step': step, 'reference': reference_time, 'flash': flash_time,
                     'same_token': torch.equal(expected_ids, decoder.ids),
                     'same_finite_choice': torch.equal(expected_logits.index_select(-1, labels).argmax(-1),
                                                     decoder.logits.index_select(-1, labels).argmax(-1)),
                     'max_logit_difference': (expected_logits.float() - decoder.logits.float()).abs().max().item(),
                     'mean_logit_difference': (expected_logits.float() - decoder.logits.float()).abs().mean().item()})
        decoder.restore_capture_snapshot(after_reference)
    result = {'settings': settings, 'input_tokens': len(tokens), 'cache_capacity': decoder.capacity,
              'reference_median_ms': statistics.median(row['reference']['cuda_event_ms'] for row in rows),
              'flash_median_ms': statistics.median(row['flash']['cuda_event_ms'] for row in rows),
              'same_token_count': sum(row['same_token'] for row in rows),
              'same_finite_choice_count': sum(row['same_finite_choice'] for row in rows), 'rows': rows}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'rows'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
