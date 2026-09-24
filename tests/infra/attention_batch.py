import argparse
import json
from pathlib import Path

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from baselines.common.config import SharedConfig
from jev_spawn.algo.structured import padded
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.cached_attention import grouped_cache_attention
from jev_spawn.runtime.decoding import CapturedDecode


@torch.inference_mode()
def run(settings):
    shared = SharedConfig.load(settings['shared_config'])
    source = json.loads(Path(settings['source_inputs']).read_text())
    sequences = [source['input_ids'][index] for index in settings['source_rows']]
    backend = Backend(shared.backend())
    ids, mask = padded(sequences, backend.tokenizer.pad_token_id, backend.device, settings['padding_side'])
    decoder = CapturedDecode(backend, len(sequences), ids.shape[1] + shared.generation.max_new_tokens)
    decoder.prefill({'input_ids': ids, 'attention_mask': mask})
    decoder.capture(shared.runtime.graph_warmup_steps)
    leftpad = (ids.shape[1] - mask.sum(-1)).to(torch.int32)
    selected = torch.tensor(settings['selected_rows'], device=backend.device)
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    rows = []

    def inspect_attention(module, query, key, value, attention_mask, **kwargs):
        output, weights = original(module, query, key, value, attention_mask, **kwargs)
        lengths = decoder.cache.layers[module.layer_idx].cumulative_length.expand(query.shape[0]).to(torch.int32)
        positions = torch.arange(key.shape[-2], device=key.device)[None, :]
        expected_mask = (positions >= leftpad[:, None]) & (positions < lengths[:, None])
        assert torch.equal(expected_mask, attention_mask[:, 0, 0])
        flash = grouped_cache_attention(query, key, value, lengths, leftpad, kwargs['scaling'], settings['num_splits'])
        compacted = grouped_cache_attention(
            query.index_select(0, selected), key.index_select(0, selected), value.index_select(0, selected),
            lengths.index_select(0, selected), leftpad.index_select(0, selected), kwargs['scaling'], settings['num_splits'])
        poisoned_key, poisoned_value = key.clone(), value.clone()
        poisoned_key.masked_fill_(~expected_mask[:, None, :, None], settings['invalid_key_value'])
        poisoned_value.masked_fill_(~expected_mask[:, None, :, None], settings['invalid_value_value'])
        poisoned = grouped_cache_attention(query, poisoned_key, poisoned_value, lengths, leftpad,
                                           kwargs['scaling'], settings['num_splits'])
        row = {'step': step, 'layer': module.layer_idx, 'valid_lengths': lengths.tolist(),
               'left_padding': leftpad.tolist(), 'row_selection_exact': torch.equal(compacted, flash.index_select(0, selected)),
               'invalid_cache_ignored_exactly': torch.equal(poisoned, flash),
               'sdpa_flash_max_difference': (output.float() - flash.float()).abs().max().item()}
        rows.append(row)
        assert row['row_selection_exact'] and row['invalid_cache_ignored_exactly'], row
        return output, weights

    ALL_ATTENTION_FUNCTIONS.register('sdpa', inspect_attention)
    for step in range(max(settings['inspect_steps']) + 1):
        if step in settings['inspect_steps']:
            decoder.step()
        else:
            decoder.graph.replay()
    result = {'settings': settings, 'input_lengths': list(map(len, sequences)), 'batch_size': len(sequences),
              'comparisons': len(rows), 'all_masks_equal': True,
              'row_selection_exact': all(row['row_selection_exact'] for row in rows),
              'invalid_cache_ignored_exactly': all(row['invalid_cache_ignored_exactly'] for row in rows), 'layers': rows}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'layers'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
