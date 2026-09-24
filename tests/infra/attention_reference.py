import argparse
import json
from pathlib import Path

import torch
from flash_attn import flash_attn_with_kvcache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from baselines.common.config import SharedConfig
from jev_spawn.infra.backend import Backend
from jev_spawn.runtime.decoding import CapturedDecode


def errors(actual, reference):
    difference = actual.to(reference.dtype) - reference
    return {'max_absolute': difference.abs().max().item(),
            'mean_absolute': difference.abs().mean().item(),
            'relative_l2': (difference.norm() / reference.norm()).item()}


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
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    dtype = getattr(torch, settings['reference_dtype'])
    rows = []

    def inspect_attention(module, query, key, value, attention_mask, **kwargs):
        output, weights = original(module, query, key, value, attention_mask, **kwargs)
        lengths = decoder.cache.layers[module.layer_idx].cumulative_length.expand(query.shape[0]).to(torch.int32)
        expected_mask = torch.arange(key.shape[-2], device=key.device)[None, :] < lengths[:, None]
        assert torch.equal(expected_mask, attention_mask[:, 0, 0])
        flash = flash_attn_with_kvcache(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            cache_seqlens=lengths, softmax_scale=kwargs['scaling'],
            causal=settings['causal'], num_splits=settings['num_splits'])
        batch, heads, width, dim = query.shape
        kv_heads = key.shape[1]
        grouped = query.to(dtype).reshape(batch, kv_heads, heads // kv_heads, width, dim)
        scores = torch.matmul(grouped, key.to(dtype)[:, :, None].transpose(-1, -2)) * kwargs['scaling']
        scores.masked_fill_(~attention_mask[:, None], float('-inf'))
        reference = torch.matmul(scores.softmax(-1), value.to(dtype)[:, :, None])
        reference = reference.reshape(batch, heads, width, dim).transpose(1, 2)
        rows.append({'step': step, 'layer': module.layer_idx, 'valid_tokens': lengths.tolist(),
                     'sdpa': errors(output, reference), 'flash': errors(flash, reference),
                     'rounded_reference': errors(reference.to(query.dtype), reference),
                     'flash_vs_sdpa': errors(flash, output.to(dtype))})
        return output, weights

    ALL_ATTENTION_FUNCTIONS.register('sdpa', inspect_attention)
    for step in range(max(settings['inspect_steps']) + 1):
        if step in settings['inspect_steps']:
            decoder.step()
        else:
            decoder.graph.replay()
    result = {'settings': settings, 'layers': rows,
              'all_valid_context_masks_equal': True,
              'flash_lower_relative_l2_count': sum(row['flash']['relative_l2'] < row['sdpa']['relative_l2'] for row in rows),
              'layer_comparisons': len(rows)}
    Path(settings['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'layers'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
