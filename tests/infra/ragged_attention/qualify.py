import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from baselines.common.config import SharedConfig
from jev_spawn.infra.cached_attention import RaggedCacheAttention
from jev_spawn.runtime.ragged_suffix import RaggedSuffix


@torch.inference_mode()
def main(settings):
    shared = SharedConfig.load(settings['shared_config'])
    model = json.loads((Path(shared.model.path) / 'config.json').read_text())['text_config']
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.manual_seed(settings['seed'])
    device = torch.device('cuda', rank)
    dtype = getattr(torch, settings['dtype'])
    heads = model['num_attention_heads'] // shared.runtime.world_size
    kv_heads = model['num_key_value_heads'] // shared.runtime.world_size
    dim = model['head_dim']
    records = []
    for case in settings['cases']:
        prefix, lengths = case['prefix_lengths'], case['suffix_lengths']
        batch, width, base = len(lengths), max(lengths), max(prefix)
        suffix = RaggedSuffix(lengths, device)
        attention = RaggedCacheAttention(suffix, prefix)
        query = torch.randn(batch, heads, width, dim, device=device, dtype=dtype)
        key = torch.randn(batch, kv_heads, base + width, dim, device=device, dtype=dtype)
        value = torch.randn_like(key)
        positions = torch.arange(base + width, device=device)
        query_positions = torch.arange(width, device=device)
        leftpad = base - torch.tensor(prefix, device=device)
        valid = ((positions[None] >= leftpad[:, None]) &
                 (positions[None] < base + torch.tensor(lengths, device=device)[:, None]))
        mask = valid[:, None, None] & (positions[None] <= base + query_positions[:, None])[None, None]
        scaling = dim ** -0.5

        def reference():
            return F.scaled_dot_product_attention(
                query, key.repeat_interleave(heads // kv_heads, dim=1),
                value.repeat_interleave(heads // kv_heads, dim=1), attn_mask=mask,
                scale=scaling).transpose(1, 2)

        def packed():
            return attention(None, query, key, value, None, scaling=scaling)[0]

        result = {}
        outputs = {}
        for name, operation in [('sdpa', reference), ('varlen', packed)]:
            for _ in range(settings['warmup']):
                operation()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            start = time.perf_counter()
            for _ in range(settings['repeats']):
                outputs[name] = operation()
            torch.cuda.synchronize(device)
            result[name] = {
                'seconds_per_call': (time.perf_counter() - start) / settings['repeats'],
                'extra_peak_bytes': torch.cuda.max_memory_allocated(device) - baseline,
            }
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                operation()
            result[name]['attention_operators'] = [event.key for event in trace.key_averages()
                                                   if 'attention' in event.key.lower()]
        expected = suffix.pack(outputs['sdpa'])
        actual = suffix.pack(outputs['varlen'])
        torch.testing.assert_close(actual, expected, rtol=settings['rtol'], atol=settings['atol'])
        records.append({'case': case, 'batch_size': batch, 'heads_per_rank': heads,
                        'kv_heads_per_rank': kv_heads, 'head_dim': dim,
                        'max_absolute_error': (actual - expected).abs().max().item(), **result})
        output = Path(settings['output'])
        output.mkdir(parents=True, exist_ok=True)
        (output / f'rank-{rank}.json').write_text(json.dumps({
            'scope': 'Exact attention operator qualification; random tensors are test inputs, not task results.',
            'records': records}, indent=2) + '\n')
        print(json.dumps({'rank': rank, 'record': records[-1]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(json.loads(parser.parse_args().config.read_text()))
