import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import time

import torch

from baselines.common.config import SharedConfig
from jev_spawn.runtime.batched_state_copy import copy_states


def make_pairs(model, lengths, world_size, device, dtype):
    width, batch = max(lengths), len(lengths)
    pairs, storages = [], []
    for kind in model['layer_types']:
        if kind == 'full_attention':
            heads = model['num_key_value_heads'] // world_size
            for component in ('keys', 'values'):
                destination = torch.empty((batch, heads, width, model['head_dim']), device=device, dtype=dtype)
                storages.append(destination)
                for row, length in enumerate(lengths):
                    source = torch.randn((1, heads, length, model['head_dim']), device=device, dtype=dtype)
                    pairs.append((destination[row:row + 1, :, width - length:], source))
        else:
            assert kind == 'linear_attention'
            key_heads = model['linear_num_key_heads'] // world_size
            value_heads = model['linear_num_value_heads'] // world_size
            key_dim, value_dim = model['linear_key_head_dim'], model['linear_value_head_dim']
            dimensions = ((2 * key_heads * key_dim + value_heads * value_dim, model['linear_conv_kernel_dim']),
                          (value_heads, key_dim, value_dim))
            for shape, state_dtype in zip(dimensions, (dtype, torch.float32), strict=True):
                destination = torch.empty((batch, *shape), device=device, dtype=state_dtype)
                storages.append(destination)
                for row in range(batch):
                    source = torch.randn((1, *shape), device=device, dtype=state_dtype)
                    pairs.append((destination[row:row + 1], source))
    return pairs, storages


def main(config):
    shared = json.loads(Path(config['profile_result']).read_text())
    record, = [r for r in shared['records'] if r['phase'] == config['phase'] and r['variant'] == config['variant']]
    model = json.loads(Path(shared['startup'][0]['backend']['model_path'], config['model_config_file']).read_text())['text_config']
    runtime = SharedConfig.load(shared['config']['shared_config']).runtime
    torch.set_num_threads(runtime.cpu_threads)
    world_size = len(shared['startup'])
    assert world_size == runtime.world_size
    lengths = record['details'][0]['prefix_tokens']
    assert all(length == 1 for length in record['details'][0]['suffix_tokens'])
    device = torch.device(config['device'])
    torch.cuda.set_device(device)
    torch.manual_seed(config['seed'])
    dtype = getattr(torch, shared['startup'][0]['backend']['dtype'])
    results = []
    for rows in config['rows']:
        selected = [lengths[row] for row in rows]
        pairs, storages = make_pairs(model, selected, world_size, device, dtype)
        timings = defaultdict(list)
        modes = {name: json.loads(Path(path).read_text()) for name, path in config['settings'].items()}
        for settings in modes.values():
            for _ in range(config['warmup_steps']):
                copy_states(pairs, settings)
        torch.cuda.synchronize(device)
        for _ in range(config['repetitions']):
            for name, settings in modes.items():
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in config['event_pair'])
                started = time.perf_counter()
                begin.record()
                copy_states(pairs, settings)
                end.record()
                torch.cuda.synchronize(device)
                timings[name].append({'wall_seconds': time.perf_counter() - started,
                                      'cuda_seconds': begin.elapsed_time(end) / config['milliseconds_per_second']})
        assert all(torch.equal(destination.contiguous().view(torch.uint8), source.contiguous().view(torch.uint8))
                   for destination, source in pairs)
        results.append({'batch_size': len(rows), 'prefix_lengths': selected, 'tensor_copies': len(pairs),
                        'payload_bytes': sum(source.numel() * source.element_size() for _, source in pairs),
                        'all_bytes_equal': True, 'measurements': timings,
                        'median_wall_seconds': {name: median(r['wall_seconds'] for r in values)
                                                for name, values in timings.items()}})
    Path(config['result']).write_text(json.dumps({'config': config, 'results': results,
        'scope': config['scope']}, indent=2) + '\n')
    print(json.dumps([{'batch_size': r['batch_size'], 'median_wall_seconds': r['median_wall_seconds']}
                      for r in results]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    main(json.loads(args.config.read_text()))
