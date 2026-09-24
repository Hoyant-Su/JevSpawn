import argparse
import json
from pathlib import Path

import torch

from jev_spawn.runtime.batched_state_copy import copy_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    settings = json.loads(Path(config['settings']).read_text())
    assert settings['mode'] == 'triton_grouped'
    device = torch.device(config['device'])
    torch.cuda.set_device(device)
    torch.manual_seed(config['seed'])
    records, pairs, originals, destinations = [], [], [], []
    for repeat in range(config['repetitions']):
        for case in config['cases']:
            dtype = getattr(torch, case['dtype'])
            source = torch.randn(case['source_shape'], device=device, dtype=dtype)
            destination = torch.randn(case['destination_shape'], device=device, dtype=dtype)
            expected = destination.clone()
            source_view = source[tuple(slice(*item) for item in case['source_slices'])].permute(case['source_permutation'])
            destination_indices = tuple(slice(*item) for item in case['destination_slices'])
            destination_view = destination[destination_indices]
            assert source_view.shape == destination_view.shape
            expected[destination_indices].copy_(source_view)
            pairs.append((destination_view, source_view))
            originals.append((source, source.clone()))
            destinations.append((destination, expected))
            records.append({'repeat': repeat, 'case': case['name'], 'dtype': case['dtype'],
                            'shape': list(source_view.shape), 'source_stride': list(source_view.stride()),
                            'destination_stride': list(destination_view.stride())})
    copy_states(pairs, settings)
    torch.cuda.synchronize(device)
    for source, original in originals:
        assert torch.equal(source.view(torch.uint8), original.view(torch.uint8))
    for destination, expected in destinations:
        assert torch.equal(destination.view(torch.uint8), expected.view(torch.uint8))
    result = {'scope': 'GPU memory-copy parity on configured tensors; no model outputs or throughput claim.',
              'settings': settings, 'gpu': torch.cuda.get_device_name(device), 'copies': records,
              'all_destination_bytes_equal': True, 'all_source_bytes_unchanged': True}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'copies': len(records), 'all_destination_bytes_equal': True, 'all_source_bytes_unchanged': True}))


if __name__ == '__main__':
    main()
