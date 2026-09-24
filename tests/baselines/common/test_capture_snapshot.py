import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from transformers.cache_utils import StaticLayer

from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.rolling_decode import RowStaticLayer


class CaptureSnapshotTest(unittest.TestCase):
    def test_full_state_restoration_for_scalar_and_row_positions(self):
        config = json.loads(Path('configs/tests/baselines/common/capture_snapshot.json').read_text())
        torch.manual_seed(config['seed'])
        shape = (config['batch'], config['heads'], 1, config['head_dim'])
        for kind in [StaticLayer, RowStaticLayer]:
            layer = kind(max_cache_len=config['capacity'])
            keys, values = torch.randn(shape), torch.randn(shape)
            layer.lazy_initialization(keys, values)
            if kind is RowStaticLayer:
                layer.cumulative_length = torch.arange(config['batch'])
            else:
                layer.cumulative_length.fill_(config['position'])
            layer.keys.normal_()
            layer.values.normal_()
            decoder = CapturedDecode.__new__(CapturedDecode)
            decoder.ids = torch.ones(config['batch'], 1, dtype=torch.long)
            decoder.positions = decoder.ids.clone()
            decoder.logits = torch.randn(config['batch'], config['vocabulary'])
            decoder.capacity = config['capacity']
            decoder.cache = SimpleNamespace(layers=[layer])
            tensors = [decoder.ids, decoder.positions, decoder.logits,
                       layer.keys, layer.values, layer.cumulative_length]
            original = [t.clone() for t in tensors]
            snapshot = decoder.capture_snapshot()
            layer.update(torch.randn(shape), torch.randn(shape))
            decoder.ids.add_(1)
            decoder.positions.add_(1)
            decoder.logits.add_(1)
            decoder.restore_capture_snapshot(snapshot)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(tensors, original)))
            self.assertLess(decoder.capture_memory['saved_state_bytes'], decoder.capture_memory['full_state_bytes'])


if __name__ == '__main__':
    unittest.main()
