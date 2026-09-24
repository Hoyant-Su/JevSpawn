from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicLayer

from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache_at
from jev_spawn.runtime.ragged_suffix import RaggedSuffix
from tests.runtime.finite_decoding.test_loader import native_state, tensors


SETTINGS = json.loads(Path('tests/runtime/ragged_suffix/fixtures.json').read_text())


class RaggedLayoutTest(unittest.TestCase):
    def test_pack_excludes_padding_and_restores_real_tokens(self):
        torch.manual_seed(SETTINGS['seed'])
        lengths = [length for length in SETTINGS['suffix_lengths'] if length]
        layout = RaggedSuffix(lengths, SETTINGS['device'])
        values = torch.randn(layout.batch_size, layout.width, SETTINGS['feature_dim'])
        packed = layout.pack(values)
        expected = torch.cat([row[:length] for row, length in zip(values, lengths, strict=True)])
        self.assertTrue(torch.equal(packed.squeeze(0), expected))
        recovered = layout.unpack(packed)
        for row, length in enumerate(lengths):
            self.assertTrue(torch.equal(recovered[row, :length], values[row, :length]))
            self.assertFalse(recovered[row, length:].count_nonzero())
        self.assertEqual(layout.cu_seqlens_cpu.diff().tolist(), lengths)

    def test_native_rows_remove_both_padding_sides_without_source_mutation(self):
        torch.manual_seed(SETTINGS['seed'])
        config = Qwen3_5TextConfig(**SETTINGS['model'])
        states = [native_state(config, length) for length in SETTINGS['lengths']]
        snapshots = deepcopy(states)
        cache, mask = pack_native_caches(states)
        width = max(SETTINGS['suffix_lengths'])
        tails = {}
        for index, layer in enumerate(cache.layers):
            if type(layer) is DynamicLayer:
                shape = (*layer.keys.shape[:-2], width, layer.keys.shape[-1])
                keys, values = torch.randn(shape), torch.randn(shape)
                tails[index] = (keys, values)
                layer.update(keys, values)
        lengths = [prefix + suffix for prefix, suffix in zip(SETTINGS['lengths'], SETTINGS['suffix_lengths'], strict=True)]
        stops = [mask.shape[1] + suffix for suffix in SETTINGS['suffix_lengths']]
        compact = split_native_cache_at(cache, lengths, stops)
        for row, (source, target, suffix) in enumerate(zip(states, compact, SETTINGS['suffix_lengths'], strict=True)):
            self.assertEqual(target.get_seq_length(), lengths[row])
            for index, (old, new) in enumerate(zip(source.layers, target.layers, strict=True)):
                if type(old) is DynamicLayer:
                    for name, extension in zip(('keys', 'values'), tails[index], strict=True):
                        expected = torch.cat([getattr(old, name), extension[row:row + 1, :, :suffix]], dim=-2)
                        self.assertTrue(torch.equal(getattr(new, name), expected))
                else:
                    for name in ('conv_states', 'recurrent_states'):
                        for key, value in getattr(old, name).items():
                            self.assertTrue(torch.equal(value, getattr(new, name)[key]))
        for source, snapshot in zip(states, snapshots, strict=True):
            for actual, expected in zip(tensors(source), tensors(snapshot), strict=True):
                self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    program = unittest.main(exit=False)
    result = {'tests_run': program.result.testsRun, 'passed': program.result.wasSuccessful(),
              'scope': 'CPU row layout and native cache compaction only; real-model ragged GDN qualification is separate.'}
    Path(SETTINGS['output']).write_text(json.dumps(result, indent=2) + '\n')
    assert program.result.wasSuccessful()
