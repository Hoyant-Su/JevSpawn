from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch
from transformers import Qwen3_5TextConfig
from transformers.cache_utils import DynamicCache, DynamicLayer, StaticLayer

from jev_spawn.runtime.cache_arena import StaticCacheArena
from jev_spawn.runtime.finite_decoding import CapturedFinite


SETTINGS = json.loads(Path('tests/runtime/finite_decoding/fixtures.json').read_text())


def tensors(cache):
    for layer in cache.layers:
        if type(layer) in (DynamicLayer, StaticLayer):
            yield layer.keys
            yield layer.values
        else:
            yield from layer.conv_states.values()
            yield from layer.recurrent_states.values()


def native_state(config, length):
    cache = DynamicCache(config=config)
    options = {'device': SETTINGS['device'], 'dtype': getattr(torch, SETTINGS['dtype'])}
    for layer in cache.layers:
        if type(layer) is DynamicLayer:
            shape = (SETTINGS['singleton'], config.num_key_value_heads, length, config.head_dim)
            layer.update(torch.randn(shape, **options), torch.randn(shape, **options))
        else:
            channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim + config.linear_num_value_heads * config.linear_value_head_dim
            layer.update_conv_state(torch.randn(SETTINGS['singleton'], channels, config.linear_conv_kernel_dim, **options),
                                    conv_kernel_size=config.linear_conv_kernel_dim)
            layer.update_recurrent_state(torch.randn(SETTINGS['singleton'], config.linear_num_value_heads,
                config.linear_key_head_dim, config.linear_value_head_dim,
                device=SETTINGS['device'], dtype=getattr(torch, SETTINGS['recurrent_dtype'])))
    return cache


class FiniteLoaderTest(unittest.TestCase):
    def test_native_states_masks_positions_and_stable_separate_storage(self):
        torch.manual_seed(SETTINGS['seed'])
        config = Qwen3_5TextConfig(**SETTINGS['model'])
        originals = [native_state(config, length) for length in SETTINGS['lengths']]
        snapshots = deepcopy(originals)
        for order in SETTINGS['cases']:
            decoder = CapturedFinite.__new__(CapturedFinite)
            decoder.copy_settings = SETTINGS['copy_settings']
            decoder.capacity = SETTINGS['capacity']
            arena = StaticCacheArena(config, len(order), decoder.capacity,
                                     getattr(torch, SETTINGS['dtype']), SETTINGS['device'])
            decoder.cache = arena.bind(len(order), decoder.capacity)
            decoder.ids = torch.empty((len(order), SETTINGS['singleton']), dtype=torch.long)
            decoder.positions = torch.empty_like(decoder.ids)
            decoder.key_positions = torch.arange(decoder.capacity)
            decoder.key_valid = torch.empty((len(order), decoder.capacity), dtype=torch.bool)
            decoder.decode_kwargs = {}
            pointers = [tensor.data_ptr() for tensor in tensors(decoder.cache)]
            lengths = [SETTINGS['lengths'][index] for index in order]
            tokens = [SETTINGS['last_tokens'][index] for index in order]
            decoder.load([originals[index] for index in order], tokens)
            self.assertEqual(decoder.positions.flatten().tolist(), lengths)
            self.assertEqual(decoder.ids.flatten().tolist(), tokens)
            width = max(lengths)
            self.assertEqual(decoder.cache.get_seq_length().item(), width)
            self.assertEqual(decoder.key_valid[:, :width].sum(-1).tolist(), lengths)
            for row, source_index in enumerate(order):
                for source, target in zip(originals[source_index].layers, decoder.cache.layers, strict=True):
                    if type(target) is StaticLayer:
                        for name in ('keys', 'values'):
                            expected = getattr(source, name)
                            actual = getattr(target, name)[row:row + SETTINGS['singleton'], :, width - lengths[row]:width]
                            self.assertTrue(torch.equal(expected, actual))
                            self.assertNotEqual(expected.untyped_storage().data_ptr(), actual.untyped_storage().data_ptr())
                    else:
                        self.assertEqual(target.has_previous_state, source.has_previous_state)
                        for name in ('conv_states', 'recurrent_states'):
                            for index, expected in getattr(source, name).items():
                                actual = getattr(target, name)[index][row:row + SETTINGS['singleton']]
                                self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(pointers, [tensor.data_ptr() for tensor in tensors(decoder.cache)])
            for tensor in tensors(decoder.cache):
                tensor.add_(SETTINGS['mutation'])
        for original, snapshot in zip(originals, snapshots, strict=True):
            for one, two in zip(tensors(original), tensors(snapshot), strict=True):
                self.assertTrue(torch.equal(one, two))


if __name__ == '__main__':
    program = unittest.main(exit=False)
    result = {'tests_run': program.result.testsRun, 'passed': program.result.wasSuccessful(),
              'scope': 'CPU loader with real HF native/static cache classes and configured tensor fixtures. No model execution or CUDA graph qualification.',
              'fixtures': 'tests/runtime/finite_decoding/fixtures.json'}
    Path(SETTINGS['output']).write_text(json.dumps(result, indent=2) + '\n')
    assert program.result.wasSuccessful()
