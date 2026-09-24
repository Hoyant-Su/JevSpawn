from copy import deepcopy
import json
from pathlib import Path
import unittest

import torch
import transformers
from transformers.cache_utils import DynamicCache, DynamicLayer, LinearAttentionLayer

from jev_spawn.runtime.native_cache_batch import pack_native_caches, split_native_cache


SETTINGS = json.loads(Path('tests/runtime/native_cache_batch/fixtures.json').read_text())


def make_cache(length, record_past):
    cache = DynamicCache(offloading=False)
    dimensions = SETTINGS
    options = {'device': dimensions['device'], 'dtype': getattr(torch, dimensions['dtype'])}
    cache.layers = []
    for kind in dimensions['layers']:
        if kind == 'full_attention':
            layer = DynamicLayer()
            shape = (dimensions['batch_size'], dimensions['heads'], length)
            layer.update(torch.randn(*shape, dimensions['key_dim'], **options),
                         torch.randn(*shape, dimensions['value_dim'], **options))
        else:
            layer = LinearAttentionLayer(number_of_states=dimensions['states_per_linear_layer'])
            for state_index in range(dimensions['states_per_linear_layer']):
                layer.update_conv_state(torch.randn(dimensions['batch_size'], dimensions['conv_channels'],
                                                    dimensions['conv_width'], **options), state_idx=state_index,
                                        conv_kernel_size=dimensions['conv_width'])
                layer.update_recurrent_state(torch.randn(dimensions['batch_size'], dimensions['heads'],
                                                         dimensions['key_dim'], dimensions['value_dim'], **options),
                                             state_idx=state_index)
            layer.record_past = record_past
        cache.layers.append(layer)
    return cache


def tensors(cache):
    for layer in cache.layers:
        if type(layer) is DynamicLayer:
            yield layer.keys
            yield layer.values
        else:
            yield from layer.conv_states.values()
            yield from layer.recurrent_states.values()


class NativeCacheBatchTest(unittest.TestCase):
    def assert_cache_equal(self, left, right):
        self.assertEqual(left.get_seq_length(), right.get_seq_length())
        for one, two in zip(left.layers, right.layers, strict=True):
            self.assertIs(type(one), type(two))
            omitted = (('keys', 'values') if type(one) is DynamicLayer else ('conv_states', 'recurrent_states'))
            self.assertEqual({key: value for key, value in vars(one).items() if key not in omitted},
                             {key: value for key, value in vars(two).items() if key not in omitted})
        for one, two in zip(tensors(left), tensors(right), strict=True):
            self.assertTrue(torch.equal(one, two))
            self.assertNotEqual(one.data_ptr(), two.data_ptr())

    def test_roundtrip_reordering_and_immutable_storage(self):
        torch.manual_seed(SETTINGS['seed'])
        for record_past in SETTINGS['record_past']:
            originals = [make_cache(length, record_past) for length in SETTINGS['lengths']]
            snapshots = deepcopy(originals)
            packed, mask = pack_native_caches(originals)
            self.assertEqual(mask.sum(dim=-1).tolist(), SETTINGS['lengths'])
            restored = split_native_cache(packed, SETTINGS['lengths'])
            for original, result in zip(originals, restored, strict=True):
                self.assert_cache_equal(original, result)
            order = SETTINGS['reorder']
            packed.reorder_cache(torch.tensor(order, device=SETTINGS['device']))
            lengths = [SETTINGS['lengths'][index] for index in order]
            reordered = split_native_cache(packed, lengths)
            for index, result in zip(order, reordered, strict=True):
                self.assert_cache_equal(originals[index], result)
            addresses = [tensor.untyped_storage().data_ptr()
                         for cache in [*originals, packed, *restored, *reordered] for tensor in tensors(cache)]
            self.assertEqual(len(addresses), len(set(addresses)))
            for cache in [packed, *restored, *reordered]:
                for tensor in tensors(cache):
                    tensor.add_(SETTINGS['mutation'])
            for original, snapshot in zip(originals, snapshots, strict=True):
                self.assert_cache_equal(original, snapshot)


if __name__ == '__main__':
    program = unittest.main(exit=False)
    result = {'transformers_version': transformers.__version__, 'tests_run': program.result.testsRun,
              'passed': program.result.wasSuccessful(), 'device': SETTINGS['device'],
              'scope': 'Real native cache classes and CPU tensor fixtures; no model computation or GPU qualification.',
              'lengths': SETTINGS['lengths'], 'reorder': SETTINGS['reorder'],
              'record_past': SETTINGS['record_past']}
    Path(SETTINGS['output']).write_text(json.dumps(result, indent=2) + '\n')
    assert program.result.wasSuccessful()
