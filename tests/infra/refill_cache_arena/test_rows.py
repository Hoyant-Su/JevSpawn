import json
from pathlib import Path
import unittest

import torch
from transformers import Qwen3_5Config
from transformers.cache_utils import LinearAttentionLayer, StaticLayer

from jev_spawn.runtime.refill_cache_arena import RefillCacheArena
from jev_spawn.runtime.rolling_decode import RowStaticLayer


FIXTURE = json.loads(Path('tests/infra/refill_cache_arena/fixtures.json').read_text())
SEED = json.loads(Path('configs/reproducibility.json').read_text())['seed']


class RefillRowsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.arena = RefillCacheArena(Qwen3_5Config(text_config=FIXTURE['text_config']),
            FIXTURE['batch_size'], FIXTURE['capacity'], getattr(torch, FIXTURE['dtype']), FIXTURE['device'])

    def prefill(self, view):
        count, width = view.stop - view.start, FIXTURE['prefill_width']
        for layer in view.cache.layers:
            if type(layer) is StaticLayer:
                shape = (count, layer.num_heads, width, layer.k_head_dim)
                layer.update(torch.randn(shape, dtype=layer.dtype), torch.randn(shape, dtype=layer.dtype))
            else:
                self.assertIs(type(layer), LinearAttentionLayer)
                for index, value in layer.conv_states.items():
                    layer.update_conv_state(torch.randn_like(value), state_idx=index)
                for index, value in layer.recurrent_states.items():
                    layer.update_recurrent_state(torch.randn_like(value), state_idx=index)
        view.ids.copy_(torch.randint(FIXTURE['text_config']['vocab_size'], view.ids.shape))
        view.positions.copy_(torch.arange(count)[:, None] + width)
        view.key_valid.copy_(torch.randint(2, view.key_valid.shape, dtype=torch.bool))
        view.logits.copy_(torch.randn_like(view.logits))
        self.arena.commit_prefill(view)

    def test_compaction_preserves_all_hybrid_states_and_metadata(self):
        self.prefill(self.arena.reserve(FIXTURE['first_admission']))
        originals = [tensor.clone() for tensor in self.arena.row_tensors()]
        addresses = [tensor.data_ptr() for tensor in self.arena.row_tensors()]
        survivors = FIXTURE['survivors']
        self.arena.compact(survivors)
        for tensor, expected in zip(self.arena.row_tensors(), originals, strict=True):
            self.assertTrue(torch.equal(tensor[:len(survivors)], expected[survivors]))
        self.assertEqual(addresses, [tensor.data_ptr() for tensor in self.arena.row_tensors()])
        self.assertEqual(self.arena.compaction_scratch_bytes, 0)

    def test_newcomer_reset_and_native_updates_cannot_touch_survivors(self):
        self.prefill(self.arena.reserve(FIXTURE['first_admission']))
        self.arena.compact(FIXTURE['survivors'])
        count = self.arena.live_count
        prior = [tensor[:count].clone() for tensor in self.arena.row_tensors()]
        active = self.arena.decode_view()
        incoming = self.arena.reserve(FIXTURE['second_admission'])
        for active_layer, fresh_layer in zip(active.cache.layers, incoming.cache.layers, strict=True):
            if isinstance(active_layer, StaticLayer):
                self.assertEqual(active_layer.keys.untyped_storage().data_ptr(),
                                 fresh_layer.keys.untyped_storage().data_ptr())
                self.assertLessEqual(active_layer.keys.storage_offset() + active_layer.keys.numel(),
                                     fresh_layer.keys.storage_offset())
                self.assertNotEqual(active_layer.cumulative_length.data_ptr(), fresh_layer.cumulative_length.data_ptr())
            else:
                self.assertTrue(all(active_layer.has_previous_state.values()))
                self.assertFalse(any(fresh_layer.has_previous_state.values()))
        incoming.cache.reset()
        self.prefill(incoming)
        for tensor, expected in zip(self.arena.row_tensors(), prior, strict=True):
            self.assertTrue(torch.equal(tensor[:count], expected))
        self.assertTrue(all(all(layer.has_previous_state.values()) for layer in active.cache.layers
                            if isinstance(layer, LinearAttentionLayer)))

    def test_decode_positions_are_per_row_and_second_compaction_is_exact(self):
        self.prefill(self.arena.reserve(FIXTURE['first_admission']))
        self.arena.compact(FIXTURE['survivors'])
        self.prefill(self.arena.reserve(FIXTURE['second_admission']))
        view = self.arena.decode_view()
        for layer in view.cache.layers:
            if type(layer) is RowStaticLayer:
                layer.cumulative_length.copy_(torch.arange(self.arena.live_count) + FIXTURE['prefill_width'])
                positions = layer.cumulative_length.clone()
                shape = (self.arena.live_count, layer.num_heads, 1, layer.k_head_dim)
                keys, values = torch.randn(shape, dtype=layer.dtype), torch.randn(shape, dtype=layer.dtype)
                layer.update(keys, values)
                for row, position in enumerate(positions):
                    self.assertTrue(torch.equal(layer.keys[row, :, position], keys[row, :, 0]))
                self.assertTrue(torch.equal(layer.cumulative_length, positions + 1))
        prior = [tensor.clone() for tensor in self.arena.row_tensors()]
        survivors = FIXTURE['survivors_after_refill']
        self.arena.compact(survivors)
        for tensor, expected in zip(self.arena.row_tensors(), prior, strict=True):
            self.assertTrue(torch.equal(tensor[:len(survivors)], expected[survivors]))

    def test_memory_is_one_backing_allocation_plus_declared_metadata(self):
        tensors = [*self.arena.row_tensors(), *(layer['length'] for layer in self.arena.storage if 'length' in layer)]
        storage = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors}
        self.assertEqual(self.arena.nbytes, sum(storage.values()))
        view = self.arena.reserve(FIXTURE['first_admission'])
        scalars = [layer.cumulative_length for layer in view.cache.layers if type(layer) is StaticLayer]
        self.assertEqual(view.scalar_metadata_bytes, sum(t.numel() * t.element_size() for t in scalars))
        self.prefill(view)
        self.assertEqual(self.arena.decode_view().scalar_metadata_bytes, 0)

    def test_unsupported_reordering_and_overlapping_reservations_fail(self):
        incoming = self.arena.reserve(FIXTURE['first_admission'])
        with self.assertRaises(AssertionError):
            self.arena.reserve(FIXTURE['second_admission'])
        self.prefill(incoming)
        with self.assertRaises(AssertionError):
            self.arena.compact(list(reversed(range(self.arena.live_count))))
        with self.assertRaises(AssertionError):
            self.arena.reserve(FIXTURE['batch_size'])


if __name__ == '__main__':
    unittest.main()
