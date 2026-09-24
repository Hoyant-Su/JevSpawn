import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen3_5Config
from transformers.cache_utils import StaticLayer

from jev_spawn.runtime.decoding import CapturedDecode
from jev_spawn.runtime.refill_cache_arena import RefillCacheArena
from jev_spawn.runtime.refill_state import RefillState
from jev_spawn.runtime.rolling_decode import RollingDecode


FIXTURE = json.loads(Path('tests/infra/refill_cache_arena/fixtures.json').read_text())
SEED = json.loads(Path('configs/reproducibility.json').read_text())['seed']


class RefillBindingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.arena = RefillCacheArena(Qwen3_5Config(text_config=FIXTURE['text_config']),
            FIXTURE['batch_size'], FIXTURE['capacity'], getattr(torch, FIXTURE['dtype']), FIXTURE['device'])
        self.state = RefillState(self.arena, FIXTURE['generation_capacity'], FIXTURE['pad_token_id'])
        self.requests = [SimpleNamespace(task_id=row['task_id'], max_tokens=row['max_tokens'],
                                        stop=tuple(row['stop'])) for row in FIXTURE['requests']]
        self.ends = [row['deadline'] for row in FIXTURE['requests']]
        self.backend = SimpleNamespace(config=FIXTURE['backend'],
            model=SimpleNamespace(model=SimpleNamespace(language_model=object())))

    def admit(self, indices):
        rows = self.state.reserve([self.requests[index] for index in indices],
                                  [self.ends[index] for index in indices])
        for layer in rows.cache.layers:
            if type(layer) is StaticLayer:
                shape = (len(indices), layer.num_heads, FIXTURE['prefill_width'], layer.k_head_dim)
                layer.update(torch.randn(shape, dtype=layer.dtype), torch.randn(shape, dtype=layer.dtype))
            else:
                for index, tensor in layer.conv_states.items():
                    layer.update_conv_state(torch.randn_like(tensor), state_idx=index)
                for index, tensor in layer.recurrent_states.items():
                    layer.update_recurrent_state(torch.randn_like(tensor), state_idx=index)
        rows.positions.fill_(FIXTURE['prefill_width'])
        rows.ids[:, 0].copy_(torch.tensor(FIXTURE['pending_token_ids'][:len(indices)]))
        rows.logits.normal_()
        self.state.commit_prefill(rows)
        return rows

    def test_binding_aliases_rows_and_uses_unchanged_native_computation(self):
        rows = self.admit(list(range(FIXTURE['first_admission'])))
        prefill = CapturedDecode.from_rows(self.backend, rows, self.state.key_positions, None, None)
        decode = self.state.decode(self.backend, None, None)
        self.assertIs(prefill.prefill.__func__, CapturedDecode.prefill)
        self.assertIs(decode.step.__func__, RollingDecode.step)
        self.assertIs(decode.capture.__func__, CapturedDecode.capture)
        for name in ('ids', 'positions', 'key_valid', 'logits'):
            self.assertEqual(getattr(prefill, name).data_ptr(), getattr(rows, name).data_ptr())
            self.assertEqual(getattr(decode, name).data_ptr(), getattr(self.arena, name).data_ptr())
        self.assertIs(decode.key_positions, self.state.key_positions)
        self.assertEqual(decode.capacity, FIXTURE['capacity'])

    def test_request_budgets_deadlines_stops_and_histories_follow_rows(self):
        first = list(range(FIXTURE['first_admission']))
        rows = self.admit(first)
        self.state.emit(rows.start, rows.stop)
        survivor = FIXTURE['survivors'][0]
        self.state.emit(survivor, survivor + 1)
        prior = [tensor.clone() for tensor in self.state.row_tensors()]
        self.state.compact(FIXTURE['survivors'])
        for actual, expected in zip(self.state.row_tensors(), prior, strict=True):
            self.assertTrue(torch.equal(actual[:self.arena.live_count], expected[FIXTURE['survivors']]))
        incoming = list(range(FIXTURE['first_admission'], len(self.requests)))
        rows = self.admit(incoming)
        self.state.emit(rows.start, rows.stop)
        expected_indices = [survivor, *incoming]
        self.assertTrue(all(actual is self.requests[index]
                            for actual, index in zip(self.state.requests, expected_indices, strict=True)))
        live = self.arena.live_count
        self.assertEqual(self.state.budgets[:live].tolist(), [self.requests[i].max_tokens for i in expected_indices])
        self.assertEqual(self.state.deadlines[:live].tolist(), [self.ends[i] for i in expected_indices])
        for pattern, mask in self.state.stop_groups():
            self.assertEqual(mask.tolist(), [self.requests[i].stop == pattern for i in expected_indices])
        self.assertEqual(self.state.counts[:live].tolist(), [2, 1, 1])
        self.state.compact(FIXTURE['survivors_after_refill'])
        self.assertTrue(all(actual is self.requests[index]
                            for actual, index in zip(self.state.requests, incoming, strict=True)))

    def test_native_capture_snapshot_restores_bound_hybrid_state(self):
        self.admit(list(range(FIXTURE['first_admission'])))
        decoder = self.state.decode(self.backend, None, None)
        original = [tensor[:self.arena.live_count].clone() for tensor in self.arena.row_tensors()]
        snapshot = decoder.capture_snapshot()
        for layer in decoder.cache.layers:
            if isinstance(layer, StaticLayer):
                shape = (self.arena.live_count, layer.num_heads, 1, layer.k_head_dim)
                layer.update(torch.randn(shape, dtype=layer.dtype), torch.randn(shape, dtype=layer.dtype))
            else:
                for tensor in [*layer.conv_states.values(), *layer.recurrent_states.values()]:
                    tensor.add_(1)
        decoder.ids.add_(1)
        decoder.positions.add_(1)
        decoder.logits.add_(1)
        decoder.restore_capture_snapshot(snapshot)
        for index, (actual, expected) in enumerate(zip(self.arena.row_tensors(), original, strict=True)):
            self.assertTrue(torch.equal(actual[:self.arena.live_count], expected), f'Cache tensor {index} changed.')

    def test_request_storage_accounting_and_full_departure(self):
        expected = self.arena.nbytes + sum(t.numel() * t.element_size()
            for t in [*self.state.row_tensors(), self.state.key_positions])
        self.assertEqual(self.state.nbytes, expected)
        rows = self.admit(list(range(FIXTURE['first_admission'])))
        self.state.emit(rows.start, rows.stop)
        self.state.compact([])
        self.assertEqual(self.state.requests, [])
        self.assertEqual(self.arena.live_count, 0)
        rows = self.admit(list(range(FIXTURE['second_admission'])))
        self.assertTrue(torch.equal(self.state.counts[:rows.stop], torch.zeros(rows.stop, dtype=torch.long)))
        self.assertTrue(bool((self.state.history[:rows.stop] == FIXTURE['pad_token_id']).all()))


if __name__ == '__main__':
    unittest.main()
