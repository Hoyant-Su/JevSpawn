import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from jev_spawn.runtime.decoding import CapturedDecode


FIXTURE = json.loads(Path('tests/infra/grouped_decode/fixtures.json').read_text())


class GroupedDecodeMetadataTests(unittest.TestCase):
    def decoder(self, configuration):
        decoder = CapturedDecode.__new__(CapturedDecode)
        decoder.backend = SimpleNamespace(config=configuration)
        decoder.key_valid = torch.tensor(FIXTURE['masks'], dtype=torch.bool, device=FIXTURE['device'])
        decoder.ids = torch.empty_like(decoder.key_valid[:, :1], dtype=torch.long)
        decoder.key_positions = torch.arange(decoder.key_valid.shape[1], device=FIXTURE['device'])
        lengths = torch.tensor(FIXTURE['lengths'], device=FIXTURE['device'])
        decoder.cache = SimpleNamespace(get_seq_length=lambda: lengths, lengths=lengths)
        decoder.configure_attention()
        return decoder

    def test_interval_metadata_matches_each_real_row(self):
        decoder = self.decoder(FIXTURE['backend'])
        decoder.validate_attention()
        self.assertEqual(decoder.leftpad.tolist(), FIXTURE['leftpad'])
        callback = decoder.attention_arguments()['decode_attention']
        self.assertIs(callback.cache, decoder.cache)
        self.assertIs(callback.leftpad, decoder.leftpad)

    def test_row_replacement_updates_captured_metadata_storage(self):
        decoder = self.decoder(FIXTURE['backend'])
        decoder.validate_attention()
        callback = decoder.decode_kwargs['decode_attention']
        address = callback.leftpad.data_ptr()
        decoder.key_valid.copy_(torch.tensor(FIXTURE['replacement_masks']))
        decoder.cache.lengths.copy_(torch.tensor(FIXTURE['replacement_lengths']))
        decoder.attention_arguments()
        decoder.validate_attention()
        self.assertEqual(callback.leftpad.data_ptr(), address)
        self.assertEqual(callback.leftpad.tolist(), FIXTURE['replacement_leftpad'])

    def test_holes_and_empty_visible_intervals_are_rejected(self):
        decoder = self.decoder(FIXTURE['backend'])
        for masks in (FIXTURE['hole_masks'], FIXTURE['empty_masks']):
            decoder.key_valid.copy_(torch.tensor(masks))
            with self.assertRaisesRegex(AssertionError, 'contiguous valid interval'):
                decoder.validate_attention()

    def test_native_mode_preserves_masked_attention(self):
        decoder = self.decoder(FIXTURE['native_backend'])
        decoder.key_valid.copy_(torch.tensor(FIXTURE['hole_masks']))
        decoder.validate_attention()
        self.assertEqual(decoder.attention_arguments(), {})


if __name__ == '__main__':
    unittest.main()
