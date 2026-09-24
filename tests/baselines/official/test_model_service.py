from concurrent.futures import Future
from types import SimpleNamespace
import time
import unittest

from baselines.official.model_service import GenerationService, Request


class TokenizerFixture:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return [row[0]['content'] for row in messages]

    def __call__(self, texts, **kwargs):
        assert kwargs == dict(padding=False, truncation=False, add_special_tokens=False)
        return {'input_ids': [[ord(character) for character in text] for text in texts]}


class RecordingService(GenerationService):
    """CPU fixture for queue isolation, without model inference."""

    def _generate(self, batch):
        self.records.append({
            'task_ids': [request.task_id for request in batch],
            'batch_size': len(batch),
            'input_tokens': [len(request.messages[0]['content']) for request in batch],
        })
        return [request.task_id for request in batch]


def request(task_id, length):
    return Request([{'role': 'user', 'content': 'x' * length}],
                   128, 0.0, (), task_id, time.perf_counter(), Future())


class InputIsolationTest(unittest.TestCase):
    def setUp(self):
        self.service = RecordingService(
            SimpleNamespace(tokenizer=TokenizerFixture(), config={'max_input_tokens': 8192}),
            batch_size=2, batch_wait_seconds=0.05)
        self.addCleanup(self.service.close)

    def test_oversize_does_not_poison_compatible_row(self):
        for invalid_first in (False, True):
            valid, invalid = request('valid', 3692), request('oversize', 8278)
            for row in ([invalid, valid] if invalid_first else [valid, invalid]):
                self.service.requests.put(row)
            self.assertEqual(valid.future.result(timeout=5), 'valid')
            with self.assertRaisesRegex(ValueError, 'Input has 8278 tokens; limit is 8192'):
                invalid.future.result(timeout=5)
            record = self.service.records[-1]
            self.assertEqual(record['task_ids'], ['valid'])
            self.assertEqual(record['input_tokens'], [3692])
            self.assertEqual(record['batch_size'], 1)
            self.assertEqual(record['input_validation_batch_size'], 2)
            self.assertGreaterEqual(record['input_validation_seconds'], 0)
            self.assertEqual(self.service.input_failures[-1]['task_id'], 'oversize')
            self.assertEqual(self.service.input_failures[-1]['input_tokens'], 8278)

    def test_all_invalid_batch_then_valid_boundary_batch(self):
        for row in (request('oversize-a', 8193), request('oversize-b', 9000)):
            self.service.requests.put(row)
        for row in (request('at-limit', 8192), request('short', 3)):
            self.service.requests.put(row)
        self.assertEqual(row.future.result(timeout=5), 'short')
        self.assertEqual(len(self.service.records), 1)
        self.assertEqual(self.service.records[0]['task_ids'], ['at-limit', 'short'])
        self.assertEqual(self.service.records[0]['input_tokens'], [8192, 3])
        self.assertEqual(self.service.records[0]['batch_size'], 2)
        self.assertEqual(len(self.service.input_failures), 2)


if __name__ == '__main__':
    unittest.main()
