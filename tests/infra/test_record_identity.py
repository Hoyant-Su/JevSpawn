from concurrent.futures import Future
from dataclasses import fields
from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from baselines.common.jevspawn_service import StructuredService
from baselines.common.service import BatchedRequest, BatchService
from baselines.official.model_service import GenerationService


class RecordIdentityTests(TestCase):
    def test_immediate_delivery_and_singleton_append_keep_generation_identity(self):
        service = object.__new__(StructuredService)
        model = Mock()
        service.backend = SimpleNamespace(model=model)
        service.shared = SimpleNamespace(runtime=SimpleNamespace(decode_engine='eager', row_delivery='immediate'))
        service.deadlines = SimpleNamespace(end=lambda task_id: float('inf'), remaining=lambda task_id: float('inf'))
        service.records, service.graph_stats, service.timeout_deliveries = [], {}, {}
        service.batch_size, service.interleaving_finite = 2, False
        service.stopping = SimpleNamespace(causes=torch.tensor([1, 1]), checked_monotonic=[])
        batch = [BatchedRequest([{'role': 'user', 'content': task}], 8, 0.0, (), task,
                                0.0, Future(), return_tokens=True) for task in ('left', 'right')]
        texts, token_ids = ['left output', 'right output'], [[11], [22]]
        appended = Event()

        def caller():
            self.assertEqual(batch[0].future.result(), {'text': texts[0], 'token_ids': token_ids[0],
                                                       'finish_reason': 'stop'})
            service.enqueue_decisions([{'id': 'singleton', 'options': [{'id': 'only'}]}],
                                      task_id='left', request_type=None)
            appended.set()

        def numerical_generation(instance, requests):
            model.register_forward_pre_hook.call_args.args[0](model, (), {
                'input_ids': torch.tensor([[1], [2]]), 'attention_mask': torch.tensor([[1], [1]])})
            record = {'task_ids': [r.task_id for r in requests], 'batch_size': 2,
                      'texts': texts, 'output_token_ids': token_ids, 'output_tokens': [1, 1],
                      'decode': [{}, {}], 'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
            instance._record_batch(requests, record)
            for request, text, ids in zip(requests, texts, token_ids):
                request.future.set_result({'text': text, 'token_ids': ids, 'finish_reason': 'stop'})
            self.assertTrue(appended.wait(timeout=5))
            return texts

        thread = Thread(target=caller)
        thread.start()
        with patch.object(GenerationService, '_generate', numerical_generation):
            outputs = BatchService._generate(service, batch)
        thread.join()
        generated, direct = service.records
        self.assertIs(batch[0].record, generated)
        self.assertNotIn('record', {field.name for field in fields(batch[0])})
        self.assertIs(batch[1].record, generated)
        self.assertEqual(generated['output_texts'], texts)
        self.assertEqual(generated['messages'], [request.messages for request in batch])
        self.assertEqual(direct['operation'], 'finite_direct')
        self.assertNotIn('output_texts', direct)
        self.assertNotIn('messages', direct)
        self.assertEqual(outputs, [request.future.result() for request in batch])
        self.assertEqual([row['token_ids'] for row in outputs], token_ids)


if __name__ == '__main__':
    main()
