from concurrent.futures import Future
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock, patch

import torch

from baselines.common.rolling_service import RollingService
from baselines.common.service import BatchService, SampleStops


class CompletionFinishReasonTests(TestCase):
    def test_immediate_delivery_preserves_actual_row_causes(self):
        service = object.__new__(BatchService)
        service.backend = SimpleNamespace(device='cpu', tokenizer=Mock())
        service.backend.tokenizer.batch_decode.return_value = ['a', 'b', 'c']
        service.current_batch = [SimpleNamespace(stop=(), return_tokens=True, future=Future()) for _ in range(3)]
        service.stopping = SimpleNamespace(causes=torch.tensor([1, 2, 3]))
        service.delivered = {}
        service._deliver(torch.tensor([[4], [5], [6]]), [0, 1, 2])
        self.assertEqual([r.future.result()['finish_reason'] for r in service.current_batch],
                         ['stop', 'length', 'timeout'])

    def test_natural_stop_at_token_cap_is_not_length_exhaustion(self):
        backend = SimpleNamespace(device='cpu', eos_ids=[7])
        requests = [SimpleNamespace(max_tokens=1, task_id=str(i), stop=()) for i in range(2)]
        stop = SampleStops(backend, requests, 1, SimpleNamespace(end=lambda _: float('inf')), None)
        stop(torch.tensor([[0, 7], [0, 8]]), None)
        self.assertEqual(stop.causes.tolist(), [1, 2])

    def test_rolling_delivery_preserves_actual_cause(self):
        for cause, reason in [(1, 'stop'), (2, 'length'), (3, 'timeout')]:
            with self.subTest(reason=reason):
                service = object.__new__(RollingService)
                service.origin_host = 0.0
                service.origin = SimpleNamespace(elapsed_time=lambda _: 0.0)
                service.backend = SimpleNamespace(device='cpu')
                record = {key: [None] for key in ('output_tokens', 'output_token_ids', 'texts',
                    'truncated', 'expired', 'finish_reasons', 'row_tail_wait_seconds', 'response_seconds', 'decode')}
                record.update(active_token_slots=0, executed_token_slots=0, started_monotonic=0.0)
                request = SimpleNamespace(return_tokens=True, future=Future(), submitted=0.0)
                entry = SimpleNamespace(record=record, row=0, count=1, events=[object()], request=request)
                with patch('torch.cuda.max_memory_allocated', return_value=0), \
                     patch('torch.cuda.max_memory_reserved', return_value=0):
                    service.finish(entry, cause, [7], 'text')
                self.assertEqual(request.future.result()['finish_reason'], reason)
                self.assertEqual(record['finish_reasons'], [reason])


if __name__ == '__main__':
    main()
