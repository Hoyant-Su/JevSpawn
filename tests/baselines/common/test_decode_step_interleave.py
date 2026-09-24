from concurrent.futures import Future
import json
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baselines.common.jevspawn_service import DecisionRequest, StructuredService
from baselines.common.service import BatchedRequest
from project_paths import ROOT


DATA = json.loads((ROOT / 'tests/fixtures/baselines/common/decode_step_interleave.json').read_text())


class InterleaveTests(unittest.TestCase):
    def test_finite_cohorts_preserve_generation_and_capacity(self):
        service = StructuredService.__new__(StructuredService)
        service.finite_scheduling = DATA['finite_scheduling']
        service.batch_size = DATA['batch_size']
        service.shared = SimpleNamespace(runtime=SimpleNamespace(branch_batch_size=DATA['batch_size']))
        service.backend = SimpleNamespace(device='cpu')
        service.requests, service.pending, service.records = Queue(), [], []
        service.interleaved_peak_allocated = service.interleaved_peak_reserved = 0
        service._validate_inputs = lambda requests: requests
        args = (DATA['messages'], DATA['max_tokens'], DATA['temperature'], (), DATA['task_id'], 0)
        generation = BatchedRequest(*args, Future())
        service.pending.append(generation)
        requests = [DecisionRequest(*args, Future(), field={'id': str(i)}) for i in range(DATA['decision_count'])]
        for request in requests:
            service.requests.put(request)

        def generate(batch):
            self.assertTrue(service.interleaving_finite)
            service.records.append({'peak_allocated_bytes': DATA['memory_bytes'],
                                    'peak_reserved_bytes': DATA['memory_bytes']})
            return [DATA['value'] for request in batch]

        service._generate = generate
        with patch('torch.cuda.max_memory_allocated', return_value=DATA['memory_bytes']), \
                patch('torch.cuda.max_memory_reserved', return_value=DATA['memory_bytes']):
            service._between_decode_steps()
            self.assertEqual(sum(request.future.done() for request in requests), DATA['batch_size'])
            self.assertIs(service.pending[0], generation)
            self.assertFalse(generation.future.done())
            service._between_decode_steps()
        self.assertEqual([request.future.result() for request in requests], [DATA['value']] * len(requests))
        self.assertEqual(service.pending, [generation])
        self.assertFalse(service.interleaving_finite)
        self.assertEqual(service.interleaved_peak_allocated, DATA['memory_bytes'])


if __name__ == '__main__':
    unittest.main()
