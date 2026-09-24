from functools import partial
from types import SimpleNamespace
import unittest

from baselines.common.config import SharedConfig
from baselines.common.parallel_refill_service import ParallelRefillService
from baselines.common.parallel_service import parallel_factory
from baselines.common.shared_refill_service import SharedRefillService


class ParallelProtocolTests(unittest.TestCase):
    def test_explicit_config_and_factory_select_true_refill(self):
        config = SharedConfig.load('configs/infra/shared_config_tp4_refill_qualification.yaml')
        self.assertEqual(config.runtime.world_size, 4)
        factory = parallel_factory(partial(SharedRefillService, settings={}, prompts={}))
        self.assertIs(factory.func, ParallelRefillService)

    def test_command_identity_sequence_and_stable_compaction(self):
        service = object.__new__(ParallelRefillService)
        service.command_sequence, service.row_ids = 0, [7, 8, 9]
        seen = []
        service.refill = SimpleNamespace(compact=lambda indices: seen.append(indices))
        service._parallel_compact({'sequence': 0, 'live_ids': [7, 8, 9], 'survivors': [0, 2]})
        self.assertEqual(service.row_ids, [7, 9])
        self.assertEqual(service.command_sequence, 1)
        self.assertEqual(seen, [[0, 2]])
        with self.assertRaises(AssertionError):
            service.accept({'sequence': 0, 'live_ids': [7, 9]})
        with self.assertRaises(AssertionError):
            service.accept({'sequence': 1, 'live_ids': [9, 7]})


if __name__ == '__main__':
    unittest.main()
