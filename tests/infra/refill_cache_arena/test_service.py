from dataclasses import replace
from functools import partial
from queue import Queue
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baselines.common.config import SharedConfig
from baselines.common.rolling_service import Entry, RollingService
from baselines.common.service import BatchService
from baselines.common.shared_refill_service import RefillWave, SharedRefillService, select_service
from tests.infra.refill_cache_arena.test_binding import FIXTURE, RefillBindingTests


class ServiceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = SharedConfig.load('configs/shared_config_arena_v2.yaml')
        self.continuous = replace(self.config, runtime=replace(self.config.runtime,
            generation_scheduling='continuous', cache_allocation='shared_refill_cache_arena_v1'))

    def test_partial_common_service_preserves_settings_and_selects_refill(self):
        settings, prompts = {}, {}
        selected = select_service(self.continuous, partial(BatchService, settings=settings, prompts=prompts))
        self.assertIs(selected.func, SharedRefillService)
        self.assertIs(selected.keywords['settings'], settings)
        self.assertIs(selected.keywords['prompts'], prompts)
        self.assertIs(select_service(self.continuous, None), SharedRefillService)

    def test_cohort_keeps_custom_service_and_continuous_rejects_it(self):
        class CustomService(BatchService):
            pass
        custom = partial(CustomService, settings={})
        self.assertIs(select_service(self.config, custom), custom)
        with self.assertRaisesRegex(AssertionError, 'common autoregressive'):
            select_service(self.continuous, custom)

    def test_existing_independent_continuous_mode_selects_rolling(self):
        independent = replace(self.continuous, runtime=replace(self.continuous.runtime,
            cache_allocation='independent_static_cache_v1'))
        self.assertIs(select_service(independent, partial(BatchService)).func, RollingService)


class ServiceStateTests(RefillBindingTests):
    def test_service_compaction_preserves_entry_and_request_identity(self):
        indices = list(range(FIXTURE['first_admission']))
        self.admit(indices)
        entries = [Entry(request, {}, index) for index, request in enumerate(self.state.requests)]
        service = SharedRefillService.__new__(SharedRefillService)
        service.refill = self.state
        survivors = [len(entries) - 1]
        result = service.compact(entries, survivors)
        self.assertIs(result[0], entries[survivors[0]])
        self.assertIs(result[0].request, self.state.requests[0])
        self.assertEqual(self.arena.live_count, len(survivors))

    def test_wave_uses_resident_histories_counts_and_request_limits(self):
        self.admit([0])
        request = self.state.requests[0]
        request.stop = ()
        service = SimpleNamespace(refill=self.state, backend=SimpleNamespace(device='cpu'),
            deadlines=SimpleNamespace(end=lambda task_id: self.ends[0]))
        wave = RefillWave(service, object(), [Entry(request, {}, 0)], 0, 1)
        for actual, expected in [(wave.history, self.state.history), (wave.counts, self.state.counts),
                                 (wave.budgets, self.state.budgets), (wave.deadlines, self.state.deadlines)]:
            self.assertEqual(actual.data_ptr(), expected.data_ptr())
        wave.counts.add_(1)
        self.assertEqual(self.state.counts[0].item(), 1)


class RefillSchedulingTests(unittest.TestCase):
    def test_short_request_refills_while_long_request_remains_live(self):
        events, completed = [], []
        service = SharedRefillService.__new__(SharedRefillService)
        service.backend = SimpleNamespace(device='cpu')
        service.batch_size, service.batch_wait_seconds = 2, 0.001
        service.requests = Queue()
        service.scheduling_records = []
        requests = [SimpleNamespace(task_id=name, max_tokens=count) for name, count in
                    [('short', 2), ('long', 5), ('followup', 2)]]
        for request in requests[:2]:
            service.requests.put(request)
        service._validate_inputs = lambda batch: batch
        service.compact = lambda entries, survivors: [entries[row] for row in survivors]

        class SimulatedWave:
            def __init__(self, entries):
                self.entries = entries
                self.decoder = SimpleNamespace(graph=SimpleNamespace(replay=lambda: None))

            def emit(self):
                survivors = []
                for row, entry in enumerate(self.entries):
                    entry.count += 1
                    events.append((entry.request.task_id, entry.count))
                    if entry.count < entry.request.max_tokens:
                        survivors.append(row)
                    else:
                        completed.append(entry.request.task_id)
                        if entry.request is requests[0]:
                            service.requests.put(requests[2])
                        if len(completed) == len(requests):
                            service.requests.put(None)
                return survivors

        def prefill(batch):
            return SimulatedWave([Entry(request, {}, row) for row, request in enumerate(batch)])

        def bind(entries):
            service.scheduling_records.append({'steps': 0})
            return SimulatedWave(entries)

        service.prefill, service.bind = prefill, bind
        with patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), patch('torch.cuda.Event'):
            service._serve()
        self.assertEqual(completed, ['short', 'followup', 'long'])
        self.assertLess(events.index(('followup', 1)), events.index(('long', 5)))
        for request in requests:
            counts = [count for task_id, count in events if task_id == request.task_id]
            self.assertEqual(counts, list(range(1, request.max_tokens + 1)))


if __name__ == '__main__':
    unittest.main()
