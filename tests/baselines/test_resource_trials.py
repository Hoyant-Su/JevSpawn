import json
from pathlib import Path
import unittest
from uuid import uuid4

from baselines.resource_trials import DeliveryTrace, deadline_quality


class ResourceTrialTest(unittest.TestCase):
    def setUp(self):
        self.events = [
            {'event': 'assigned', 'task_ids': ['a', 'b', 'c', 'd']},
            {'event': 'start', 'monotonic': 100.},
            {'event': 'delivery', 'monotonic': 101., 'elapsed_seconds': 1.,
             'results': [{'task_id': 'a', 'answer': 'A', 'valid': True},
                         {'task_id': 'b', 'answer': 'B', 'valid': False}]},
            {'event': 'delivery', 'monotonic': 103., 'elapsed_seconds': 3.,
             'results': [{'task_id': 'c', 'answer': 'A', 'valid': True}]},
            {'event': 'deadline', 'monotonic': 104.02},
        ]
        self.labels = {'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D'}

    def test_exact_cutoffs_keep_unfinished_and_invalid_tasks(self):
        curves = deadline_quality(self.events, self.labels, [.999, 1., 2., 4.])
        self.assertEqual([r['delivered'] for r in curves], [0, 2, 2, 3])
        self.assertEqual([r['correct'] for r in curves], [0, 1, 1, 1])
        self.assertEqual([r['assigned'] for r in curves], [4, 4, 4, 4])
        self.assertEqual(curves[-1]['accuracy'], .25)
        self.assertEqual(curves[-1]['coverage'], .5)

    def test_interruption_and_oom_cannot_become_quality_curves(self):
        for terminal in ['interrupted', 'out_of_memory', 'failed']:
            with self.subTest(terminal=terminal):
                self.events[-1]['event'] = terminal
                with self.assertRaises(AssertionError):
                    deadline_quality(self.events, self.labels, [4.])

    def test_cannot_extrapolate_an_early_stopped_trial(self):
        with self.assertRaises(AssertionError):
            deadline_quality(self.events, self.labels, [5.])

    def test_missing_answers_prevent_claiming_full_completion(self):
        self.events[-1]['event'] = 'completed'
        with self.assertRaises(AssertionError):
            deadline_quality(self.events, self.labels, [4.])

    def test_duplicate_delivery_is_not_extra_coverage(self):
        self.events[3]['results'][0]['task_id'] = 'a'
        with self.assertRaises(AssertionError):
            deadline_quality(self.events, self.labels, [4.])

    def test_delivery_trace_preserves_shared_origin_and_whole_batches(self):
        output = Path(__file__).resolve().parents[4] / 'runtime/research_v1/resource-trial-tests'
        output.mkdir(parents=True, exist_ok=True)
        path = output / (str(uuid4()) + '.jsonl')
        ticks = iter([200., 201., 203.])
        trace = DeliveryTrace(path, ['a', 'b'], clock=lambda: next(ticks))
        trace.start()
        trace.deliver([{'task_id': 'a', 'valid': True, 'answer': 'A'},
                       {'task_id': 'b', 'valid': True, 'answer': 'B'}])
        trace.finish('completed')
        events = [json.loads(line) for line in path.read_text().splitlines()]
        curves = deadline_quality(events, {'a': 'A', 'b': 'B'}, [.5, 1., 4.])
        self.assertEqual([row['correct'] for row in curves], [0, 2, 2])


if __name__ == '__main__':
    unittest.main()
