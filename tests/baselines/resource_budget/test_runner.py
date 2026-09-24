import json
from pathlib import Path
import time
import unittest
from uuid import uuid4

from baselines.resource_budget.runner import supervise
from baselines.resource_trials import deadline_quality


def run(config, output, connection):
    if config['scenario'] == 'startup_timeout':
        time.sleep(60)
        return
    connection.send({'event': 'ready', 'scope': 'CPU supervisor test, no model or benchmark execution.'})
    assert connection.recv() == {'event': 'start'}
    if config['scenario'] == 'deadline':
        time.sleep(60)
    elif config['scenario'] == 'out_of_memory':
        connection.send({'event': 'out_of_memory', 'error': 'Synthetic exception event for control testing.'})
    elif config['scenario'] == 'duplicate':
        event = {'event': 'delivery', 'results': [{'task_id': 'a', 'valid': True, 'answer': 'A'}]}
        connection.send(event)
        connection.send(event)
        time.sleep(60)
    else:
        assert config['scenario'] in {'completed', 'completed_hang'}
        connection.send({'event': 'delivery', 'results': [{'task_id': 'a', 'valid': True, 'answer': 'A'}]})
        connection.send({'event': 'completed'})
        if config['scenario'] == 'completed_hang':
            time.sleep(60)
    connection.close()


class SupervisorTest(unittest.TestCase):
    def trial(self, scenario):
        output = Path(__file__).resolve().parents[5] / 'runtime/research_v1/resource-trial-tests' / str(uuid4())
        config = {'scenario': scenario, 'startup_timeout_seconds': .5 if scenario == 'startup_timeout' else 10,
                  'deadline_seconds': .15, 'poll_seconds': .01, 'join_timeout_seconds': .2}
        result = supervise(config, output, ['a'], module=__name__)
        events = [json.loads(line) for line in (output / 'delivery.jsonl').read_text().splitlines()]
        return result, events

    def test_complete_delivery_uses_one_origin(self):
        result, events = self.trial('completed')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['worker_exitcode'], 0)
        self.assertEqual(deadline_quality(events, {'a': 'A'}, [.15])[0]['correct'], 1)

    def test_deadline_stops_the_owned_process(self):
        result, events = self.trial('deadline')
        self.assertEqual(result['status'], 'deadline')
        self.assertLess(result['service_seconds'], 5)
        self.assertGreaterEqual(result['stopping_overshoot_seconds'], 0)
        self.assertEqual(deadline_quality(events, {'a': 'A'}, [.15])[0]['correct'], 0)

    def test_startup_timeout_has_no_service_curve(self):
        result, events = self.trial('startup_timeout')
        self.assertEqual(result['status'], 'startup_timeout')
        self.assertIsNone(result['service_seconds'])
        self.assertEqual([row['event'] for row in events], ['assigned'])

    def test_oom_remains_infeasible(self):
        result, events = self.trial('out_of_memory')
        self.assertEqual(result['status'], 'out_of_memory')
        with self.assertRaises(AssertionError):
            deadline_quality(events, {'a': 'A'}, [.15])

    def test_invalid_delivery_terminates_instead_of_leaking_worker(self):
        result, events = self.trial('duplicate')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(events[-1]['event'], 'failed')
        self.assertEqual(sum(row['event'] == 'delivery' for row in events), 1)

    def test_completion_requires_the_worker_to_exit(self):
        result, events = self.trial('completed_hang')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(events[-1]['event'], 'failed')
