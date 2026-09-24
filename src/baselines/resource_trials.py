"""Record resident-service answer delivery without consulting task labels."""

import json
from pathlib import Path
from threading import Lock
import time


class DeliveryTrace:
    def __init__(self, path, task_ids, *, clock=time.perf_counter):
        assert task_ids and len(task_ids) == len(set(task_ids))
        self.task_ids = set(task_ids)
        self.delivered = set()
        self.clock = clock
        self.lock = Lock()
        self.stream = Path(path).open('x')
        self.origin = None
        self.terminal = False
        self.write({'event': 'assigned', 'task_ids': list(task_ids)})

    def write(self, event):
        self.stream.write(json.dumps(event) + '\n')
        self.stream.flush()

    def start(self):
        with self.lock:
            assert self.origin is None and not self.terminal
            self.origin = self.clock()
            self.write({'event': 'start', 'monotonic': self.origin})

    def deliver(self, results):
        """Invoke when parsed final answers become available to the caller."""
        with self.lock:
            assert self.origin is not None and not self.terminal
            ids = [result['task_id'] for result in results]
            assert ids and len(ids) == len(set(ids))
            assert set(ids) <= self.task_ids - self.delivered
            assert all(set(result) == {'task_id', 'valid', 'answer'} for result in results)
            assert all(type(result['valid']) is bool for result in results)
            now = self.clock()
            self.write({'event': 'delivery', 'monotonic': now,
                        'elapsed_seconds': now - self.origin, 'results': results})
            self.delivered.update(ids)

    def finish(self, status):
        with self.lock:
            assert self.origin is not None and not self.terminal
            assert status in {'completed', 'deadline', 'out_of_memory', 'interrupted', 'failed'}
            assert status != 'completed' or self.delivered == self.task_ids
            self.write({'event': status, 'monotonic': self.clock()})
            self.terminal = True
            self.stream.close()


def deadline_quality(events, labels, deadlines):
    """Score one uninterrupted terminal trial at its observed service cutoffs."""
    assert events[0]['event'] == 'assigned' and events[1]['event'] == 'start'
    assert events[-1]['event'] in {'completed', 'deadline'}
    task_ids = events[0]['task_ids']
    assert len(task_ids) == len(set(task_ids)) and set(labels) == set(task_ids)
    assert deadlines and all(t > 0 for t in deadlines)
    assert list(deadlines) == sorted(set(deadlines))
    origin = events[1]['monotonic']
    finished = events[-1]['monotonic'] - origin
    assert finished >= 0
    assert events[-1]['event'] == 'completed' or finished >= max(deadlines)
    deliveries = events[2:-1]
    seen = set()
    previous = 0
    for event in deliveries:
        assert event['event'] == 'delivery'
        elapsed = event['elapsed_seconds']
        assert elapsed == event['monotonic'] - origin
        assert previous <= elapsed <= finished
        previous = elapsed
        ids = [row['task_id'] for row in event['results']]
        assert ids and len(ids) == len(set(ids))
        assert set(ids) <= set(task_ids) - seen
        seen.update(ids)
    assert events[-1]['event'] != 'completed' or seen == set(task_ids)
    curves = []
    for deadline in deadlines:
        answers = [row for event in deliveries if event['elapsed_seconds'] <= deadline
                   for row in event['results']]
        correct = sum(row['valid'] and row['answer'] == labels[row['task_id']] for row in answers)
        valid = sum(row['valid'] for row in answers)
        curves.append({'deadline_seconds': deadline, 'assigned': len(task_ids),
                       'delivered': len(answers), 'valid': valid, 'correct': correct,
                       'accuracy': correct / len(task_ids), 'coverage': valid / len(task_ids)})
    return curves
