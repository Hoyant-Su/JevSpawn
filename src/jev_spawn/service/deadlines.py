import math
import time


class SampleDeadlines:
    def __init__(self, seconds):
        self.seconds = seconds
        self.started = {}

    def start(self, task_id):
        assert task_id not in self.started
        self.started[task_id] = time.perf_counter()

    def end(self, task_id):
        return self.started[task_id] + (math.inf if self.seconds is None else self.seconds)

    def remaining(self, task_id):
        if self.seconds is None:
            return None
        remaining = self.end(task_id) - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError('Complete sample deadline exceeded: ' + task_id)
        return remaining

    def elapsed(self, task_id):
        return time.perf_counter() - self.started[task_id]
