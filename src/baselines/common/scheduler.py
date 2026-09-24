from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time

from baselines.common.errors import FlowInputError, InputLimitError, InvalidOutputError, TaskLimitError


def run_tasks(tasks, solve, capacity, deadlines, commit):
    """Admit a new root whenever a root finishes, without a block barrier."""
    iterator = iter(enumerate(tasks))
    results = [None] * len(tasks)
    started = time.perf_counter()

    def execute(index, task):
        identity = task['task_id']
        deadlines.start(identity)
        try:
            result = solve(task)
            deadlines.remaining(identity)
            assert result['task_id'] == identity
            result['status'] = 'completed'
        except TimeoutError as error:
            result = {'task_id': identity, 'status': 'timeout', 'answer': None, 'error': str(error),
                      'trace': getattr(error, 'trace', None)}
        except (InvalidOutputError, FlowInputError) as error:
            result = {'task_id': identity, 'status': 'invalid_output', 'answer': None, 'error': str(error),
                      'trace': getattr(error, 'trace', None)}
        except (InputLimitError, TaskLimitError) as error:
            result = {'task_id': identity, 'status': 'limit_exceeded', 'answer': None, 'error': str(error),
                      'trace': getattr(error, 'trace', None)}
        result['admitted_monotonic'] = deadlines.started[identity]
        result['finished_monotonic'] = time.perf_counter()
        result['elapsed_seconds'] = deadlines.elapsed(identity)
        commit(index, result)
        return index, result

    with ThreadPoolExecutor(max_workers=capacity) as pool:
        pending = set()
        for _ in range(min(capacity, len(tasks))):
            index, task = next(iterator)
            pending.add(pool.submit(execute, index, task))
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed = [future.result() for future in ready]
            for index, result in completed:
                results[index] = result
                item = next(iterator, None)
                if item is not None:
                    pending.add(pool.submit(execute, *item))
    return results, time.perf_counter() - started
