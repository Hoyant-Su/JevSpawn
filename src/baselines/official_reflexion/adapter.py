import importlib
import json
from pathlib import Path
import sys
from threading import local

from baselines.lats.adapter import ChatModel, SandboxExecutor, mbpp_task

TASK_CONTEXT = local()


def load_upstream(upstream):
    programming = Path(upstream).resolve() / 'programming_runs'
    sys.path.insert(0, str(programming))
    core = importlib.import_module('reflexion')
    result_type = importlib.import_module('executors.executor_types').ExecuteResult
    assert Path(core.__file__).resolve() == programming / 'reflexion.py'
    assert Path(sys.modules['generators'].__file__).resolve().parent == programming / 'generators'
    core.model_factory = lambda name: TASK_CONTEXT.model
    core.executor_factory = lambda language, is_leet: TASK_CONTEXT.executor
    return core, result_type


def run_task(core, row, model, executor, *, max_iters, pass_at_k, log_path, verbose):
    task = mbpp_task(row)
    TASK_CONTEXT.model = model
    TASK_CONTEXT.executor = executor
    core.run_reflexion(dataset=[task], model_name=model.name, language='py', max_iters=max_iters,
                       pass_at_k=pass_at_k, log_path=str(log_path), verbose=verbose, is_leetcode=False)
    record = json.loads(Path(log_path).read_text().splitlines()[-1])
    return {'task_id': row['task_id'], 'solution': record['solution'],
            'public_test_passed': record['is_solved'], 'reflections': record['reflections'],
            'implementations': record['implementations'], 'test_feedback': record['test_feedback'],
            'public_evaluations': executor.evaluations, 'tool_calls': executor.calls}
