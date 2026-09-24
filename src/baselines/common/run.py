import fcntl
from functools import partial
import importlib
import json
from pathlib import Path
import sys
import time

import jsonschema

from baselines.common.service_factory import service_factory
from baselines.common.runtime import InferenceRuntime
from baselines.common.transition_budget import TransitionBudget
from baselines.common.persistence import pending_tasks, save, validate_result
from data.task_context import rows
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.configuration import resolve_symbol
from project_paths import ROOT


def read(path):
    return json.loads(Path(path).read_text())


def finalize(tasks, output):
    results = [read(output / f'task-{index:05d}.json') for index in range(len(tasks))]
    for task, result in zip(tasks, results, strict=True):
        validate_result(task, result)
    save(output / 'completion.json', {'tasks': len(results),
         'completed': sum(result['status'] == 'completed' for result in results),
         'answers_present': sum(result['answer'] is not None for result in results),
         'task_ids': [task['task_id'] for task in tasks], 'labels_loaded': False})


def run(specification, output, backend=None):
    definition = specification['environment_execution']
    factory = partial(resolve_symbol(definition['class']), **definition['parameters'])
    return run_with_environment(specification, output, backend, environment_factory=factory,
                                environment_contract={'environment_execution': definition})


def run_with_environment(specification, output, backend, *, environment_factory, environment_contract):
    tasks = rows(specification['tasks'])
    assert len(tasks) == specification['task_count']
    assert len({task['task_id'] for task in tasks}) == len(tasks)
    assert specification['stage'] in {'qualification', 'formal'}
    method = read(specification['method'])
    prompts = load_prompt(method['prompts'])
    tools = read(specification['environment'])
    inference = read(ROOT / 'configs/inference/shared_service.json')
    session_contract = read(ROOT / 'configs/environments/session.json')
    sys.path[:0] = [str(Path(path).resolve()) for path in method['python_paths']]
    solve = importlib.import_module(method['module']).solve
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = {'specification': specification, 'method': method, 'prompts': prompts,
                    'tools': tools, 'tasks': tasks,
                    'shared_config_text': Path(specification['shared_config']).read_text(),
                    'inference': inference, 'session_contract': session_contract, **environment_contract}
        if (output / 'protocol.json').exists():
            assert read(output / 'protocol.json') == contract
        else:
            save(output / 'protocol.json', contract)
        pending = pending_tasks(tasks, output)
        if not pending:
            finalize(tasks, output)
            return
        runtime = InferenceRuntime(specification['shared_config'],
            service_factory(method, inference), backend=backend)
        settings = runtime.config.method_settings(method['settings'])
        runtime.service.configure_runtime_contract(settings)
        if 'trace' in settings:
            settings['trace'] = {**settings['trace'],
                                 'directory': str(output / settings['trace']['directory'])}
        runtime.service.agent_tokenizers = {task['task_id']: runtime.backend.tokenizer for task in pending}
        indexes = {task['task_id']: index for index, task in enumerate(tasks)}
        session = output / f'session-{len(list(output.glob("session-*"))):04d}'
        session.mkdir()
        save(session / 'runtime.json', runtime.metadata())

        def execute(task):
            environment = environment_factory(task, tools, session / 'tools' / str(indexes[task['task_id']]),
                deadline=partial(runtime.deadlines.remaining, task['task_id']), evidence=None)
            environment = TransitionBudget(environment, runtime.config.runtime.max_turns,
                session_contract['submission_tool'], session_contract['initial_transition_depth'], [])
            context = environment.reset()
            public_task = {'task_id': task['task_id'], 'context': context}
            save(session / f"context-{indexes[task['task_id']]:05d}.json", public_task)
            result = solve(public_task, environment, runtime.complete(task['task_id']), settings, prompts)
            result['tool_timings'] = environment.tool_timings
            result['state_transitions'] = environment.transitions
            if result['answer'] is not None:
                jsonschema.validate(result['answer'], task['answer_schema'])
            return result

        def commit(index, result):
            validate_result(pending[index], result)
            save(output / f"task-{indexes[result['task_id']]:05d}.json", result)
            print(json.dumps({'task_id': result['task_id'], 'status': result['status'],
                              'elapsed_seconds': result['elapsed_seconds']}), flush=True)

        started = time.perf_counter()
        try:
            results, elapsed = runtime.run(pending, execute, commit)
            save(session / 'completion.json', {'elapsed_seconds': elapsed, 'tasks': len(results),
                 'task_ids': [result['task_id'] for result in results],
                 'scope': 'Complete task lifecycles with tools, queue waits and graph captures. Model loading is separate.'})
        finally:
            runtime.close()
            save(session / 'batches.json', runtime.service.records)
            if runtime.config.runtime.generation_scheduling == 'continuous':
                save(session / 'scheduling.json', runtime.service.scheduling_records)
            save(session / 'input_failures.json', runtime.service.input_failures)
            save(session / 'elapsed.json', {'seconds': time.perf_counter() - started})
        finalize(tasks, output)
