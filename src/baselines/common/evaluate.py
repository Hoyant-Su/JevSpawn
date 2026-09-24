import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib
from itertools import repeat
import json
from multiprocessing import get_context
from numbers import Real
from pathlib import Path
import statistics
import time

import jsonschema
import yaml

from baselines.common.persistence import save
from project_paths import ROOT


def score_task(task, path, factory, parameters, tools, directory):
    score = {'task_id': task['task_id'],
             'artifact': str(path), 'status': 'missing', 'answer': None,
             'valid_answer': False, 'correct': False, 'score': 0.0,
             'elapsed_seconds': None, 'evaluation_seconds': None,
             'tool_calls': None, 'tool_seconds': None}
    if not path.exists():
        return score
    result = json.loads(path.read_text())
    assert result['task_id'] == task['task_id']
    score.update(status=result['status'], answer=result['answer'],
                 elapsed_seconds=result['elapsed_seconds'])
    if 'tool_timings' in result:
        timings = result['tool_timings']
        score.update(tool_calls=len(timings), tool_seconds=sum(
            item['finished_monotonic'] - item['started_monotonic'] for item in timings))
    for name in ('error', 'termination', 'pending_branches', 'selected_terminal'):
        if name in result:
            score[name] = result[name]
    if result['status'] != 'completed' or result['answer'] is None:
        return score
    errors = list(jsonschema.Draft202012Validator(task['answer_schema']).iter_errors(result['answer']))
    if errors:
        score['validation_errors'] = [error.message for error in errors]
        return score
    environment = factory(task, tools, directory, deadline=lambda: None, **parameters)
    started = time.perf_counter()
    try:
        outcome = environment.evaluate(result['answer'])
    finally:
        close = getattr(environment, 'close', None)
        if close is not None:
            close()
    assert isinstance(outcome, (bool, Real)), 'Native evaluator must return a boolean or numeric score.'
    score.update(valid_answer=True, correct=outcome if isinstance(outcome, bool) else None,
                 score=float(outcome), evaluation_seconds=time.perf_counter() - started)
    return score


def summarize(scores):
    elapsed = [row['elapsed_seconds'] for row in scores if row['elapsed_seconds'] is not None]
    correctness = [row['correct'] for row in scores]
    return {'declared_tasks': len(scores),
            'artifacts': sum(row['status'] != 'missing' for row in scores),
            'status_counts': dict(Counter(row['status'] for row in scores)),
            'valid_answers': sum(row['valid_answer'] for row in scores),
            'correct': sum(value is True for value in correctness),
            'accuracy': None if any(value is None for value in correctness) else statistics.mean(correctness),
            'mean_native_score': statistics.mean(row['score'] for row in scores),
            'timed_tasks': len(elapsed),
            'mean_sample_seconds': statistics.mean(elapsed) if elapsed else None}


def evaluate_run(run):
    contract = json.loads((run / 'protocol.json').read_text())
    tasks = contract['tasks']
    assert len(tasks) == contract['specification']['task_count'] and tasks
    assert len({task['task_id'] for task in tasks}) == len(tasks)
    definition = contract['environment_execution']
    assert definition == contract['specification']['environment_execution']
    module, name = definition['class'].rsplit('.', maxsplit=1)
    factory = getattr(importlib.import_module(module), name)
    workers = yaml.safe_load(contract['shared_config_text'])['runtime']['cpu_threads']
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        scores = list(pool.map(score_task, tasks,
            [run / f'task-{index:05d}.json' for index in range(len(tasks))],
            repeat(factory), repeat(definition['parameters']), repeat(contract['tools']),
            [run / 'offline_evaluation' / f'task-{index:05d}' for index in range(len(tasks))]))
    return {'run': str(run), 'observed_unix': time.time(),
            'evaluation_execution': {'workers': workers, 'process_start_method': 'spawn'},
            'stage': contract['specification']['stage'],
            'method': contract['specification']['method'],
            'task_source': contract['specification']['tasks'],
            'environment_execution': definition,
            'metric': 'Native environment evaluation of the submitted answer.',
            'failure_semantics': 'Every declared task counts. Missing artifacts, failed runs, and absent or invalid answers receive zero score. Numeric native scores are preserved without inventing a success threshold.',
            'summary': summarize(scores),
            'scores': scores}


def evaluate(run, output):
    report = evaluate_run(run)
    save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [evaluate_run(path if path.is_absolute() else ROOT / path) for path in args.run]
    save(args.output, {'runs': reports})
    print(json.dumps([{'run': report['run'], **report['summary']} for report in reports], indent=2))


if __name__ == '__main__':
    main()
