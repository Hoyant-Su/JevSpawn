import ast
import json
import os
from pathlib import Path
import stat
from types import ModuleType

import pytest

from baselines.common import persistence
from jev_spawn.infra.prompts import load_prompt


# Exercise the real host runner without importing its GPU backend on a CPU-only test host.
run = ModuleType('resume_test_runner')
runner_source = Path('src/baselines/common/run.py')
runner_tree = ast.parse(runner_source.read_text())
runner_tree.body = [node for node in runner_tree.body if not (
    isinstance(node, ast.ImportFrom) and node.module == 'baselines.common.runtime')]
exec(compile(runner_tree, str(runner_source), 'exec'), run.__dict__)


@pytest.fixture
def tasks():
    return [json.loads(line) for line in Path(
        'data/qualification/native_screen/ppnl/tasks.jsonl').read_text().splitlines()[:2]]


def result(task, status='completed'):
    record = {'task_id': task['task_id'], 'status': status, 'answer': None, 'elapsed_seconds': 0.5}
    if status != 'completed':
        record['error'] = 'Recorded lifecycle failure.'
    return record


def test_save_syncs_file_then_replaces_then_syncs_directory(tmp_path, monkeypatch):
    events = []
    original_sync, original_replace = os.fsync, os.replace

    def sync(fd):
        events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
        original_sync(fd)

    def replace(source, destination):
        events.append('replace')
        original_replace(source, destination)

    monkeypatch.setattr(os, 'fsync', sync)
    monkeypatch.setattr(os, 'replace', replace)
    path = tmp_path / 'nested' / 'task.json'
    persistence.save(path, {'value': 'real committed payload'})
    assert events == ['file', 'replace', 'directory']
    assert json.loads(path.read_text()) == {'value': 'real committed payload'}


def test_interruption_before_rename_keeps_committed_output(tmp_path, monkeypatch):
    path = tmp_path / 'task.json'
    persistence.save(path, {'version': 'committed'})

    def interrupt(source, destination):
        raise OSError('Injected interruption before commit.')

    monkeypatch.setattr(os, 'replace', interrupt)
    with pytest.raises(OSError, match='Injected interruption'):
        persistence.save(path, {'version': 'in flight'})
    assert json.loads(path.read_text()) == {'version': 'committed'}
    assert json.loads(path.with_suffix('.json.partial').read_text()) == {'version': 'in flight'}


@pytest.mark.parametrize('status', ['completed', 'timeout', 'invalid_output', 'limit_exceeded'])
def test_resume_preserves_committed_outcomes_and_restarts_only_uncommitted(tasks, tmp_path, status):
    persistence.save(tmp_path / 'task-00000.json', result(tasks[0], status))
    (tmp_path / 'task-00001.json.partial').write_text('{interrupted')
    assert persistence.pending_tasks(tasks, tmp_path) == [tasks[1]]


def test_resume_rejects_wrong_identity_invalid_status_and_corrupt_json(tasks, tmp_path):
    path = tmp_path / 'task-00000.json'
    persistence.save(path, result(tasks[1]))
    with pytest.raises(AssertionError, match='identity'):
        persistence.pending_tasks(tasks, tmp_path)
    persistence.save(path, {**result(tasks[0]), 'status': 'running'})
    with pytest.raises(AssertionError, match='terminal lifecycle'):
        persistence.pending_tasks(tasks, tmp_path)
    path.write_text('{corrupt committed output')
    with pytest.raises(json.JSONDecodeError):
        persistence.pending_tasks(tasks, tmp_path)


@pytest.mark.parametrize('method', ['react', 'latentmas'])
def test_completed_native_resume_validates_contract_before_any_runtime(tasks, tmp_path, monkeypatch, method):
    source = tmp_path / 'tasks.jsonl'
    source.write_text(''.join(json.dumps(task) + '\n' for task in tasks))
    specification = json.loads(Path(f'configs/experiments/native_baselines/screen/ppnl_{method}.json').read_text())
    specification.update(tasks=str(source), task_count=len(tasks))
    definition = specification['environment_execution']
    method_config = run.read(specification['method'])
    contract = {'specification': specification, 'method': method_config,
        'prompts': load_prompt(method_config['prompts']), 'tools': run.read(specification['environment']),
        'tasks': tasks, 'shared_config_text': Path(specification['shared_config']).read_text(),
        'inference': run.read('configs/inference/shared_service.json'),
        'session_contract': run.read('configs/environments/session.json'),
        'environment_execution': definition}
    output = tmp_path / 'run'
    persistence.save(output / 'protocol.json', contract)
    for index, task in enumerate(tasks):
        persistence.save(output / f'task-{index:05d}.json', result(task, 'timeout'))

    def no_runtime(*args, **kwargs):
        pytest.fail('Committed task resume must not construct an inference runtime or environment.')

    monkeypatch.setattr(run, 'InferenceRuntime', no_runtime, raising=False)
    run.run_with_environment(specification, output, None, environment_factory=no_runtime,
                             environment_contract={'environment_execution': definition})
    assert run.read(output / 'completion.json')['task_ids'] == [task['task_id'] for task in tasks]
    contract['shared_config_text'] += '\nchanged configuration\n'
    persistence.save(output / 'protocol.json', contract)
    with pytest.raises(AssertionError):
        run.run_with_environment(specification, output, None, environment_factory=no_runtime,
                                 environment_contract={'environment_execution': definition})
