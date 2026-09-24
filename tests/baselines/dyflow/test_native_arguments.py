import ast
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import jsonschema

from baselines.common import dyflow
from baselines.common.errors import InvalidOutputError
from jev_spawn.infra.configuration import ROOT
from jev_spawn.infra.prompts import load_prompt


def test_unknown_native_argument_is_invalid_output_without_execution():
    config = json.loads((ROOT / 'configs/data/maze/runtime.json').read_text())
    schema = config['action_schema']
    original = deepcopy(schema)
    environment = SimpleNamespace(tools=[config['tool_name']],
        input_schemas={config['tool_name']: schema}, execute=Mock())
    source = ast.parse((ROOT / 'src/baselines/common/dyflow.py').read_text())
    function = next(node for node in ast.walk(source)
                    if isinstance(node, ast.FunctionDef) and node.name == '_process_output')
    namespace = dict(vars(dyflow), environment=environment,
        prompts=load_prompt('configs/baselines/common/schema/dyflow.json'))
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-tool-boundary>', 'exec'), namespace)
    output = json.dumps({'tool': config['tool_name'], 'arguments': {'direction': 'left'}})
    with pytest.raises(InvalidOutputError, match='tool arguments were rejected'):
        namespace['_process_output'](None, output, 'TOOL_CALL')
    environment.execute.assert_not_called()
    assert schema == original


@pytest.fixture
def final_submission(monkeypatch):
    schema = json.loads((ROOT / 'configs/data/maze/runtime.json').read_text())['answer_schema']

    def submit(answer, execution_error=None):
        class Workflow:
            def __init__(self, *args, **kwargs):
                self.design_history = []
                self.state = SimpleNamespace()

            def execute(self):
                return json.dumps(answer)

        monkeypatch.setattr(dyflow, 'load_core', lambda *args: {
            'DESIGN_STAGE_PROMPT': '', 'PROMPT_TEMPLATES': {},
            'InstructExecutorOperator': object, 'WorkflowExecutor': Workflow})
        environment = SimpleNamespace(reset=lambda: '', answer=None, actions=[],
                                      input_schemas={'finish': schema})

        def execute(tool, arguments):
            assert tool == 'finish'
            if execution_error is not None:
                raise execution_error
            dyflow.validate_arguments(arguments, schema)
            environment.answer = arguments

        environment.execute = execute
        return dyflow.solve({'task_id': 'submission'}, environment, Mock(),
                            {'upstream_commit': 'test'}, {'designer': '', 'tool': '', 'organize': ''})

    return submit


@pytest.mark.parametrize('answer', [
    [{'actions': ['up', 'up', 'up', 'right', 'right', 'right', 'right']}],
    {'actions': ['up']},
])
def test_invalid_final_submission_preserves_failure(final_submission, answer):
    original = deepcopy(answer)
    with pytest.raises(InvalidOutputError, match='final answer was rejected') as raised:
        final_submission(answer)
    assert isinstance(raised.value.__cause__, jsonschema.ValidationError)
    assert raised.value.trace['answer'] == original
    assert answer == original


def test_valid_final_submission_is_unchanged(final_submission):
    answer = {'actions': ['move up']}
    assert final_submission(answer)['answer'] == answer


def test_final_submission_infrastructure_error_propagates(final_submission):
    error = RuntimeError('Environment execution failed')
    with pytest.raises(RuntimeError) as raised:
        final_submission({'actions': ['move up']}, error)
    assert raised.value is error


def test_final_submission_environment_json_error_propagates(final_submission):
    error = json.JSONDecodeError('Environment internal JSON failed', 'broken', 0)
    with pytest.raises(json.JSONDecodeError) as raised:
        final_submission({'actions': ['move up']}, error)
    assert raised.value is error


def test_model_scalar_failure_preserves_original_value():
    config = json.loads((ROOT / 'configs/evaluation/native_context/llfbench_gridworld.json').read_text())
    schema = config['public_contract']['answer_schema']['properties']['actions']['items']
    with pytest.raises(InvalidOutputError, match='invalid JSON scalar') as raised:
        dyflow.validate_model_arguments('north', schema, 'tool arguments', {'output': 'north'})
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)
    assert raised.value.trace == {'output': 'north'}


def test_code_operator_unavailable_tool_is_invalid_output():
    source = ast.parse((ROOT / 'src/baselines/common/dyflow.py').read_text())
    operator = next(node for node in ast.walk(source)
                    if isinstance(node, ast.ClassDef) and node.name == 'Operator')
    function = next(node for node in operator.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_execute_code')
    run = ROOT / 'runs/native-baselines-formal-20260923/dyflow/textarena_lightsout'
    batches = json.loads((run / 'session-0000/batches.json').read_text())
    calls = [{'messages': batch['messages'][index], 'output': batch['texts'][index]}
             for batch in batches for index, identity in enumerate(batch['task_ids'])
             if identity == 'textarena_lightsout_seed61']
    log = (ROOT / 'logs/native_baselines/formal-dyflow-textarena_lightsout.log').read_text()
    code = log.split('  Code to execute:\n')[-1].split('\n\nERROR ENCOUNTERED:', 1)[0].rstrip()
    environment = SimpleNamespace(tools=['execute', 'finish'], observe=Mock())
    workflow = SimpleNamespace(design_history=calls)
    namespace = dict(vars(dyflow), environment=environment, calls=calls, workflow=workflow)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-code-boundary>', 'exec'), namespace)
    with pytest.raises(InvalidOutputError, match='unavailable tool: run_tests') as raised:
        namespace['_execute_code'](None, code)
    assert raised.value.trace == {'code': code, 'calls': calls, 'design_history': calls}
    environment.observe.assert_not_called()


def test_code_operator_environment_error_propagates():
    source = ast.parse((ROOT / 'src/baselines/common/dyflow.py').read_text())
    operator = next(node for node in ast.walk(source)
                    if isinstance(node, ast.ClassDef) and node.name == 'Operator')
    function = next(node for node in operator.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_execute_code')
    error = KeyError('environment internal failure')
    environment = SimpleNamespace(tools=['run_tests'], observe=Mock(side_effect=error))
    namespace = dict(vars(dyflow), environment=environment)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-code-boundary>', 'exec'), namespace)
    with pytest.raises(KeyError) as raised:
        namespace['_execute_code'](None, 'pass')
    assert raised.value is error
