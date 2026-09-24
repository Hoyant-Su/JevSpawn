import ast
import json
from types import SimpleNamespace
from unittest.mock import patch

from baselines.common import lats
from baselines.common.config import SharedConfig
from baselines.common.transition_budget import TransitionBudget
from environments.ppnl import PPNLEnvironment
from jev_spawn.infra.configuration import ROOT


def test_unknown_tool_returns_protocol_feedback_without_native_execution():
    specification = json.loads((ROOT / 'configs/experiments/native_baselines/screen/ppnl_lats.json').read_text())
    task = json.loads((ROOT / specification['tasks']).read_text().splitlines()[0])
    environment = PPNLEnvironment(task, {}, ROOT / 'runs/cpu-lats-unknown-tool',
        deadline=lambda: 60, **specification['environment_execution']['parameters'])
    shared = SharedConfig.load(specification['shared_config'])
    contract = json.loads((ROOT / 'configs/environments/session.json').read_text())
    environment = TransitionBudget(environment, shared.runtime.max_turns, contract['submission_tool'],
                                   contract['initial_transition_depth'], [])
    source = ast.parse((ROOT / 'src/baselines/common/lats.py').read_text())
    step = next(node for node in ast.walk(source)
                if isinstance(node, ast.FunctionDef) and node.name == 'step')
    namespace = dict(vars(lats), environment=environment,
        current_node=[SimpleNamespace(environment=environment)], current_environment=[None])
    exec(compile(ast.Module(body=[step], type_ignores=[]), '<actual-lats-tool-boundary>', 'exec'), namespace)
    initial = environment.position
    with patch.object(PPNLEnvironment, 'execute') as execute:
        observation, reward, done, info = namespace['step'](None, 'impossible[{}]')
    execute.assert_not_called()
    assert 'Unavailable tool: impossible' in json.loads(observation)['error']['message']
    assert reward == 0 and done is False and info == {}
    assert environment.position == initial and environment.answer is None
    assert environment.actions[-1]['result']['done'] is False
    assert current_depth(namespace) == environment.transition_depth + 1


def current_depth(namespace):
    return namespace['current_environment'][0].transition_depth
