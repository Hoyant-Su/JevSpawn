import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from baselines.common.agentprune_v2 import Provider
from baselines.common.config import SharedConfig
from baselines.common.transition_budget import TransitionBudget
from environments.ppnl import PPNLEnvironment
from jev_spawn.infra.prompts import load_prompt


@pytest.fixture
def runtime(tmp_path):
    specification = json.loads(Path('configs/experiments/native_baselines/screen/ppnl_lats.json').read_text())
    task = json.loads(Path(specification['tasks']).read_text().splitlines()[0])
    shared = SharedConfig.load(specification['shared_config'])
    method = json.loads(Path('configs/baselines/common/methods/agentprune_v2.json').read_text())
    settings = shared.method_settings(method['settings'])
    contract = json.loads(Path('configs/environments/session.json').read_text())
    native = PPNLEnvironment(task, {}, tmp_path, deadline=lambda: shared.runtime.sample_timeout_seconds,
                             **specification['environment_execution']['parameters'])
    environment = TransitionBudget(native, shared.runtime.max_turns, contract['submission_tool'],
                                   contract['initial_transition_depth'], [])
    return environment, settings, load_prompt(method['prompts'])


def provider(runtime, outputs, decision):
    environment, settings, prompts = runtime
    pending = iter(outputs)
    records = []
    return Provider(SimpleNamespace(id='recorded-test-node', role='test-role'), decision,
                    environment, lambda *args: [next(pending)], settings, prompts, records), records


def test_invalid_final_structure_is_observed_then_corrected(runtime):
    environment, _, _ = runtime
    agent, records = provider(runtime, ['[]', '{"actions":"left down down"}'], True)
    answer = asyncio.run(agent.agen([{'role': 'system', 'content': 'Final decision.'},
                                    {'role': 'user', 'content': environment.reset()}]))
    assert environment.done
    assert json.loads(answer) == environment.answer
    assert environment.evaluate(environment.answer)
    assert environment.transition_depth == 1
    assert 'ValidationError' in records[1]['messages'][-1]['content']


def test_invalid_tool_envelope_is_observed_then_analysis_continues(runtime):
    environment, _, prompts = runtime
    initial = environment.position
    agent, records = provider(runtime, [prompts['tool_prefix'] + '{', 'Completed analysis.'], False)
    result = asyncio.run(agent.agen([{'role': 'system', 'content': 'Analyze.'}]))
    assert result == 'Completed analysis.'
    assert environment.position == initial
    assert environment.answer is None
    assert environment.transition_depth == 1
    assert 'JSONDecodeError' in records[1]['messages'][-1]['content']


def test_native_infrastructure_error_is_not_converted_to_feedback(runtime):
    environment, _, prompts = runtime
    agent, _ = provider(runtime, [prompts['tool_prefix'] + '{"tool":"execute","arguments":{"actions":"left"}}'], False)
    with patch.object(environment._environment, 'execute', side_effect=RuntimeError('Backend unavailable')):
        with pytest.raises(RuntimeError, match='Backend unavailable'):
            asyncio.run(agent.agen([{'role': 'system', 'content': 'Analyze.'}]))
    assert environment.actions == []
