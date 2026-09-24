import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from baselines.common.foldagent import Environment
from baselines.common.config import SharedConfig
from baselines.common.errors import InvalidOutputError
from baselines.common.llmcompiler import invoke, solve_with_models
from baselines.common.react import EnvironmentAdapter
from jev_spawn.infra.prompts import load_prompt
from tests.baselines.agentprune.test_error_feedback import runtime


CASES = json.loads(Path(__file__).with_name('configs').joinpath('transport_feedback.json').read_text())


def test_react_rejected_commands_continue_in_native_environment(runtime):
    environment, _, _ = runtime
    adapter = EnvironmentAdapter(environment, CASES['react_prompts'])
    for command in CASES['invalid_frames']:
        initial = environment.position
        depth = environment.transition_depth
        observation, _, done, _ = adapter.step(command)
        assert 'error' in json.loads(observation)
        assert not done and environment.position == initial
        assert environment.transition_depth > depth
    expected = environment.fork().execute(CASES['tool'], CASES['valid_payload'])
    observation, _, done, _ = adapter.step(CASES['valid_action'])
    assert (observation, done) == expected


@pytest.mark.parametrize('case', ['invalid_xml_call', 'invalid_xml_finish', 'unknown_xml_call'])
def test_foldagent_rejection_preserves_state_then_accepts_valid_call(runtime, case):
    environment, _, _ = runtime
    initial = environment.position
    adapter = Environment(environment, lambda response: CASES[response])
    response = asyncio.run(adapter.run_action(case))
    assert 'error' in json.loads(response['observation'])
    assert environment.position == initial and not environment.done
    expected = environment.fork().execute(CASES['tool'], CASES['valid_payload'])
    corrected = asyncio.run(adapter.run_action('valid_xml_call'))
    assert corrected['observation'] == expected[0]


def test_compiler_argument_errors_are_recorded_and_valid_calls_match(runtime):
    environment, _, _ = runtime
    initial = environment.position
    for values in [(), tuple(CASES['invalid_payload'].values())]:
        response = asyncio.run(invoke(*values, environment=environment,
            name=CASES['tool'], fields=CASES['compiler_fields']))
        assert 'error' in json.loads(response)
        assert environment.position == initial
    expected = environment.fork().execute(CASES['tool'], CASES['valid_payload'])
    actual = asyncio.run(invoke(*CASES['valid_payload'].values(), environment=environment,
        name=CASES['tool'], fields=CASES['compiler_fields']))
    assert actual == expected[0]


def test_native_failures_propagate_through_all_transports(runtime):
    environment, _, _ = runtime
    react = EnvironmentAdapter(environment, CASES['react_prompts'])
    fold = Environment(environment, lambda response: CASES['valid_xml_call'])
    with patch.object(environment._environment, 'execute', side_effect=RuntimeError(CASES['infra_error'])):
        with pytest.raises(RuntimeError, match=CASES['infra_error']):
            react.step(CASES['valid_action'])
        with pytest.raises(RuntimeError, match=CASES['infra_error']):
            asyncio.run(fold.run_action('valid_xml_call'))
        with pytest.raises(RuntimeError, match=CASES['infra_error']):
            asyncio.run(invoke(*CASES['valid_payload'].values(), environment=environment,
                name=CASES['tool'], fields=CASES['compiler_fields']))


def test_rejected_compiler_final_answer_cannot_be_recorded_completed(runtime):
    environment, _, _ = runtime
    method = json.loads(Path(CASES['compiler_method']).read_text())
    shared = SharedConfig.load(CASES['shared_config'])

    class FinishedCompiler:
        def __init__(self, **kwargs):
            self.planner = SimpleNamespace(output_parser=None)

        async def arun(self, question):
            return CASES['invalid_final']

    with pytest.raises(InvalidOutputError) as failure:
        solve_with_models({}, environment, None, shared.method_settings(method['settings']),
            load_prompt(method['prompts']), lambda **kwargs: None, False, FinishedCompiler)
    assert 'ValidationError' in str(failure.value)
    assert environment.answer is None and not environment.done
    assert failure.value.trace['actions'][-1]['result']['error']['type'] == 'ValidationError'
