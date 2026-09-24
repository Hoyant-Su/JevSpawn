from copy import deepcopy
import hashlib
import json
import pickle
from unittest.mock import patch

import pytest

from environments.robotouille import RobotouilleEnvironment
from project_paths import ROOT


def test_official_invalid_action_boundary():
    specification = json.loads((ROOT / 'configs/experiments/native_expanded/robotouille_jevspawn.json').read_text())
    tasks = [json.loads(line) for line in (ROOT / specification['tasks']).read_text().splitlines()]
    invalid = json.loads((ROOT / 'tests/environments/robotouille_cases.json').read_text())['invalid_action']
    records = []
    for task in tasks:
        env = RobotouilleEnvironment(task, {}, ROOT / 'runs/robotouille-boundary-cpu-048',
            deadline=lambda: None, **specification['environment_execution']['parameters'])
        initial_state = pickle.dumps(env.native)
        initial_observation = env.api.LanguageSpace.state_to_language_description(env.native)
        initial_done, initial_answer = env.done, deepcopy(env.answer)
        valid_actions = env.native.get_valid_actions_and_str()[1]
        assert invalid not in valid_actions
        with patch.object(env.native, 'step', wraps=env.native.step) as native_step:
            rejected = env.observe('execute', {'action': invalid})
            native_step.assert_not_called()
        assert pickle.dumps(env.native) == initial_state
        assert env.executed_actions == []
        assert env.done == initial_done and env.answer == initial_answer
        assert rejected == {'observation':
            f"Error Feedback: The action '{invalid}' is not valid. Please provide a valid action.\n{initial_observation}",
            'done': initial_done}
        assert env.native.get_valid_actions_and_str()[1] == valid_actions
        assert env.actions == [{'tool': 'execute', 'arguments': {'action': invalid}, 'result': rejected}]
        assert len(env.tool_timings) == 1
        assert env.evaluate({'actions': [invalid]}) is False
        recoveries = []
        for command in valid_actions:
            child = env.fork()
            reference = deepcopy(env.native)
            actions, descriptions = reference.get_valid_actions_and_str()
            expected_done = reference.step([actions[descriptions.index(command)]])
            expected_observation = env.api.LanguageSpace.state_to_language_description(reference)
            with patch.object(child.native, 'step', wraps=child.native.step) as native_step:
                result = child.observe('execute', {'action': command})
                native_step.assert_called_once()
            assert result == {'observation': expected_observation, 'done': expected_done}
            assert child.native.predicates == reference.predicates
            assert child.native.current_player == reference.current_player
            assert repr(child.native.special_effects) == repr(reference.special_effects)
            assert child.executed_actions == [command]
            assert len(child.actions) == 2 and len(child.tool_timings) == 2
            assert child.evaluate({'actions': [command]}) == expected_done
            assert child.evaluate({'actions': [invalid, command]}) is False
            recoveries.append({'action': command, 'native_step_calls': 1, 'goal_reached': expected_done})
        assert pickle.dumps(env.native) == initial_state
        records.append({'task_id': task['task_id'], 'native_state_sha256': hashlib.sha256(initial_state).hexdigest(),
            'rejection_preserved_state': True, 'rejected_native_step_calls': 0,
            'invalid_final_replay': False, 'legal_recoveries': recoveries})
    report = {
        'scope': 'CPU regression using real official native environments; no model inference or substituted actions.',
        'source_commit': 'ddec56bbb864c897d833028b62f190ca791efa2b',
        'sources': {
            'official_react_rejection': 'https://github.com/portal-cornell/robotouille/blob/ddec56bbb864c897d833028b62f190ca791efa2b/agents/ReAct_agent.py#L256-L261',
            'native_step': 'external/Robotouille-native-pinned/backend/state.py:441',
            'native_language_observation': 'external/Robotouille-native-pinned/robotouille/env.py:383'},
        'test': 'tests/environments/test_robotouille_boundary_048.py',
        'task_count': len(records), 'records': records}
    target = ROOT / 'results/validation/native_robotouille_boundary_048.json'
    target.write_text(json.dumps(report, indent=2) + '\n')


def test_unexpected_native_error_propagates():
    specification = json.loads((ROOT / 'configs/experiments/native_context/robotouille_jevspawn.json').read_text())
    tasks = [json.loads(line) for line in (ROOT / specification['tasks']).read_text().splitlines()]
    for task in tasks:
        env = RobotouilleEnvironment(task, {}, ROOT / 'runs/robotouille-boundary-cpu-048',
            deadline=lambda: None, **specification['environment_execution']['parameters'])
        for command in env.native.get_valid_actions_and_str()[1]:
            with patch.object(env.native, 'step', side_effect=RuntimeError('native computation failed')):
                with pytest.raises(RuntimeError, match='native computation failed'):
                    env.observe('execute', {'action': command})
        assert env.executed_actions == [] and env.actions == []
