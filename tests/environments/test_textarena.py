import json
import pickle
import random

import numpy as np
import textarena as ta

from jev_spawn.infra.configuration import resolve_symbol
from project_paths import ROOT


def test_public_terminal_feedback_matches_native_close_for_both_interfaces():
    case = json.loads((ROOT / 'tests/environments/textarena_cases.json').read_text())['terminal_case']
    spec = json.loads((ROOT / case['specification']).read_text())
    tasks = [json.loads(line) for line in (ROOT / spec['tasks']).read_text().splitlines()]
    task = next(task for task in tasks if task['task_id'] == case['task_id'])
    definition = spec['environment_execution']
    env = resolve_symbol(definition['class'])(task, {}, ROOT / case['directory'],
        deadline=lambda: None, **definition['parameters'])
    original = pickle.dumps(env.native)
    structured, textual = env.fork(), env.fork()
    native = env.fork().native
    records = []
    previous = env.official_context
    for action in case['actions']:
        done, info = native.step(action)
        _, cumulative = native.get_observation()
        assert cumulative.startswith(previous)
        expected = cumulative[len(previous):]
        previous += expected
        assert previous == cumulative
        if done:
            rewards, game_info = native.close()
            expected += env.configuration['observation_separator'] + json.dumps(
                {'rewards': rewards, 'game_info': game_info}, **env.session['serialization'])
        observed = structured.observe(env.configuration['tool_name'], {'action': action})
        text, stopped = textual.execute(env.configuration['tool_name'], {'action': action})
        assert observed == {'observation': expected, 'done': done}
        assert (text, stopped) == (expected, done)
        assert pickle.dumps(env.native) == original
        records.append({'action': action, 'native_step_info': info, 'observation': observed})
    assert structured.done and textual.done
    assert structured.answer == textual.answer
    assert env.evaluate(structured.answer) == native.close()[0][env.player_id]
    (ROOT / case['evidence']).write_text(json.dumps({
        'scope': 'CPU official replay; no model inference or task-success claim.',
        'task_id': task['task_id'], 'interfaces_equal': True, 'parent_unchanged': True,
        'incremental_observations_reconstruct_native_history': True, 'records': records}, indent=2) + '\n')


def test_native_textarena_shared_context_fork_and_score():
    controls = json.loads((ROOT / 'tests/environments/textarena_cases.json').read_text())
    evidence = []
    for track, first_action in controls['tracks'].items():
        spec = json.loads((ROOT / f'configs/experiments/native_context/textarena_{track}_jevspawn.json').read_text())
        definition = spec['environment_execution']
        factory = resolve_symbol(definition['class'])
        tasks = [json.loads(line) for line in (ROOT / spec['tasks']).read_text().splitlines()]
        for method in controls['methods']:
            other = json.loads((ROOT / f'configs/experiments/native_context/textarena_{track}_{method}.json').read_text())
            assert other['tasks'] == spec['tasks']
            assert other['environment_execution'] == definition
            assert other['shared_config'] == spec['shared_config']
        for task in tasks:
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            env = factory(task, {}, ROOT / 'runs/textarena-cpu-validation', deadline=lambda: None, **definition['parameters'])
            assert random.getstate() == python_rng
            assert pickle.dumps(np.random.get_state()) == pickle.dumps(numpy_rng)
            assert env.reset() == task['context']
            random.seed(task['source']['seed'])
            np.random.seed(task['source']['seed'])
            official = ta.make(task['source']['env_id'], **task['source']['native_parameters'])
            official.reset(num_players=env.configuration['num_players'], seed=task['source']['seed'])
            _, official_context = official.get_observation()
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            assert env.official_context == official_context
            assert pickle.dumps(env.native.state.game_state) == pickle.dumps(official.state.game_state)
            before = pickle.dumps(env.native)
            child = env.fork()
            observation = child.observe('execute', {'action': first_action})
            official_done, _ = official.step(first_action)
            _, official_observation = official.get_observation()
            assert observation == {'observation': official_observation[len(official_context):], 'done': official_done}
            assert official_context + observation['observation'] == official_observation
            assert pickle.dumps(child.native.state.game_state) == pickle.dumps(official.state.game_state)
            assert pickle.dumps(env.native) == before
            assert not env.executed_actions
            while not child.done:
                child.observe('execute', {'action': controls['invalid_action']})
            native_rewards, native_info = child.native.close()
            score = env.evaluate(child.answer)
            assert score == native_rewards[child.player_id]
            assert native_info[child.player_id]['invalid_move'] is True
            assert env.evaluate({'actions': []}) == env.configuration['unfinished_score']
            evidence.append({'task_id': task['task_id'], 'env_id': task['source']['env_id'],
                'official_wrappers': [wrapper.__name__ for wrapper in ta.envs.registration.ENV_REGISTRY[task['source']['env_id']].default_wrappers],
                'registered_initial_context_equal': True, 'registered_history_reconstructed_exactly': True,
                'shared_context_verified': True, 'fork_isolation_verified': True,
                'first_action': first_action, 'first_observation': observation,
                'negative_terminal_score': score, 'native_terminal_info': native_info,
                'negative_terminal_is_success': False})
    target = ROOT / 'results/validation/textarena_shared_evaluator_stage039_20260923.json'
    target.write_text(json.dumps({'scope': 'CPU native environment integration; no model inference.',
                                 'tasks': evidence}, ensure_ascii=False, indent=2) + '\n')


def test_lightsout_positive_native_replay():
    spec = json.loads((ROOT / 'configs/experiments/native_context/textarena_lightsout_jevspawn.json').read_text())
    definition = spec['environment_execution']
    factory = resolve_symbol(definition['class'])
    tasks = [json.loads(line) for line in (ROOT / spec['tasks']).read_text().splitlines()]
    evidence = []
    for task in tasks:
        env = factory(task, {}, ROOT / 'runs/textarena-cpu-validation', deadline=lambda: None, **definition['parameters'])
        native = env.native
        size = native.size
        target = np.array(native.state.game_state['grid'], dtype=np.uint8).reshape(-1)
        columns = []
        for row in range(size):
            for col in range(size):
                board = [[False] * size for _ in range(size)]
                native._toggle_lights(board, row, col)
                columns.append(np.array(board, dtype=np.uint8).reshape(-1))
        matrix = np.column_stack([*columns, target])
        pivot_row, pivots = 0, []
        for col in range(size * size):
            candidates = np.flatnonzero(matrix[pivot_row:, col])
            if not len(candidates):
                continue
            index = pivot_row + candidates[0]
            matrix[[pivot_row, index]] = matrix[[index, pivot_row]]
            for row in range(size * size):
                if row != pivot_row and matrix[row, col]:
                    matrix[row] ^= matrix[pivot_row]
            pivots.append(col)
            pivot_row += 1
        solution = np.zeros(size * size, dtype=np.uint8)
        solution[pivots] = matrix[:pivot_row, -1]
        actions = [f'[{index // size} {index % size}]' for index in np.flatnonzero(solution)]
        replay = env.fork()
        for action in actions:
            replay.observe('execute', {'action': action})
        assert replay.native._is_solved(replay.native.state.game_state['grid'])
        assert replay.done
        assert env.evaluate(replay.answer) == 1.0
        evidence.append({'task_id': task['task_id'], 'source': 'Test-only GF(2) solve using official native toggle operator; not supplied to model.',
                         'actions': actions, 'native_reward': env.evaluate(replay.answer)})
    path = ROOT / 'results/validation/textarena_positive_controls_stage039_20260923.json'
    path.write_text(json.dumps({'scope': 'CPU positive controls, not model inference.', 'tasks': evidence}, indent=2) + '\n')
