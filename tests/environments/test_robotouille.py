import importlib
import json
import random
import sys

from jev_spawn.infra.configuration import resolve_symbol
from project_paths import ROOT


def test_robotouille_native_context_fork_transition_and_replay():
    spec = json.loads((ROOT / 'configs/experiments/native_context/robotouille_jevspawn.json').read_text())
    definition = spec['environment_execution']
    factory = resolve_symbol(definition['class'])
    own_package = sys.modules['environments']
    tasks = [json.loads(line) for line in (ROOT / spec['tasks']).read_text().splitlines()]
    controls = json.loads((ROOT / 'tests/environments/robotouille_cases.json').read_text())
    evidence = []
    for method in controls['methods']:
        other = json.loads((ROOT / f'configs/experiments/native_context/robotouille_{method}.json').read_text())
        assert other['tasks'] == spec['tasks']
        assert other['environment_execution'] == definition
        assert other['shared_config'] == spec['shared_config']
    for task in tasks:
        rng = random.getstate()
        env = factory(task, {}, ROOT / 'runs/robotouille-cpu-validation', deadline=lambda: None, **definition['parameters'])
        assert random.getstate() == rng
        assert sys.modules['environments'] is own_package
        assert env.reset() == task['context']
        assert env.evaluate({'actions': []}) is False
        assert env.evaluate({'actions': [controls['invalid_action']]}) is False
        parent_observation = env.observation
        branches = []
        for action in env.native.get_valid_actions_and_str()[1]:
            child = env.fork()
            observation = child.observe('execute', {'action': action})
            assert observation['observation']
            assert env.api.LanguageSpace.state_to_language_description(env.native) == parent_observation
            assert env.executed_actions == []
            answer = {'actions': [action]}
            assert env.evaluate(answer) == child.native.is_goal_reached()
            branches.append({'action': action, 'observation': observation,
                             'native_goal_reached': child.native.is_goal_reached(),
                             'replay_score': env.evaluate(answer)})
        assert any(branch['native_goal_reached'] for branch in branches)
        assert any(not branch['native_goal_reached'] for branch in branches)
        evidence.append({'task_id': task['task_id'], 'official_context': env.official_context,
                         'context_shared_across_methods': True, 'fork_isolation': True,
                         'invalid_action_replay_score': False, 'branches': branches})
    importlib.import_module('environments.textarena')
    target = ROOT / 'results/validation/robotouille_shared_evaluator_20260923.json'
    target.write_text(json.dumps({'scope': 'CPU positive and negative controls through official native state transitions; no model inference.',
                                 'source_commit': tasks[0]['source']['commit'], 'tasks': evidence}, indent=2) + '\n')
