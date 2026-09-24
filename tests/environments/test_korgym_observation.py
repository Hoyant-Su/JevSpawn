import json
from pathlib import Path

from environments.korgym import KORGymEnvironment


CASES = json.loads(Path('tests/fixtures/environments/korgym_observation.json').read_text())


def environment(case):
    tasks = (json.loads(line) for line in Path(CASES['task_source'].format(**case)).read_text().splitlines())
    task = next(task for task in tasks if task['source']['seed'] == CASES['seed'])
    return KORGymEnvironment(task, {}, CASES['scratch'], deadline=lambda: None,
                             configuration=case['configuration'])


def test_dynamic_fields_preserve_native_observation_and_branch_state():
    for case in CASES['tracks']:
        root = environment(case)
        original = json.dumps(root.item, sort_keys=True)
        structured, textual = root.fork(), root.fork()
        result = structured.observe(root.configuration['tool_name'], {'action': case['action']})
        text, done = textual.execute(root.configuration['tool_name'], {'action': case['action']})
        fields = json.loads(result['observation'])
        expected = structured._call('print_board', structured.item)
        assert structured.observation_format.template.format(**fields) == expected
        assert (text, done) == (result['observation'], result['done'])
        assert structured.item['board'] == textual.item['board']
        assert result['score'] == structured.item['score']
        assert len(result['observation']) < len(expected)
        assert json.dumps(root.item, sort_keys=True) == original


def test_native_horizon_records_real_score_without_model_submission():
    case = next(case for case in CASES['tracks'] if case['name'] == CASES['horizon_track'])
    root = environment(case)
    branch = root.fork()
    for _ in range(branch.episode['max_steps']):
        assert not branch.done
        result = branch.observe(branch.configuration['tool_name'], {'action': case['action']})
        assert result['done'] == (len(branch.executed_actions) == branch.episode['max_steps'])
    assert branch.done
    assert branch.answer == {'actions': branch.executed_actions}
    assert branch.item['score'] > root.item['score']
    assert root.evaluate(branch.answer) == branch.item['score']
    assert not root.done and not root.executed_actions
