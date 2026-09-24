from copy import deepcopy
import json
from pathlib import Path

from environments.robotouille import RobotouilleEnvironment


def test_real_boil_and_cut_tasks_have_independent_pending_effects(tmp_path):
    tasks = {task['task_id']: task for task in map(json.loads, Path(
        'data/qualification/native_screen/robotouille/tasks.jsonl').read_text().splitlines())}

    def build(identity):
        return RobotouilleEnvironment(tasks[identity], {}, tmp_path / identity,
            deadline=lambda: 300, configuration='configs/environments/robotouille_runtime.json')

    boiling = build('robotouille_base_boil_water_seed42')
    cutting = build('robotouille_base_cut_seed42')
    isolated = deepcopy(cutting.native)
    assert boiling.native.special_effects is not cutting.native.special_effects
    boiling.execute('execute', {'action': "Boil pot1's contents on stove1 using robot1"})
    assert boiling.native.special_effects
    assert not cutting.native.special_effects
    action = 'Cut lettuce1 on board1 using robot1'
    cutting.execute('execute', {'action': action})
    native_actions, descriptions = isolated.get_valid_actions_and_str()
    isolated.step([native_actions[descriptions.index(action)]])
    assert cutting.native.predicates == isolated.predicates
    assert cutting.native.special_effects == isolated.special_effects
    assert cutting.native.is_goal_reached() == isolated.is_goal_reached()
