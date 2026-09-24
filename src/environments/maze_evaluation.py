from copy import deepcopy
import json

from environments.maze import MazeEnvironment
from project_paths import ROOT


class MazeScorer:
    @staticmethod
    def evaluate(native, initial_history, text_type, actions, settings):
        replay, history = deepcopy(native), deepcopy(initial_history)
        done = False
        for action in actions:
            if done:
                return False
            history, _, done = replay.step(history + (text_type(action + settings['action_suffix'], True),))
        return done and tuple(replay.position) == tuple(replay.goal)

    def __init__(self, settings):
        self.settings = settings
        self.tasks = {task['task_id']: task for task in map(
            json.loads, (ROOT / settings['tasks']).read_text().splitlines())}

    def score(self, task, result, reference):
        identity = task['task_id']
        assert identity == result['task_id'] == reference['task_id'], 'Maze task identities do not match.'
        assert task == self.tasks[identity], 'Maze task differs from its declared original episode.'
        detail = {'task_id': identity, 'status': result['status']}
        if result['status'] != self.settings['completed_status']:
            assert result['status'] in self.settings['noncompleted_statuses'], 'Unsupported maze run status.'
            return {**detail, 'score': self.settings['noncompleted_score'], 'replayed': False,
                    'native_success': None, 'terminal': None, 'action_count': None, 'total_reward': None}
        environment = MazeEnvironment(task, {}, ROOT / self.settings['work_directory'],
            deadline=lambda: None, configuration=self.settings['environment_configuration'])
        for recorded in result['actions']:
            observation, _ = environment.execute(recorded['tool'], recorded['arguments'])
            assert json.loads(observation) == recorded['result'], (
                f'Recorded maze transition differs from native replay for {identity}.')
        assert environment.done, f'Completed maze record has no native terminal state: {identity}.'
        assert result['answer'] == environment.answer, (
            f'Recorded maze answer contradicts native terminal state for {identity}.')
        success = environment.evaluate(environment.answer)
        return {**detail, 'score': self.settings['score_by_success'][success], 'replayed': True,
                'native_success': success, 'terminal': environment.done,
                'action_count': len(environment.actions),
                'total_reward': sum((action['result']['reward'] for action in environment.actions),
                                    self.settings['reward_sum_start'])}
