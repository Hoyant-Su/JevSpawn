from copy import deepcopy


class MazeScorer:
    @staticmethod
    def evaluate(native, initial_history, text_type, actions, settings):
        replay, history = deepcopy(native), deepcopy(initial_history)
        done = False
        for action in actions:
            if done:
                return False
            history, _, done = replay.step(history + (text_type(action + settings["action_suffix"], True),))
        return done and tuple(replay.position) == tuple(replay.goal)
