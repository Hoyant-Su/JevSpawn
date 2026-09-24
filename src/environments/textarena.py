from copy import copy, deepcopy
import json
from pathlib import Path
import random
from threading import Lock
import time

import jsonschema
import textarena as ta

from environments.arguments import validate_arguments
import numpy as np

from data.task_context import render_context
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


_NATIVE_RESET_LOCK = Lock()


class TextArenaEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        # Native constructors and reset use global RNGs; isolate concurrent task initialization.
        with _NATIVE_RESET_LOCK:
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            try:
                random.seed(task['source']['seed'])
                np.random.seed(task['source']['seed'])
                self.native = ta.make(task['source']['env_id'], **task['source']['native_parameters'])
                self.native.reset(num_players=self.configuration['num_players'], seed=task['source']['seed'])
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
        self.player_id, self.official_context = self.native.get_observation()
        self.initial_observation = self.observation = self.official_context
        self.initial_native = deepcopy(self.native)
        self.done, self.answer = self.native.state.done, None
        self.actions, self.tool_timings, self.executed_actions = [], [], []
        self.source_lock, self.observed_source_ids = Lock(), set()
        self.tools = [self.configuration['tool_name'], self.session['submission_tool']]
        self.input_schemas = {self.configuration['tool_name']: self.configuration['action_schema'],
                              self.session['submission_tool']: self.display_answer_schema()}
        descriptions = {self.configuration['tool_name']: self.prompts['tool_description'],
                        self.session['submission_tool']: self.session_prompts['submission_description']}
        self.tool_definitions = {name: {'description': descriptions[name],
            'arguments': {key: value['type'] for key, value in schema['properties'].items()}}
            for name, schema in self.input_schemas.items()}
        self.context_text = render_context(self.official_context, self.display_answer_schema(),
                                           self.display_tool_interface(True), self.session['serialization'])

    def fork(self):
        child = copy(self)
        child.native = deepcopy(self.native)
        child.actions, child.tool_timings = deepcopy(self.actions), deepcopy(self.tool_timings)
        child.executed_actions, child.answer = self.executed_actions.copy(), deepcopy(self.answer)
        child.observed_source_ids, child.source_lock = set(self.observed_source_ids), Lock()
        return child

    def display_answer_schema(self):
        return self.task['answer_schema']

    def display_tool_schema(self, name):
        return self.input_schemas[name]

    def display_tool_output_schema(self, name):
        return self.configuration['observation_schema']

    def display_tool_interface(self, include_finish):
        return {name: {**definition, 'input_schema': self.input_schemas[name]}
                for name, definition in self.tool_definitions.items()}

    def context(self, include_finish, serialization):
        return self.context_text

    def reset(self):
        return self.context_text

    def _step(self, arguments):
        self.done, _ = self.native.step(arguments['action'])
        self.executed_actions.append(arguments['action'])
        # The agent runtime owns history; render only unread native observations.
        self.native.full_observations.clear()
        _, self.observation = self.native.get_observation()
        if self.done:
            rewards, game_info = self.native.close()
            self.observation += self.configuration['observation_separator'] + json.dumps(
                {'rewards': rewards, 'game_info': game_info}, **self.session['serialization'])
            self.answer = {'actions': self.executed_actions.copy()}

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation, self.done = self.session_prompts['submission_observation'], True

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step,
                    self.session['submission_tool']: self._finish}
        handlers[name](arguments)
        result = {'observation': self.observation, 'done': self.done}
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments), 'result': result})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                 'finished_monotonic': time.perf_counter()})
        self.deadline()
        return result

    def execute(self, name, arguments):
        result = self.observe(name, arguments)
        return result['observation'], self.done

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        replay = deepcopy(self.initial_native)
        for action in answer['actions']:
            if replay.state.done:
                return self.configuration['invalid_sequence_score']
            replay.step(action)
        if not replay.state.done:
            return self.configuration['unfinished_score']
        rewards, _ = replay.close()
        return rewards[self.player_id]
