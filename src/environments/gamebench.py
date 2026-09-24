from copy import copy, deepcopy
from dataclasses import asdict
from functools import cache
import importlib
import json
from pathlib import Path
from random import Random
import sys
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments

from data.task_context import render_context
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


@cache
def native_modules(game_module, opponent_module, action_module):
    return (importlib.import_module(game_module), importlib.import_module(opponent_module),
            importlib.import_module(action_module), Lock())


class GameBenchEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration_path = configuration
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        sys.path.insert(self.configuration['module_path_index'], str(ROOT / self.configuration['source_root']))
        self.game_module, self.opponent_module, self.action_module, self.native_lock = native_modules(
            self.configuration['game_module'], self.configuration['opponent_module'],
            self.configuration['action_module'])
        self.native = getattr(self.game_module, self.configuration['game_class'])()
        opponent = getattr(self.opponent_module, self.configuration['opponent_class'])
        self.native.init_game(opponent, opponent)
        seed = json.loads((ROOT / self.configuration['reproducibility']).read_text())['seed']
        self.rng = Random(seed)
        self.controlled_id = task['source']['controlled_agent_id']
        self.opponent_id, = [agent.agent_id for agent in self.native.agents
                            if agent.agent_id != self.controlled_id]
        self.controlled_agent = self.native.agents[self.controlled_id]
        if self.controlled_id != self.configuration['starting_agent_id']:
            self._opponent_turn()
        self.observation = self._observation()
        self.official_context = json.dumps(
            {'rules': asdict(self.native.rules), **self.observation}, **self.session['serialization'])
        self.initial_observation = self.observation
        self.done = self.native.game_is_over
        self.answer = None
        self.actions, self.tool_timings, self.executed_actions = [], [], []
        self.source_lock = Lock()
        self.observed_source_ids = set()
        self.tools = [self.configuration['tool_name'], self.session['submission_tool']]
        self.input_schemas = {self.configuration['tool_name']: self.configuration['action_schema'],
                              self.session['submission_tool']: self.display_answer_schema()}
        self.tool_definitions = {
            self.configuration['tool_name']: {
                'description': self.prompts['tool_description'],
                'arguments': {name: schema['type'] for name, schema in
                              self.configuration['action_schema']['properties'].items()}},
            self.session['submission_tool']: {
                'description': self.session_prompts['submission_description'],
                'arguments': {name: schema['type'] for name, schema in
                              self.display_answer_schema()['properties'].items()}}}
        self.context_text = render_context(self.official_context, self.display_answer_schema(),
                                           self.display_tool_interface(True), self.session['serialization'])

    def _opponent_turn(self):
        with self.native_lock:
            self.game_module.random = self.rng
            self.opponent_module.random = self.rng
            opponent = self.native.agents[self.opponent_id]
            observation, actions = self.native.get_observation(opponent)
            action = opponent.take_action(self.native.rules, observation, actions, self.native.show_state)
            self.native.update(action, actions, opponent)

    def _observation(self):
        observation, actions = self.native.get_observation(self.controlled_agent)
        return {'observation': observation.text, 'available_actions': asdict(actions),
                'done': self.native.game_is_over}

    def fork(self):
        child = copy(self)
        child.native = deepcopy(self.native)
        child.controlled_agent = child.native.agents[child.controlled_id]
        child.rng = deepcopy(self.rng)
        child.observation = deepcopy(self.observation)
        child.actions = deepcopy(self.actions)
        child.tool_timings = deepcopy(self.tool_timings)
        child.executed_actions = self.executed_actions.copy()
        child.answer = deepcopy(self.answer)
        child.observed_source_ids = set(self.observed_source_ids)
        child.source_lock = Lock()
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
        with self.native_lock:
            self.game_module.random = self.rng
            _, actions = self.native.get_observation(self.controlled_agent)
            action = getattr(self.action_module, self.configuration['action_class'])(
                action_id=arguments['action'])
            self.native.update(action, actions, self.controlled_agent)
        self.executed_actions.append(arguments['action'])
        if not self.native.game_is_over:
            self._opponent_turn()
        self.observation = self._observation()
        self.done = self.native.game_is_over
        if self.done:
            self.answer = {'actions': self.executed_actions.copy()}

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation = {'observation': self.session_prompts['submission_observation'], 'done': True}
        self.done = True

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step,
                    self.session['submission_tool']: self._finish}
        handlers[name](arguments)
        result = deepcopy(self.observation)
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments), 'result': result})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                 'finished_monotonic': time.perf_counter()})
        self.deadline()
        return result

    def execute(self, name, arguments):
        result = self.observe(name, arguments)
        return json.dumps(result, **self.session['serialization']), self.done

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        replay = type(self)(self.task, self.settings, self.directory, deadline=self.deadline,
                            evidence=self.evidence, configuration=self.configuration_path)
        for action in answer['actions']:
            replay._step({'action': action})
            if replay.native.game_is_over:
                break
        if not replay.native.game_is_over:
            return self.configuration['scores']['unfinished']
        if replay.native.winning_team is None:
            return self.configuration['scores']['draw']
        return self.configuration['scores'][str(replay.native.winning_team == replay.controlled_agent.team_id)]
