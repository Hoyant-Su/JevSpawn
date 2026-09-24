from copy import copy, deepcopy
from functools import cache
import importlib.util
import json
from pathlib import Path
from random import Random
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments
from environments.observation import FormattedObservation

from data.task_context import render_context
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


@cache
def native_module(source, name):
    specification = importlib.util.spec_from_file_location(name, ROOT / source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module, Lock()


class KORGymEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration_path = configuration
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.episode = json.loads((ROOT / self.configuration['episode']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        self.native, self.native_lock = native_module(
            self.configuration['native_source'], self.configuration['native_module'])
        self.observation_format = FormattedObservation(
            getattr(self.native, self.configuration['observation_template']))
        self.rng = Random(task['source']['seed'])
        self.native_state = deepcopy(self.configuration['native_state'])
        self.item = self._call('generate', task['source']['seed'])
        assert self.item['board'] == task['source']['initial_state']['board']
        self.official_context = self._call('print_board', self.item)
        assert self.official_context == task['source']['context']
        self.initial_observation = self.official_context
        self.observation = self.initial_observation
        self.done = bool(self.item['is_end'])
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
        context = self.official_context + self.prompts['episode'].format(**self.episode)
        self.context_text = render_context(context, self.display_answer_schema(),
                                           self.display_tool_interface(True), self.session['serialization'])

    def _call(self, name, *arguments):
        # Native modules use global RNG and environment stores; each branch owns their contents.
        with self.native_lock:
            self.native.random = self.rng
            for key, value in self.native_state.items():
                setattr(self.native, key, value)
            return getattr(self.native, name)(*arguments)

    def fork(self):
        child = copy(self)
        child.rng = deepcopy(self.rng)
        child.native_state = deepcopy(self.native_state)
        child.item = deepcopy(self.item)
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
        self.item['action'] = arguments['action']
        self.item = self._call('verify', self.item)
        self.executed_actions.append(arguments['action'])
        observation = self._call('print_board', self.item)
        self.observation = json.dumps(self.observation_format.fields(observation),
                                      **self.session['serialization'])
        self.done = bool(self.item['is_end']) or len(self.executed_actions) >= self.episode['max_steps']
        if self.done:
            self.answer = {'actions': self.executed_actions.copy()}

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation = self.session_prompts['submission_observation']
        self.done = True

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step,
                    self.session['submission_tool']: self._finish}
        handlers[name](arguments)
        result = {'observation': self.observation, 'done': self.done, 'score': self.item['score']}
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
        replay = type(self)(self.task, self.settings, self.directory,
                            deadline=self.deadline, evidence=self.evidence,
                            configuration=self.configuration_path)
        for action in answer['actions']:
            replay._step({'action': action})
        return replay.item['score']
