from copy import copy, deepcopy
import importlib
import json
from pathlib import Path
import sys
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments

from data.task_context import render_context
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


class AgentQuestEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        sys.path.insert(self.configuration['module_path_index'], str(ROOT / self.configuration['source_root']))
        module = importlib.import_module(self.configuration['module'])
        self.native = getattr(module, self.configuration['driver'])(**task['source']['native_parameters'])
        self.observation = self.native.reset().model_dump()
        self.initial_observation = self.observation['output']
        self.official_context = self.initial_observation
        self.done = False
        self.answer = None
        self.actions, self.tool_timings = [], []
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

    def fork(self):
        child = copy(self)
        child.native = deepcopy(self.native)
        # Native Driver.record closes over the original driver; bind the copied driver explicitly.
        child.native.step = child.native.record(type(child.native).step.__get__(child.native))
        child.actions = deepcopy(self.actions)
        child.tool_timings = deepcopy(self.tool_timings)
        child.observation = deepcopy(self.observation)
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

    def execute(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step,
                    self.session['submission_tool']: self._finish}
        handlers[name](arguments)
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments),
                             'result': deepcopy(self.observation)})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                 'finished_monotonic': time.perf_counter()})
        self.deadline()
        return self.observation['output'], self.done

    def _step(self, arguments):
        self.observation = self.native.step_raw(arguments['action']).model_dump()
        self.done = not self.observation['can_proceed']
        if self.done:
            self.answer = {'answer': deepcopy(self.native.current_state.value)}

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation = {'output': self.session_prompts['submission_observation'], 'can_proceed': False}
        self.done = True

    def observe(self, name, arguments):
        self.execute(name, arguments)
        return deepcopy(self.observation)

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        return answer['answer'] == self.native.goal
