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


class SmartPlayEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        sys.path.insert(self.configuration['module_path_index'], str(ROOT / self.configuration['source_root']))
        module = importlib.import_module(self.configuration['module'])
        self.native = getattr(module, self.configuration['class'])(**task['source']['native_parameters'])
        self.native.action_space.seed(task['source']['seed'])
        _, self.info = self.native.reset()
        self.initial_native = deepcopy(self.native)
        self.initial_observation = self.info['obs']
        self.official_context = self.configuration['context_separator'].join([
            self.info['manual'], self.initial_observation])
        self.observation = self.initial_observation
        self.done, self.answer = self.native.done, None
        self.actions, self.tool_timings, self.executed_actions = [], [], []
        self.source_lock = Lock()
        self.observed_source_ids = set()
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
        child.native, child.info = deepcopy(self.native), deepcopy(self.info)
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
        _, _, self.done, self.info = self.native.step(arguments['action'] - self.configuration['action_offset'])
        self.executed_actions.append(arguments['action'])
        self.observation = self.info['obs']
        if self.done:
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
        result = {'observation': self.observation, 'done': self.done,
                  'score': self.info['score']}
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
            if replay.done:
                return False
            replay.step(action - self.configuration['action_offset'])
        return replay.done
