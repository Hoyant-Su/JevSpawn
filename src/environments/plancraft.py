from copy import copy, deepcopy
import importlib
import json
from operator import attrgetter
from pathlib import Path
import sys
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments

from baselines.common.config import SharedConfig
from data.task_context import render_context
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


class PlancraftEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration_path = configuration
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        sys.path.insert(self.configuration['module_path_index'], str(ROOT / self.configuration['source_root']))
        native = importlib.import_module('plancraft.simple')
        native_actions = importlib.import_module('plancraft.environment.actions')
        native_prompts = importlib.import_module('plancraft.environment.prompts')
        handlers = [getattr(native_actions, name)() for name in self.configuration['handlers']]
        self.native = native.PlancraftGymWrapper(
            native.PlancraftExample(**task['source']['example']), actions=handlers,
            max_steps=SharedConfig.load(ROOT / self.configuration['shared_config']).runtime.max_turns,
            **self.configuration['environment'])
        initial, _, _, _, _ = self.native.step()
        self.initial_observation = initial['text']
        system = native_prompts.get_system_prompt(handlers=handlers, **self.configuration['system_prompt'])
        self.official_context = self.configuration['context_separator'].join([
            system['content'], self.initial_observation])
        self.observation = self.initial_observation
        self.done = False
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

    def fork(self):
        child = copy(self)
        immutable = [attrgetter(path)(self.native) for path in self.configuration['shared_native_objects']]
        child.native = deepcopy(self.native, {id(value): value for value in immutable})
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
        result, reward, terminated, truncated, _ = self.native.step(arguments['action'])
        self.executed_actions.append(arguments['action'])
        self.observation = result['text']
        self.done = terminated or truncated
        if self.done:
            self.answer = {'actions': self.executed_actions.copy()}
        return {'observation': self.observation, 'reward': reward,
                'terminated': terminated, 'truncated': truncated}

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation = self.session_prompts['submission_observation']
        self.done = True
        return {'observation': self.observation}

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step,
                    self.session['submission_tool']: self._finish}
        result = handlers[name](arguments)
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
        replay = type(self)(self.task, self.settings, self.directory, deadline=self.deadline,
                            evidence=self.evidence, configuration=self.configuration_path)
        for action in answer['actions']:
            replay._step({'action': action})
            if replay.done:
                break
        return replay.native.success
