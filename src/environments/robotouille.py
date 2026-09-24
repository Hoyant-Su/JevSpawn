from copy import copy, deepcopy
from functools import cache
import importlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments

from data.task_context import render_context
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


_NATIVE_LOCK = Lock()


@cache
def native_api(source_root):
    source = ROOT / source_root
    with _NATIVE_LOCK:
        own_package, search_path = sys.modules['environments'], sys.path.copy()
        try:
            sys.path.insert(0, str(source))
            package_spec = importlib.util.spec_from_file_location('environments', source / 'environments/__init__.py')
            package = importlib.util.module_from_spec(package_spec)
            sys.modules['environments'] = package
            package_spec.loader.exec_module(package)
            builder = importlib.import_module('environments.env_generator.builder')
            specification = importlib.util.spec_from_file_location('robotouille_native_state_api', source / 'robotouille/env.py')
            module = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
        finally:
            sys.modules['environments'] = own_package
            sys.path[:] = search_path
    return builder, module


class RobotouilleEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.boundary_prompts = load_prompt(self.configuration['feedback_prompts'])
        self.session_prompts = load_prompt(self.session['prompts'])
        builder, self.api = native_api(self.configuration['source_root'])
        domain = json.loads((ROOT / self.configuration['domain']).read_text())
        with _NATIVE_LOCK:
            rng = random.getstate()
            try:
                random.seed(task['source']['seed'])
                scene = builder.load_environment(str(ROOT / task['source']['sample']))
                _, named_scene = builder.build_problem(scene)
                self.native = self.api.build_state(domain, named_scene)
            finally:
                random.setstate(rng)
        self.initial_native = deepcopy(self.native)
        self.official_context = self.api.LanguageSpace.state_to_language_description(self.native)
        self.initial_observation = self.observation = self.official_context
        self.done, self.answer = self.native.is_goal_reached(), None
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
        native_actions, descriptions = self.native.get_valid_actions_and_str()
        if arguments['action'] not in descriptions:
            feedback = self.boundary_prompts['invalid_action'].format(action=arguments['action'])
            self.observation = self.boundary_prompts['rejected_observation'].format(
                feedback=feedback,
                observation=self.api.LanguageSpace.state_to_language_description(self.native))
            return
        action = native_actions[descriptions.index(arguments['action'])]
        self.done = self.native.step([action])
        self.executed_actions.append(arguments['action'])
        self.observation = self.api.LanguageSpace.state_to_language_description(self.native)
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
            native_actions, descriptions = replay.get_valid_actions_and_str()
            if replay.is_goal_reached() or action not in descriptions:
                return False
            replay.step([native_actions[descriptions.index(action)]])
        return replay.is_goal_reached()
