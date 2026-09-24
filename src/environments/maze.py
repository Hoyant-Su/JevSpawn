from copy import copy, deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments

from data.task_context import render_context
from jev_spawn.infra.configuration import resolve_symbol
from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


def native_modules(configuration):
    source = ROOT / configuration['source_root']
    sys.path.insert(configuration['module_path_index'], str(ROOT / configuration['dependency_root']))
    sys.path.insert(configuration['module_path_index'], str(source))
    specification = importlib.util.spec_from_file_location(
        configuration['factory_module_name'], source / configuration['factory_file'])
    factory = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(factory)
    return factory, importlib.import_module(configuration['text_module'])


class MazeEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.prompts = load_prompt(self.configuration['prompts'])
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.session_prompts = load_prompt(self.session['prompts'])
        factory, text = native_modules(self.configuration)
        self.text_type = text.Text
        self.native, _, _ = factory.setup_maze_env(
            start_position=task['source']['init_position'], **self.configuration['environment'])
        self.native.last_k = self.configuration['native_history_limit']
        self.history = self.native.reset(seed=self.configuration['seed'], options={
            'goal': task['source']['goal'], 'init_position': task['source']['init_position']})
        self.initial_native, self.initial_history = deepcopy(self.native), deepcopy(self.history)
        self.initial_observation = self.history[self.configuration['latest_history_index']].text
        self.observation = self.initial_observation
        self.answer, self.done = None, False
        self.actions, self.tool_timings, self.executed_actions = [], [], []
        self.source_lock, self.observed_source_ids = Lock(), set()
        self.native_actions = {action.removesuffix(self.configuration['action_suffix']): action
                               for action in self.native.actions}
        self.tools = [self.configuration['tool_name'], self.session['submission_tool']]
        self.input_schemas = {self.configuration['tool_name']: self.configuration['action_schema'],
                             self.session['submission_tool']: self.display_answer_schema()}
        descriptions = {self.configuration['tool_name']: self.prompts['tool_description'],
                        self.session['submission_tool']: self.session_prompts['submission_description']}
        self.tool_definitions = {name: {'description': descriptions[name],
            'arguments': {key: value['type'] for key, value in schema['properties'].items()}}
            for name, schema in self.input_schemas.items()}
        official = self.configuration['initial_separator'].join((task['instruction'], self.initial_observation))
        self.context_text = render_context(official, self.display_answer_schema(), self.display_tool_interface(True),
                                           self.session['serialization'])

    def fork(self):
        child = copy(self)
        child.native, child.history = deepcopy(self.native), deepcopy(self.history)
        child.actions, child.tool_timings = deepcopy(self.actions), deepcopy(self.tool_timings)
        child.executed_actions, child.answer = self.executed_actions.copy(), deepcopy(self.answer)
        child.observed_source_ids, child.source_lock = set(self.observed_source_ids), Lock()
        return child

    def display_answer_schema(self):
        return self.task['answer_schema']

    def display_tool_schema(self, name):
        return self.input_schemas[name]

    def display_tool_output_schema(self, name):
        return self.configuration['tool_output_schemas'][name]

    def display_tool_interface(self, include_finish):
        return {name: {**definition, 'input_schema': self.input_schemas[name]}
                for name, definition in self.tool_definitions.items()}

    def context(self, include_finish, serialization):
        return self.context_text

    def reset(self):
        return self.context_text

    def _step(self, arguments):
        action = arguments[self.configuration['action_field']]
        self.history, reward, self.done = self.native.step(
            self.history + (self.text_type(self.native_actions[action], True),))
        self.executed_actions.append(action)
        self.observation = self.history[self.configuration['latest_history_index']].text
        if self.done:
            self.answer = {self.configuration['answer_field']: self.executed_actions.copy()}
        return {'observation': self.observation, 'reward': reward, 'done': self.done}

    def _finish(self, arguments):
        self.answer, self.done = deepcopy(arguments), True
        self.observation = self.session_prompts['submission_observation']
        return {'observation': self.observation, 'reward': self.configuration['submission_reward'], 'done': self.done}

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step, self.session['submission_tool']: self._finish}
        result = handlers[name](arguments)
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments), 'result': result})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                  'finished_monotonic': time.perf_counter()})
        self.deadline()
        return result

    def execute(self, name, arguments):
        result = self.observe(name, arguments)
        return json.dumps(result, **self.configuration['serialization']), result['done']

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        return resolve_symbol(self.configuration['scorer_class']).evaluate(
            self.initial_native, self.initial_history, self.text_type,
            answer[self.configuration['answer_field']], self.configuration)
