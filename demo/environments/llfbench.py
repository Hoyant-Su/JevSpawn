from copy import copy, deepcopy
import json
from pathlib import Path
import sys
from threading import Lock
import time

import jsonschema

from demo.environments.arguments import validate_arguments

from demo.environments.common import render_context
from demo.environments.common import resolve_symbol
from demo.environments.common import load_prompt
from demo.environments.common import ROOT


RESET_LOCK = Lock()


class LLFEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.prompts = load_prompt(self.session['prompts'])
        sys.path.insert(self.configuration['module_path_index'], str(ROOT / self.configuration['source_root']))
        # Native reset seeds process-global RNGs; serialize only construction.
        with RESET_LOCK:
            native = resolve_symbol(self.configuration['native_class'])(**task['source']['native_parameters'])
            self.native = resolve_symbol(self.configuration['native_wrapper'])(native, **self.configuration['wrapper_parameters'])
            self.native.set_paraphrase_method(self.configuration['paraphrase_method'])
            initial, self.info = self.native.reset(seed=task['source']['seed'], options=task['source']['reset_options'])
        self.initial_native = deepcopy(self.native)
        self.observation = initial
        self.done, self.answer = False, None
        self.actions, self.executed_actions, self.tool_timings = [], [], []
        self.source_lock, self.observed_source_ids = Lock(), set()
        self.tool_definitions = self.configuration['public_contract']['tools']
        self.tools = list(self.tool_definitions)
        self.input_schemas = {name: tool['input_schema'] for name, tool in self.tool_definitions.items()}
        official = self.configuration['initial_separator'].join(initial[key] for key in self.configuration['initial_text_fields'])
        self.context_text = render_context(official, self.display_answer_schema(), self.tool_definitions, self.session['serialization'])

    def context(self, include_finish, serialization):
        return self.context_text

    def reset(self):
        return self.context_text

    def display_answer_schema(self):
        return self.task['answer_schema']

    def display_tool_interface(self, include_finish):
        return self.tool_definitions

    def display_tool_schema(self, name):
        return self.input_schemas[name]

    def display_tool_output_schema(self, name):
        return self.configuration['observation_schema']

    def fork(self):
        child = copy(self)
        child.native, child.info = deepcopy(self.native), deepcopy(self.info)
        child.observation, child.answer = deepcopy(self.observation), deepcopy(self.answer)
        child.actions, child.executed_actions = deepcopy(self.actions), self.executed_actions.copy()
        child.tool_timings = deepcopy(self.tool_timings)
        child.source_lock, child.observed_source_ids = Lock(), set(self.observed_source_ids)
        return child

    def _step(self, arguments):
        self.observation, _, terminated, truncated, self.info = self.native.step(arguments[self.configuration['action_field']])
        self.executed_actions.append(arguments[self.configuration['action_field']])
        self.done = terminated or truncated
        if self.done:
            self.answer = {self.configuration['answer_field']: self.executed_actions.copy()}
        return json.dumps(self.observation, **self.session['serialization'])

    def _finish(self, arguments):
        self.answer, self.done = deepcopy(arguments), True
        return self.prompts['submission_observation']

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._step, self.session['submission_tool']: self._finish}
        observation = handlers[name](arguments)
        result = {'observation': observation, 'done': self.done}
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments), 'result': result})
        self.tool_timings.append({'tool': name, 'started_monotonic': started, 'finished_monotonic': time.perf_counter()})
        self.deadline()
        return result

    def execute(self, name, arguments):
        result = self.observe(name, arguments)
        return result['observation'], result['done']

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        replay = deepcopy(self.initial_native)
        done = False
        for action in answer[self.configuration['answer_field']]:
            if done:
                return False
            _, _, terminated, truncated, _ = replay.step(action)
            done = terminated or truncated
        return bool(replay.unwrapped.goal_prev_visited)
