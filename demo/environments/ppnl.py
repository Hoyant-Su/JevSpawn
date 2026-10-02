import ast
from copy import copy, deepcopy
import json
from pathlib import Path
from threading import Lock
import time
from types import FunctionType

import jsonschema

from demo.environments.arguments import validate_arguments

from demo.environments.common import render_context
from demo.environments.common import load_prompt
from demo.environments.common import ROOT


def native_executor(source, name):
    source = ROOT / source
    tree = ast.parse(source.read_text(), filename=str(source))
    definition, = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == name]
    namespace = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


class PPNLEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings, self.directory = task, settings, Path(directory)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.session_prompts = load_prompt(self.session['prompts'])
        self.prompts = load_prompt(self.configuration['prompts'])
        self.native = native_executor(self.configuration['native_source'], self.configuration['native_function'])
        self.scorer = native_executor(self.configuration['evaluation_source'], self.configuration['evaluation_function'])
        self.world = deepcopy(task['source']['world'])
        self.position, = [(row, column) for row, cells in enumerate(self.world)
                          for column, value in enumerate(cells)
                          if value == self.configuration['start_marker']]
        self.initial_position = self.position
        self.official_context = self.prompts['context'].format(
            rules=self.prompts['rules'],
            instruction=task['source']['nl_description'])
        self.initial_observation = task['source']['nl_description']
        self.observation = self.initial_observation
        self.done = False
        self.answer = None
        self.actions, self.tool_timings, self.executed_actions = [], [], []
        self.source_lock = Lock()
        self.observed_source_ids = set()
        self.tools = [self.configuration['tool_name'], self.session['submission_tool']]
        self.input_schemas = {self.configuration['tool_name']: self.configuration['action_schema'],
                              self.session['submission_tool']: self.display_answer_schema()}
        self.tool_definitions = {self.configuration['tool_name']: {
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
        return self.context(False, self.configuration['serialization'])

    def execute(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._move,
                    self.session['submission_tool']: self._finish}
        handlers[name](arguments)
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments),
                             'result': {'observation': self.observation, 'done': self.done}})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                 'finished_monotonic': time.perf_counter()})
        self.deadline()
        return self.observation, self.done

    def _move(self, arguments):
        attempted = []
        # The upstream executor prints every attempted move, including the blocked move.
        execute = FunctionType(self.native.__code__, {'print': attempted.append},
                               self.native.__name__, self.native.__defaults__)
        status, self.observation, self.position = execute(arguments['actions'], self.position, self.world)
        self.executed_actions.extend(attempted[:self.configuration['executed_slice_stop'][str(status)]])
        self.done = status == self.configuration['success_status']
        if self.done:
            self.answer = {'actions': self.configuration['action_separator'].join(self.executed_actions)}

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation = self.session_prompts['submission_observation']
        self.done = True

    def observe(self, name, arguments):
        observation, done = self.execute(name, arguments)
        return {'observation': observation, 'done': done}

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        status = self.scorer(self.world, self.initial_position, answer['actions'],
                             len(self.world), len(self.world[self.initial_position[0]]))
        return status == self.configuration['success_status']
