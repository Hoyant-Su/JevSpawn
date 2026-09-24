import json
from pathlib import Path
import time

import jsonschema

from baselines.common.resources import TEMPLATES
from baselines.common.tasks import render
from baselines.tool_agents.tools import ActionError
from jev_spawn.infra.prompts import load_prompt


class SubmissionEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence, contract):
        self.task, self.settings, self.contract = task, settings, contract
        self.directory, self.deadline = Path(directory), deadline
        self.answer = None
        self.actions, self.tool_timings = [], []
        self.prompts = load_prompt(contract['prompts'])
        self.tools = list(self.prompts['tools'])
        self.tool_definitions = self.prompts['tools']
        self.input_schemas = {name: task['answer_schema'] for name in self.tools}

    def display_answer_schema(self):
        return self.task['answer_schema']

    def display_tool_schema(self, name):
        return self.input_schemas[name]

    def display_tool_interface(self, include_finish):
        return {name: {**definition, 'input_schema': self.input_schemas[name]}
                for name, definition in self.tool_definitions.items() if include_finish}

    def context(self, include_finish, serialization):
        return TEMPLATES['tools'].format(task=render(self.task),
            tools=json.dumps(self.display_tool_interface(include_finish), **serialization))

    def reset(self):
        return self.context(self.contract['include_finish'], self.contract['serialization'])

    def execute(self, name, arguments):
        started = time.perf_counter()
        self.deadline()
        if name not in self.input_schemas:
            raise ActionError(f'Unknown tool: {name}')
        jsonschema.Draft202012Validator(self.input_schemas[name]).validate(arguments)
        self.answer = arguments
        result = self.contract['submission_result']
        self.actions.append({'tool': name, 'arguments': arguments, 'result': result})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                  'finished_monotonic': time.perf_counter()})
        self.deadline()
        return json.dumps(result, **self.contract['serialization']), self.contract['terminal']
