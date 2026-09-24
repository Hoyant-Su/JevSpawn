import atexit
from copy import copy, deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time

import jsonschema

from environments.arguments import validate_arguments
from jinja2 import Template

from data.task_context import render_context
from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


def native_worker(configuration):
    settings = json.loads(Path(configuration).read_text())
    source = ROOT / settings['native_source']
    spec = importlib.util.spec_from_file_location('autoplanbench_official', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ['VAL'] = str(ROOT / settings['validator_directory'])
    request = json.loads(sys.stdin.readline())
    native = module.RawPDDLEnvironment(**request)
    template = Template((ROOT / settings['official_prompt']).read_text())
    context = template.render(task_description=native.get_description_goal_state(),
                              actions=Path(native.lowercase_domain_file).read_text(),
                              **settings['template_arguments'])
    context += settings['context_separator'] + native.get_description_initial_state()
    print(json.dumps({'context': context, 'facts': native.facts_current_state,
                      'observation': native.get_description_initial_state()}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        native.facts_current_state = request['facts']
        native.completed = request['completed']
        observation, valid, done = native.step(request['action'])
        print(json.dumps({'facts': native.facts_current_state, 'observation': observation,
                          'valid': valid, 'done': done}), flush=True)


class AutoPlanBenchEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None, configuration):
        self.task, self.settings = task, settings
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.deadline, self.evidence = deadline, evidence
        self.configuration = json.loads((ROOT / configuration).read_text())
        self.session = json.loads((ROOT / self.configuration['session']).read_text())
        self.session_prompts = load_prompt(self.session['prompts'])
        self.source_lock = Lock()
        self.observed_source_ids = set()
        self.worker = subprocess.Popen(
            [str(ROOT / self.configuration['python']), '-u', str(Path(__file__).resolve()),
             str(ROOT / configuration)], cwd=self.directory,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**os.environ, 'PYTHONPATH': os.pathsep.join(str(ROOT / path) for path in self.configuration['python_paths'])})
        # One native worker owns its VAL scratch files; branches supply isolated fact states.
        atexit.register(self.close)
        self.actions, self.tool_timings, self.executed_actions = [], [], []
        self.tools = [self.configuration['tool_name'], self.session['submission_tool']]
        self.input_schemas = {self.configuration['tool_name']: self.configuration['action_schema'],
                              self.session['submission_tool']: self.display_answer_schema()}
        self.tool_definitions = {self.configuration['tool_name']: self.configuration['tool_definition'],
            self.session['submission_tool']: {
                'description': self.session_prompts['submission_description'],
                'arguments': {name: schema['type'] for name, schema in
                              self.display_answer_schema()['properties'].items()}}}
        source = {key: str(ROOT / task['source'][key]) for key in self.configuration['source_fields']}
        # The native parser lowercases a domain copy. Keep that write inside the run.
        domain = self.directory / self.configuration['local_domain_file']
        domain.write_text(Path(source['domain_file']).read_text())
        source['domain_file'] = str(domain.resolve())
        initial = self.exchange(source)
        self.official_context = initial['context']
        self.initial_observation = self.observation = initial['observation']
        self.facts = initial['facts']
        self.initial_facts = self.facts.copy()
        self.context_text = render_context(self.official_context, self.display_answer_schema(),
            self.display_tool_interface(True), self.session['serialization'])
        self.done = False
        self.answer = None

    def exchange(self, request):
        with self.source_lock:
            self.worker.stdin.write(json.dumps(request) + '\n')
            self.worker.stdin.flush()
            return json.loads(self.worker.stdout.readline())

    def close(self):
        self.worker.stdin.close()
        self.worker.wait()

    def fork(self):
        child = copy(self)
        child.facts = self.facts.copy()
        child.actions = deepcopy(self.actions)
        child.tool_timings = deepcopy(self.tool_timings)
        child.executed_actions = self.executed_actions.copy()
        child.answer = deepcopy(self.answer)
        child.observed_source_ids = set(self.observed_source_ids)
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

    def _move(self, arguments):
        response = self.exchange({'facts': self.facts, 'completed': self.done,
                                  'action': arguments['action']})
        self.facts = response.pop('facts')
        self.observation, self.done = response['observation'], response['done']
        if response['valid']:
            self.executed_actions.append(arguments['action'])
        if self.done:
            self.answer = {'actions': self.executed_actions.copy()}
        return response

    def _finish(self, arguments):
        self.answer = deepcopy(arguments)
        self.observation = self.session_prompts['submission_observation']
        self.done = True
        return {'observation': self.observation, 'done': self.done}

    def observe(self, name, arguments):
        self.deadline()
        validate_arguments(arguments, self.input_schemas[name])
        started = time.perf_counter()
        handlers = {self.configuration['tool_name']: self._move,
                    self.session['submission_tool']: self._finish}
        response = handlers[name](arguments)
        self.actions.append({'tool': name, 'arguments': deepcopy(arguments), 'result': response})
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                 'finished_monotonic': time.perf_counter()})
        self.deadline()
        return response

    def evaluate(self, answer):
        jsonschema.validate(answer, self.display_answer_schema())
        branch = self.fork()
        branch.facts = self.initial_facts.copy()
        branch.done = False
        branch.executed_actions = []
        for action in answer['actions']:
            result = branch._move({'action': action})
            if not result['valid']:
                return False
        return branch.done

    def execute(self, name, arguments):
        result = self.observe(name, arguments)
        return json.dumps(result, **self.configuration['serialization']), self.done


if __name__ == '__main__':
    native_worker(sys.argv[1])
