import importlib.util
import json
from pathlib import Path
from threading import Lock
import time
from types import SimpleNamespace

import jsonschema

from baselines.common.tasks import read, render
from baselines.common.errors import InvalidOutputError
from baselines.tool_agents.tools import ActionError, calculate
from methods.evidence_interfaces.interfaces import InvalidResponse
from methods.evidence_flow.environment import InvalidSearchQuery
from methods.hallucination_localization.inputs import locate
from baselines.common.resources import TEMPLATES

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt
from jev_spawn.infra.configuration import load_resource


class TaskEnvironment:
    def __init__(self, task, settings, directory, *, deadline, evidence=None):
        self.task, self.settings = task, settings
        self.directory = Path(directory)
        self.deadline = deadline
        self.evidence = evidence.episode(task['task_id']) if task['kind'] == 'answer' else None
        self.answer = None
        self.actions = []
        self.tool_timings = []
        self.observed_source_ids = set()
        self.source_lock = Lock()
        self.tools = ['calculate', 'finish']
        self.tools += {'fields': [], 'grid': [], 'spans': [], 'completion': [], 'function_calls': [], 'code': ['run_tests'],
                       'ranking': ['read_documents'], 'answer': ['search', 'read']}[task['kind']]
        definitions = load_prompt('configs/baselines/common/schema/tools.json')
        self.tool_definitions = {name: definitions[name] for name in self.tools}
        self.input_schemas = read(ROOT / 'configs/baselines/common/schema/tool_inputs.json')
        self.output_schemas = load_resource('tool_outputs')
        self.input_schemas['calculate']['properties']['expression']['maxLength'] = settings['calculator']['max_characters']
        self.input_schemas['read_documents']['properties']['ids']['maxItems'] = settings['documents_per_read']
        self.input_schemas['search']['properties']['k']['maximum'] = settings['evidence']['search_limit']
        self.input_schemas['read']['properties']['ids']['maxItems'] = settings['evidence']['read_limit']
        self.input_schemas['finish'] = task['answer_schema']
        if task['kind'] == 'answer':
            self.input_schemas['read']['properties']['ids']['items']['enum'] = []
        if task['kind'] == 'ranking':
            self.input_schemas['read_documents']['properties']['ids']['items']['enum'] = [
                document['document_id'] for document in task['source']['candidates']]

    def display_answer_schema(self):
        return self.task['answer_schema']

    def display_tool_schema(self, name):
        return self.input_schemas[name]

    def display_tool_output_schema(self, name):
        return self.output_schemas[name]

    def display_tool_interface(self, include_finish):
        return {name: {**definition, 'input_schema': self.input_schemas[name]}
                for name, definition in self.tool_definitions.items() if include_finish or name != 'finish'}

    def context(self, include_finish, serialization):
        return TEMPLATES['tools'].format(task=render(self.task),
            tools=json.dumps(self.display_tool_interface(include_finish), **serialization))

    def reset(self):
        return self.context(True, {})

    def observe(self, name, arguments):
        try:
            observation, done = self.execute(name, arguments)
        except ActionError as error:
            result = {'error': {'type': type(error).__name__, 'message': str(error)}}
            self.actions.append({'tool': name, 'arguments': arguments, 'result': result})
            self.deadline()
            return result
        assert not done, 'Only the submission action may finish a task.'
        return json.loads(observation)

    def execute(self, name, arguments):
        started = time.perf_counter()
        result = self._execute(name, arguments)
        self.tool_timings.append({'tool': name, 'started_monotonic': started,
                                  'finished_monotonic': time.perf_counter()})
        return result

    def _execute(self, name, arguments):
        self.deadline()
        if name not in self.tools:
            raise ActionError(TEMPLATES['unavailable_tool'].format(name=name))
        if name != 'finish':
            with self.source_lock:
                for error in jsonschema.Draft202012Validator(self.input_schemas[name]).iter_errors(arguments):
                    raise ActionError(error.message)
        if name == 'finish':
            try:
                jsonschema.validate(arguments, self.task['answer_schema'])
                if self.task['kind'] == 'spans':
                    locate(self.task['input']['response'], arguments['spans'])
            except (jsonschema.ValidationError, InvalidResponse) as error:
                raise InvalidOutputError(str(error)) from error
            self.answer = arguments
            result = {'submitted': True}
        elif name == 'calculate':
            result = calculate(arguments['expression'], self.settings['calculator'])
        elif name == 'read_documents':
            ids = arguments['ids']
            documents = {row['document_id']: row for row in self.task['source']['candidates']}
            result = [documents[identity] for identity in ids]
        elif name == 'search':
            try:
                result = self.evidence.search(arguments['query'], arguments['k'])
            except InvalidSearchQuery as error:
                raise ActionError(str(error)) from error
            with self.source_lock:
                self.observed_source_ids.update(row['id'] for row in result)
                self.input_schemas['read']['properties']['ids']['items']['enum'] = sorted(self.observed_source_ids)
        elif name == 'read':
            result = self.evidence.read(arguments['ids'])
        elif name == 'run_tests':
            spec = importlib.util.spec_from_file_location('common_code_evaluation', self.settings['evaluation_script'])
            evaluator = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator)
            self.directory.mkdir(parents=True, exist_ok=True)
            args = SimpleNamespace(sandbox=Path(self.settings['sandbox']), work_dir=self.directory,
                                   timeout=self.settings['code_timeout_seconds'], memory_mb=self.settings['code_memory_mb'])
            result = evaluator.evaluate_one({'task_id': self.task['task_id'], 'solution': arguments['code']},
                                            {'dataset': self.task['dataset'], 'test_code': arguments['assertions']}, args)
        self.actions.append({'tool': name, 'arguments': arguments, 'result': result})
        self.deadline()
        return json.dumps(result, ensure_ascii=False), name == 'finish'
