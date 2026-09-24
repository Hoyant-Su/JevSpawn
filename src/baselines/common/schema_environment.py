from copy import deepcopy
from functools import partial
import importlib
import json

import jsonschema

from baselines.common.environment import TaskEnvironment
from baselines.tool_agents.tools import ActionError
from methods.evidence_flow.environment import InvalidSearchQuery
from jev_spawn.infra.prompts import resolve_prompts


def environment_definition(specification):
    definition = resolve_prompts(specification['environment_execution'])
    module, name = definition['class'].rsplit('.', maxsplit=1)
    cls = getattr(importlib.import_module(module), name)
    return partial(cls, **definition['parameters']), definition


class SchemaObservationEnvironment(TaskEnvironment):
    def __init__(self, *args, read_id_schema, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_validation_schema = deepcopy(self.input_schemas['read'])
        self.input_schemas['read']['properties']['ids']['items'] = deepcopy(read_id_schema)

    def execute(self, name, arguments):
        if name == 'read':
            with self.source_lock:
                for error in jsonschema.Draft202012Validator(self.read_validation_schema).iter_errors(arguments):
                    raise ActionError(error.message)
        if name != 'search' or name not in self.tools:
            return super().execute(name, arguments)
        self.deadline()
        for error in jsonschema.Draft202012Validator(self.input_schemas[name]).iter_errors(arguments):
            raise ActionError(error.message)
        try:
            result = self.evidence.search(arguments['query'], arguments['k'])
        except InvalidSearchQuery as error:
            raise ActionError(str(error)) from error
        with self.source_lock:
            self.observed_source_ids.update(row['id'] for row in result)
            self.read_validation_schema['properties']['ids']['items']['enum'] = sorted(self.observed_source_ids)
        self.actions.append({'tool': name, 'arguments': arguments, 'result': result})
        self.deadline()
        return json.dumps(result, ensure_ascii=False), False
