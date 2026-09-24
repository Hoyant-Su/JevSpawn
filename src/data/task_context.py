import json
from pathlib import Path

from project_paths import ROOT
from jev_spawn.infra.prompts import load_prompt


CONTEXT_FORMAT = json.loads((ROOT / 'configs/data/task_context.json').read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def serialize_context(task, fields, serialization):
    return json.dumps({key: task[key] for key in fields}, **serialization)


def render(task):
    return serialize_context(task, CONTEXT_FORMAT['context_fields'], CONTEXT_FORMAT['serialization'])


def render_context(official_context, answer_schema, tool_interface, serialization):
    return load_prompt('shared.context')['context'].format(
        official_context=official_context,
        answer_schema=json.dumps(answer_schema, **serialization),
        tools=json.dumps(tool_interface, **serialization))
