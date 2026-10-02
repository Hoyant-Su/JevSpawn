import importlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROMPTS = json.loads((ROOT / "prompts.json").read_text())


def load_prompt(name):
    return PROMPTS[name]


def resolve_symbol(name):
    module, symbol = name.rsplit(".", 1)
    return getattr(importlib.import_module(module), symbol)


def render_context(official_context, answer_schema, tool_interface, serialization):
    return load_prompt("shared.context")["context"].format(
        official_context=official_context,
        answer_schema=json.dumps(answer_schema, **serialization),
        tools=json.dumps(tool_interface, **serialization))
