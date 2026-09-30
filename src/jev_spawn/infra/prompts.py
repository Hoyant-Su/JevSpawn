from copy import deepcopy
from functools import cache
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / 'configs/prompt_sources.json').read_text())


@cache
def prompt_file(path):
    return json.loads((ROOT / path).read_text())


def load_prompt(source):
    path = Path(source)
    key = str(path.relative_to(ROOT) if path.is_absolute() else path)
    filename, name = SOURCES[key]
    return deepcopy(prompt_file(filename)[name])


def resolve_prompts(value):
    if isinstance(value, dict):
        if set(value) == {'$prompt'}:
            return load_prompt(value['$prompt'])
        return {key: resolve_prompts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_prompts(item) for item in value]
    return value
