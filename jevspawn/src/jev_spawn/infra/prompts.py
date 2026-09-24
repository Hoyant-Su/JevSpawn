from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PROMPTS = json.loads((ROOT / 'template/prompts.json').read_text())


def load_prompt(source):
    path = Path(source)
    key = str(path.relative_to(ROOT) if path.is_absolute() else path)
    return deepcopy(PROMPTS['templates'][PROMPTS['references'][key]])


def resolve_prompts(value):
    if isinstance(value, dict):
        if set(value) == {'$prompt'}:
            return load_prompt(value['$prompt'])
        return {key: resolve_prompts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_prompts(item) for item in value]
    return value
