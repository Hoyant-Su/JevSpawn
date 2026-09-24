import json
from pathlib import Path

from dyflow_v2 import solve as previous_solve


TEMPLATES = json.loads((Path(__file__).parent / "template/prompts_v3.json").read_text())["dyflow_format"]


def solve(task, environment, complete, settings, prompts):
    return previous_solve(task, environment, complete, settings, {**prompts, **TEMPLATES})
