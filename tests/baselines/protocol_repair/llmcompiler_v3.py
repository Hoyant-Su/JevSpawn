import json
from pathlib import Path

import llmcompiler_v2 as previous


previous.TEMPLATES = json.loads((Path(__file__).parent / "template/prompts_v3.json").read_text())["llmcompiler"]
solve = previous.solve
