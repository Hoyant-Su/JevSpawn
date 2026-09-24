import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
