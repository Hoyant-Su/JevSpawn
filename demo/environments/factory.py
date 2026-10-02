import json
from pathlib import Path
import time

from demo.environments.common import ROOT, resolve_symbol


REGISTRY = json.loads((ROOT / "registry.json").read_text())


def create(sample, output_dir, timeout_seconds):
    configuration = REGISTRY[sample["dataset"]]
    expires = time.monotonic() + timeout_seconds

    def deadline():
        if time.monotonic() >= expires:
            raise TimeoutError("The environment execution deadline expired.")

    environment = resolve_symbol(configuration["class"])(
        sample["task"], {}, Path(output_dir), deadline=deadline,
        configuration=configuration["configuration"])
    assert environment.context_text == sample["context"], sample["task_id"]
    return environment


def evaluate(sample, answer):
    environment = create(sample, ROOT, float("inf"))
    return float(environment.evaluate(answer))
