from functools import partial
import json
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch.distributed as dist

from jev_spawn.infra.prompts import ROOT, load_prompt
from jev_spawn.rollout.run import solve
from jev_spawn.service.config import SharedConfig
from jev_spawn.service.parallel import initialize_parallel
from jev_spawn.service.runtime import InferenceRuntime
from jev_spawn.service.values import ValueService


class Environment(Protocol):
    """An executable, forkable environment. Task rules belong in the context."""

    done: bool
    answer: object
    tool_timings: list

    def fork(self) -> 'Environment':
        """Return an independent snapshot, including the current tool timings."""
        ...

    def observe(self, tool: str, arguments: dict) -> object:
        """Execute an action and update the observation, timing, and terminal state."""
        ...

    def display_tool_interface(self, include_finish: bool) -> object:
        ...

    def display_answer_schema(self) -> dict:
        ...


def run(tasks: Sequence[dict], environment_factory: Callable[[dict], Environment],
        commit: Callable[[int, dict], None], *, shared_config: Path, method_config: Path,
        inference_config: Path, parallel_config: Path, on_turn=None):
    """Run under torchrun; each task supplies task_id and context, without a dataset key.

    All ranks call run. Only the leader creates environments and commits results.
    Configuration paths are explicit; packaged configs are available under ROOT.
    The caller owns tool execution and persistence. No evaluator is loaded here.
    """
    shared = SharedConfig.load(shared_config)
    parallel = json.loads(parallel_config.read_text())
    method = json.loads(method_config.read_text())
    inference = json.loads(inference_config.read_text())
    backend, commands, _ = initialize_parallel(shared, parallel)
    runtime = InferenceRuntime(shared_config, partial(ValueService,
        settings=inference['settings'], prompts=load_prompt(inference['prompts'])), backend=backend)
    settings = shared.method_settings(method)
    runtime.service.configure_runtime_contract(settings)

    def execute(task):
        return solve(task, environment_factory(task), runtime.complete(task['task_id']),
                     settings, load_prompt(inference['prompts']),
                     on_turn=partial(on_turn, task['task_id']) if on_turn is not None else None)

    try:
        if commands.is_leader:
            results, elapsed = runtime.run(tasks, execute, commit)
            return {'results': results, 'elapsed_seconds': elapsed,
                    'inference': runtime.service.records, 'metadata': runtime.metadata()}
        commands.serve()
    finally:
        try:
            runtime.close()
        finally:
            if commands.is_leader:
                commands.finish()
            dist.destroy_process_group(commands.control_group)
            dist.destroy_process_group()
