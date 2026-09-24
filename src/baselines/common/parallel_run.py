import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.service_factory import service_factory
from baselines.common.run import run
from baselines.common.persistence import save
from baselines.common.runtime import InferenceRuntime
from jev_spawn.infra.backend import Backend
from jev_spawn.infra.qwen35.commands import ParallelCommands
from project_paths import ROOT


def initialize_parallel(shared, settings):
    assert shared.runtime.world_size > 1
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group(backend=settings['tensor_backend'],
                            timeout=timedelta(seconds=settings['distributed_timeout_seconds']))
    assert dist.get_world_size() == shared.runtime.world_size
    control = dist.new_group(backend=settings['control_backend'],
                             timeout=timedelta(seconds=settings['distributed_timeout_seconds']))
    commands = ParallelCommands(control, dist.group.WORLD, settings)
    started = time.perf_counter()
    backend = Backend(shared.backend())
    backend.parallel_commands = commands
    torch.cuda.synchronize(backend.device)
    startup = {'rank': dist.get_rank(), 'device': str(backend.device),
               'load_and_shard_seconds': time.perf_counter() - started,
               'backend': backend.metadata}
    starts = [None] * shared.runtime.world_size
    dist.all_gather_object(starts, startup, group=control)
    return backend, commands, starts


def execute(specification, output, settings):
    return execute_with_runner(specification, output, settings, run)


def execute_with_runner(specification, output, settings, runner):
    shared = SharedConfig.load(specification['shared_config'])
    backend, commands, starts = initialize_parallel(shared, settings)
    return execute_initialized(specification, output, settings, runner, backend, commands, starts)


def execute_initialized(specification, output, settings, runner, backend, commands, starts):
    method = json.loads(Path(specification['method']).read_text())
    sys.path[:0] = [str(Path(path).resolve()) for path in method['python_paths']]
    pending = commands.leader_value(any(not (output / f'task-{index:05d}.json').exists()
        for index in range(specification['task_count'])) if commands.is_leader else None)
    if not pending:
        if commands.is_leader:
            runner(specification, output, backend=backend)
        dist.destroy_process_group(commands.control_group)
        dist.destroy_process_group()
        return
    if commands.is_leader:
        output.mkdir(parents=True, exist_ok=True)
        save(output / 'parallel_startup.json', {
            'ranks': starts, 'host_owner_rank': settings['leader_rank'],
            'scope': 'Only the leader performs task admission, methods, tools, result commits and evaluation. Followers execute dispatched numerical cohorts.'})
        try:
            runner(specification, output, backend=backend)
        finally:
            commands.finish()
    else:
        inference = json.loads((ROOT / 'configs/inference/shared_service.json').read_text())
        runtime = InferenceRuntime(specification['shared_config'],
            service_factory(method, inference), backend=backend)
        try:
            commands.serve()
        finally:
            runtime.close()
    dist.destroy_process_group(commands.control_group)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-settings', type=Path, required=True)
    args = parser.parse_args()
    execute(json.loads(args.specification.read_text()), args.output,
            json.loads(args.parallel_settings.read_text()))


if __name__ == '__main__':
    main()
