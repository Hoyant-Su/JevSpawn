from datetime import timedelta
import os
import time

import torch
import torch.distributed as dist

from jev_spawn.infra.backend import Backend
from jev_spawn.infra.qwen35.commands import ParallelCommands


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
