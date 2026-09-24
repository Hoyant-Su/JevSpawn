import argparse
import json
from pathlib import Path
import time

import torch.distributed as dist

from baselines.common.config import SharedConfig
from baselines.common.parallel_run import execute_initialized, initialize_parallel
from baselines.common.run import run
from jev_spawn.infra.kernel_tuning import load_tuning


def execute(args, runner):
    specification = json.loads(Path(args.specification).read_text())
    parallel = json.loads(Path(args.parallel_settings).read_text())
    tuning = json.loads(Path(args.tuning).read_text())
    backend, commands, starts = initialize_parallel(SharedConfig.load(specification['shared_config']), parallel)
    started = time.perf_counter()
    loaded = load_tuning(Path(tuning['cache_directory']) / tuning['cache_file'].format(rank=dist.get_rank()),
                        backend, json.loads(Path(tuning['settings']).read_text()))
    metadata = {'rank': dist.get_rank(), 'load_seconds': time.perf_counter() - started, 'loaded': loaded,
                'configuration': tuning}
    all_metadata = [None for _ in starts]
    dist.all_gather_object(all_metadata, metadata, group=commands.control_group)
    for startup, record in zip(starts, all_metadata, strict=True):
        startup['kernel_tuning'] = record
    execute_initialized(specification, args.output, parallel, runner, backend, commands, starts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--specification', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parallel-settings', required=True)
    parser.add_argument('--tuning', required=True)
    args = parser.parse_args()
    execute(args, run)


if __name__ == '__main__':
    main()
