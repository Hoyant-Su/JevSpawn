import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

from tests.infra.native_chunk_graph.qualify import run


def qualify(settings):
    original = torch._foreach_copy_
    captured_shapes = set()
    output = Path(settings['copy_profile_output'])
    output.mkdir(parents=True, exist_ok=True)

    def observed(targets, sources, non_blocking):
        key = (str(targets[0].dtype), tuple(tuple(t.shape) for t in targets))
        if key in captured_shapes:
            return original(targets, sources, non_blocking=non_blocking)
        captured_shapes.add(key)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            result = original(targets, sources, non_blocking=non_blocking)
        name = settings['copy_profile_file'].format(rank=dist.get_rank(), index=len(captured_shapes))
        captured.export_chrome_trace(str(output / name))
        return result

    with patch.object(torch, '_foreach_copy_', observed):
        run(settings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    qualify(json.loads(parser.parse_args().config.read_text()))
