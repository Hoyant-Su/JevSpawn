import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument('--settings', type=Path, required=True)
parser.add_argument('--parallel-settings', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
settings = json.loads(args.settings.read_text())
parallel = json.loads(args.parallel_settings.read_text())
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group(parallel['tensor_backend'], timeout=timedelta(seconds=parallel['distributed_timeout_seconds']))
rows = []
for shape in settings['shapes']:
    value = torch.full(shape, rank, dtype=getattr(torch, settings['dtype']), device='cuda')
    dist.all_reduce(value)
    torch.cuda.synchronize()
    expected = sum(range(dist.get_world_size()))
    assert torch.all(value == expected).item()
    rows.append({'shape': shape, 'sum': expected, 'exact': True})
if rank == parallel['leader_rank']:
    args.output.write_text(json.dumps({'world_size': dist.get_world_size(), 'operations': rows}, indent=2) + '\n')
dist.destroy_process_group()
