import argparse
import json
import os
from pathlib import Path

import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--specification', type=Path, required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--devices', required=True)
parser.add_argument('--parallel-settings', required=True)
args = parser.parse_args()
specification = json.loads(args.specification.read_text())
shared = yaml.safe_load(Path(specification['shared_config']).read_text())
devices = args.devices.split(',')
assert len(devices) == len(set(devices)) == shared['runtime']['world_size']
environment = dict(os.environ, PYTHONHASHSEED=str(shared['runtime']['seed']))
command = ['bash', 'scripts/run.sh', args.devices, '-m', 'torch.distributed.run',
           '--standalone', '--nproc_per_node=' + str(shared['runtime']['world_size']),
           '-m', 'baselines.common.parallel_run', '--specification', str(args.specification),
           '--output', args.output, '--parallel-settings', args.parallel_settings]
os.execvpe(command[0], command, environment)
