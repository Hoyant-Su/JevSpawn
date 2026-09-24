import json
import os
from pathlib import Path
import sys

import yaml


config_path = sys.argv[1]
config = json.loads(Path(config_path).read_text())
shared = yaml.safe_load(Path(config['shared_config']).read_text())
command = ['bash', 'scripts/run.sh', ','.join(map(str, config['devices'])),
           '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node=' + str(shared['runtime']['world_size']),
           config['entrypoint'], '--config', config_path]
os.execvpe(command[0], command,
          dict(os.environ, PYTHONHASHSEED=str(shared['runtime']['seed'])))
