import argparse
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--samples', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    shared = yaml.safe_load((ROOT / settings['shared_config']).read_text())
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT), os.environ.get('PYTHONPATH', '')])
    os.execv(sys.executable, [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node', str(shared['runtime']['world_size']), '-m', 'demo.infer',
        '--config', str(args.config.resolve()), '--output', str(args.output.resolve()), '--samples', *args.samples])


if __name__ == '__main__':
    main()
