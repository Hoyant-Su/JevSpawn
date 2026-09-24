import argparse
import json
from pathlib import Path
import socket


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--matrix', required=True)
parser.add_argument('--method', required=True)
args = parser.parse_args()
settings = json.loads(args.config.read_text())
if args.matrix in settings['matrices']:
    owner = settings['matrices'][args.matrix][args.method]
    current = socket.gethostname().split(settings['hostname_separator'])[0]
    if owner != current:
        print(json.dumps({'dispatch': 'skipped_nonowner', 'method': args.method,
                          'assigned_instance': owner, 'current_instance': current}), flush=True)
        raise SystemExit(settings['skip_exit_code'])
