import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from jev_spawn.cli.evaluate import POLICY, TEMPLATES, sandbox_command


parser = argparse.ArgumentParser()
parser.add_argument('--settings', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True)
args = parser.parse_args()
settings = json.loads(args.settings.read_text())
args.work_dir.mkdir(parents=True, exist_ok=True)
program = Path(tempfile.mkdtemp(prefix=POLICY['preflight_prefix'], dir=args.work_dir)) / POLICY['program_name']
program.write_text(TEMPLATES['preflight'])
subprocess.run(sandbox_command(settings['code']['sandbox'], program), check=True,
               timeout=POLICY['preflight_timeout_seconds'])
print(json.dumps({'sandbox_preflight': 'passed', 'program': str(program)}))
