import argparse
import json
from pathlib import Path
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--lane', required=True)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
lane = manifest['lanes'][args.lane]
for entry in lane['methods']:
    completed = all(Path(path).exists() for path in entry['evaluations'])
    if completed:
        continue
    occupancy = subprocess.check_output(lane['occupancy_command'], text=True)
    memory = [int(line.strip()) for line in occupancy.splitlines()]
    assert all(value <= manifest['idle_memory_mib'] for value in memory), occupancy
    record = {'method': entry['method'], 'started_unix': time.time(),
              'occupancy_before': occupancy, 'status': 'running'}
    Path(lane['progress']).write_text(json.dumps(record, indent=2) + '\n')
    subprocess.run(entry['command'], check=True)
    assert all(Path(path).exists() for path in entry['evaluations'])
    record.update(status='completed', finished_unix=time.time())
    Path(lane['progress']).write_text(json.dumps(record, indent=2) + '\n')
