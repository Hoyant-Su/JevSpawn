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
progress = []
for entry in lane['methods']:
    if Path(entry['evaluation']).exists():
        evaluation = json.loads(Path(entry['evaluation']).read_text())
        assert evaluation['tasks'] == manifest['task_count']
        progress.append({'method': entry['method'], 'status': 'existing_completed_evaluation'})
        continue
    occupancy = subprocess.check_output(lane['occupancy_command'], text=True)
    memory = [int(line.split(',')[manifest['memory_column']].strip()) for line in occupancy.splitlines()]
    assert all(value <= manifest['idle_memory_mib'] for value in memory), occupancy
    started = time.time()
    with Path(entry['log']).open('a') as log:
        log.write(occupancy)
        log.flush()
        for command in entry['commands']:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    progress.append({'method': entry['method'], 'status': 'completed',
                     'started_unix': started, 'finished_unix': time.time(), 'occupancy_before': occupancy})
    Path(lane['progress']).write_text(json.dumps(progress, indent=2) + '\n')
