import argparse
import json
from pathlib import Path
import subprocess
import time

from baselines.common.persistence import pending_tasks, save
from data.task_context import rows


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
for entry in config['entries']:
    progress = Path(entry['progress'])
    if progress.exists() and json.loads(progress.read_text())['status'] == 'completed':
        specification = json.loads(Path(entry['specification']).read_text())
        assert not pending_tasks(rows(specification['tasks']), Path(entry['run']))
        json.loads(Path(entry['evaluation']).read_text())
        continue
    occupancy = subprocess.check_output(config['occupancy_command'], text=True)
    memory = [int(row.strip()) for row in occupancy.splitlines()]
    assert len(memory) == config['world_size']
    assert all(value <= config['idle_memory_mib'] for value in memory), occupancy
    record = {'method': entry['method'], 'dataset': entry['dataset'],
              'status': 'running', 'started_unix': time.time(), 'commands': [],
              'occupancy_before': memory, 'run': entry['run'], 'evaluation': entry['evaluation']}
    progress.parent.mkdir(parents=True, exist_ok=True)
    save(progress, record)
    with Path(entry['log']).open('a') as log:
        for command in entry['commands']:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            record['commands'].append({'command': command, 'exit_code': result.returncode})
            if result.returncode:
                break
    record.update(status='failed' if result.returncode else 'completed', finished_unix=time.time())
    save(progress, record)
    print(json.dumps(record), flush=True)
    if result.returncode and config['stop_on_failure']:
        raise SystemExit(result.returncode)
