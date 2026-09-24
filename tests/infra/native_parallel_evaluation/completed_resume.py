import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from baselines.common.persistence import save


def run(settings):
    directory = Path(settings['run'])
    specification = json.loads(Path(settings['specification']).read_text())
    shared = yaml.safe_load(Path(specification['shared_config']).read_text())
    protocol = directory / 'protocol.json'
    assert json.loads(protocol.read_text())['specification'] == specification
    files = [protocol, directory / 'completion.json', directory / 'parallel_startup.json',
             *(directory / f'task-{index:05d}.json' for index in range(specification['task_count']))]
    snapshots = {path: path.read_bytes() for path in files}
    sessions = sorted(directory.glob('session-*'))
    environment = {**os.environ, 'PYTHONHASHSEED': str(shared['runtime']['seed'])}
    started = time.perf_counter()
    result = subprocess.run(['bash', 'scripts/run.sh', '', '-m', 'torch.distributed.run',
        '--standalone', '--nproc_per_node', str(shared['runtime']['world_size']),
        '-m', 'baselines.common.parallel_run', '--specification', settings['specification'],
        '--output', settings['run'], '--parallel-settings', settings['parallel_settings']],
        env=environment, check=True)
    unchanged = all(path.read_bytes() == content for path, content in snapshots.items())
    same_sessions = sorted(directory.glob('session-*')) == sessions
    report = {'settings': settings, 'exit_code': result.returncode,
        'elapsed_seconds': time.perf_counter() - started,
        'saved_files_verified': len(snapshots), 'all_saved_files_byte_identical': unchanged,
        'no_new_inference_sessions': same_sessions, 'cuda_visible_devices': '',
        'world_size': shared['runtime']['world_size'], 'seed': shared['runtime']['seed']}
    save(settings['output'], report)
    assert unchanged and same_sessions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(json.loads(parser.parse_args().config.read_text()))
