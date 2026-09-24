from baselines.official.phases import write, run_phase
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import json
import os
from pathlib import Path
import statistics
from threading import Barrier
import time

from baselines.official.model_service import GenerationService
from jev_spawn.infra.backend import Backend






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    native = json.loads(Path(settings['native_config']).read_text())
    prompts = json.loads(Path(settings['prompts']).read_text())
    args.output.mkdir(parents=True)
    rows = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()][:settings['task_count']]
    formal_ids = {json.loads(line)['task_id'] for line in Path(settings['formal_tasks']).read_text().splitlines()}
    assert not formal_ids & {row['task_id'] for row in rows}
    assert len(rows) == settings['batch_size'] == native['batch_size']
    write(args.output / 'protocol.json', {'settings': settings, 'native': native,
                                        'task_ids': [r['task_id'] for r in rows],
                                        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
    backend = Backend(native)
    write(args.output / 'backend.json', backend.metadata)
    service = GenerationService(backend, settings['batch_size'], settings['batch_wait_seconds'])
    try:
        for phase in ['warmup', 'measured']:
            run_phase(service, rows, settings, prompts, args.output, phase)
    finally:
        service.close()
    labels = {r['task_id']: r['labels']['q0'] for r in
              map(json.loads, Path(settings['labels']).read_text().splitlines())}
    results = [json.loads((args.output / 'measured' / (r['task_id'].replace('/', '_') + '.json')).read_text())
               for r in rows]
    write(args.output / 'development_quality.json', {
        'correct': sum(r.get('answer') == labels[r['task_id']] for r in results),
        'total': len(results),
        'valid': sum(r.get('answer') in [o['id'] for o in task['fields']['q0']['options']]
                     for r, task in zip(results, rows))})


if __name__ == '__main__':
    main()
