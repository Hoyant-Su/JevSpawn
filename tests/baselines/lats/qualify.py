from baselines.lats.stages import write, timing, run_stage
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from threading import Barrier
import time
import traceback

import torch

from baselines.lats.adapter import ChatModel, SandboxExecutor, load_upstream, run_task
from baselines.official.model_service import GenerationService
from jev_spawn.infra.backend import Backend






def qualify_service(service, rows, output, budget):
    messages = [[{'role': 'user', 'content': row['prompt']}] for row in rows]

    def batch(stops):
        barrier = Barrier(len(rows))

        def complete(index):
            barrier.wait()
            return service.complete(messages[index], budget, 0, stop=stops,
                                    task_id=rows[index]['task_id'])[0]

        with ThreadPoolExecutor(max_workers=len(rows)) as pool:
            return list(pool.map(complete, range(len(rows))))

    result = {}
    for phase in ['warmup', 'measured']:
        start = len(service.records)
        raw = batch(None)
        stopped = batch(['return'])
        sampled = service.complete(messages[0], budget, .8, n=8, task_id=rows[0]['task_id'])
        records = service.records[start:]
        checks = {'n_returns_eight': len(sampled) == 8,
                  'all_batches_eight': all(row['batch_size'] == 8 for row in records),
                  'stop_matches_unstopped_prefix': [a.split('return')[0] == b for a, b in zip(raw, stopped)],
                  'sampled_unique_completions': len(set(sampled)),
                  'stopped_output_token_counts': records[1]['output_tokens']}
        result[phase] = {'checks': checks, 'timing': timing(records)}
        write(output / f'service-{phase}-batches.json', records)
        write(output / 'service-qualification.json', result)
        assert checks['n_returns_eight'] and checks['all_batches_eight']
        assert all(checks['stop_matches_unstopped_prefix'])
    print(json.dumps({'event': 'service_qualified', **result['measured']}), flush=True)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    config = json.loads(Path(settings['native_config']).read_text())
    args.output.mkdir(parents=True)
    config['run_dir'] = str(args.output)
    rows = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()][:settings['task_count']]
    test_ids = {json.loads(line)['task_id'] for line in Path(settings['formal_tasks']).read_text().splitlines()}
    assert len(rows) == settings['task_count'] == config['batch_size']
    assert not test_ids & {row['task_id'] for row in rows}
    write(args.output / 'protocol.json', {'settings': settings, 'native': config,
                                        'task_ids': [row['task_id'] for row in rows],
                                        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                                        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                                                       '--format=csv,noheader'], text=True)})
    core, result_type = load_upstream(settings['upstream'])
    backend = Backend(config)
    write(args.output / 'backend.json', backend.metadata)
    service = GenerationService(backend, config['batch_size'], settings['batch_wait_seconds'])
    try:
        source = Path(settings['service_qualification_source'])
        qualification = json.loads(source.read_text())
        assert qualification['measured']['timing']['intervals_over_100ms'] == 0
        write(args.output / 'service-qualification.json', {'source': str(source), 'result': qualification})
        for stage in ['warmup', 'measured']:
            run_stage(core, result_type, service, rows, settings, args.output, stage)
    finally:
        service.close()


if __name__ == '__main__':
    main()
