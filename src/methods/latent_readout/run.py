import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from baselines.latentmas.adapter import SOURCE
from jev_spawn.infra.backend import Backend
from methods.latent_readout.backend import ReadoutExperiment
from methods.latent_readout.inputs import ROOT, condition_id, load_tasks, preflight, schedule, validate_settings
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def save(path, value):
    assert not path.exists(), f'Completed artifact already exists: {path}'
    temporary = path.with_name(path.name + f'.partial-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = resolve_prompts(json.loads(args.config.read_text()))
    validate_settings(settings)
    prompts = load_prompt(settings['prompts'])
    native = resolve_prompts(json.loads((ROOT / settings['native_config']).read_text()))
    for key in ['model_path', 'dtype', 'batch_size', 'max_input_tokens', 'seed']:
        assert native[key] == settings[key]
    tasks = load_tasks(settings)
    source_revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    protocol = {'settings': settings, 'native': native, 'prompts': prompts, 'tasks': tasks,
                'upstream_source': str(SOURCE), 'upstream_revision': source_revision,
                'source_methods': ['ModelWrapper.generate_latent_batch', 'ModelWrapper._build_latent_realign_matrix',
                                   'ModelWrapper._apply_latent_realignment'],
                'scope': 'LatentMAS recurrence with a controlled finite readout, not a new method or a spawning experiment.'}
    if args.resume:
        assert json.loads((args.output / 'protocol.json').read_text()) == protocol
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / 'protocol.json', protocol)
    with (args.output / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output / 'completion.json').exists():
            print((args.output / 'completion.json').read_text())
            return
        execute(args.output, settings, native, prompts, tasks)


def execute(output, settings, native, prompts, tasks):
    attempt = output / f'attempt-{len(list(output.glob("attempt-*"))):03d}'
    attempt.mkdir()
    warm = attempt / 'warmup'
    warm.mkdir()
    started = time.perf_counter()
    backend = Backend(native)
    experiment = ReadoutExperiment(backend, settings, prompts)
    torch.cuda.synchronize()
    loading = time.perf_counter() - started
    preflight_start = time.perf_counter()
    preflight_result = preflight(backend.tokenizer, tasks, settings, prompts)
    save(attempt / 'preflight.json', preflight_result)
    preflight_seconds = time.perf_counter() - preflight_start
    alignment_start = time.perf_counter()
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        experiment.wrapper._ensure_latent_realign_matrix(experiment.wrapper.model, backend.device, experiment.wrapper.args)
    torch.cuda.synchronize()
    setup = {'model_load_seconds': loading, 'preflight_seconds': preflight_seconds,
             'alignment_seconds': time.perf_counter() - alignment_start,
             'alignment_resident_bytes': torch.cuda.memory_allocated() - before,
             'model': backend.metadata, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
             'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)}
    setup['startup_seconds'] = time.perf_counter() - started
    save(attempt / 'setup.json', setup)
    measured = output / 'measured'
    measured.mkdir(exist_ok=True)
    order = []
    for phase, directory in [('warmup', warm), ('measured', measured)]:
        for block, condition, batch in schedule(tasks, settings, measured=phase == 'measured'):
            name = f'block-{block:03d}-{condition_id(condition)}.json'
            target = directory / name
            if target.exists():
                previous = json.loads(target.read_text())
                assert previous['condition'] == condition
                assert [row['task_id'] for row in previous['predictions']] == [task['task_id'] for task in batch]
                continue
            result = experiment.measure(batch, condition)
            result.update(block=block, attempt=attempt.name, phase=phase)
            save(target, result)
            order.append({'phase': phase, 'block': block, 'condition': condition_id(condition),
                          'compute_seconds': result['compute_seconds']})
            print(json.dumps({'phase': phase, 'block': block, 'condition': condition_id(condition),
                              'valid': sum(row['valid'] for row in result['predictions']),
                              'compute_seconds': result['compute_seconds']}), flush=True)
    save(attempt / 'completion.json', {'executed_order': order,
         'startup_seconds': setup['startup_seconds'],
         'warmup_seconds': sum(row['compute_seconds'] for row in order if row['phase'] == 'warmup'),
         'new_measured_seconds': sum(row['compute_seconds'] for row in order if row['phase'] == 'measured')})
    records = [json.loads(path.read_text()) for path in measured.glob('block-*.json')]
    assert sum(len(record['predictions']) for record in records) == len(tasks) * len(settings['conditions'])
    completion = output / 'completion.json'
    if not completion.exists():
        save(completion, {'measured_task_condition_outputs': len(tasks) * len(settings['conditions']),
                          'measured_blocks': len(records), 'evaluation': 'Labels have not been opened by inference.'})


if __name__ == '__main__':
    main()
