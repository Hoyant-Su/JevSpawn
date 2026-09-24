import argparse
import collections
import concurrent.futures
import json

import math
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from jev_spawn.infra.configuration import ROOT
from jev_spawn.infra.prompts import load_prompt


POLICY = json.loads((ROOT / 'configs/jevspawn/evaluation.json').read_text())
TEMPLATES = load_prompt(POLICY['templates'])
SEED = json.loads((ROOT / POLICY['seed_config']).read_text())['seed']


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sandbox_command(sandbox, program):
    return [argument.format(sandbox=sandbox, program=program, seed=SEED)
            for argument in POLICY['command']]


def evaluate_one(sample, test, args):
    started = time.monotonic()
    marker = POLICY['marker_prefix'] + uuid.uuid4().hex
    limits = {POLICY['memory_resource']: args.memory_mb * POLICY['memory_unit_bytes'],
              POLICY['cpu_resource']: math.ceil(args.timeout), **POLICY['fixed_resources']}
    program_text = TEMPLATES['program'].format(
        limits='\n'.join(TEMPLATES['resource'].format(name=name, value=value) for name, value in limits.items()),
        solution=repr(sample['solution']), tests=repr(test['test_code']), marker=repr(marker))
    path = Path(tempfile.mkdtemp(prefix=POLICY['run_prefix'], dir=args.work_dir))
    program = path / POLICY['program_name']
    program.write_text(program_text)
    stdout_path, stderr_path = path / POLICY['stdout_name'], path / POLICY['stderr_name']
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        try:
            process = subprocess.run(sandbox_command(args.sandbox, program), stdout=stdout, stderr=stderr,
                                     timeout=args.timeout + POLICY['process_grace_seconds'])
            passed = (process.returncode == POLICY['success_returncode'] and
                      stdout_path.read_text(errors='replace').rstrip().endswith(marker))
            status = POLICY['statuses']['passed'] if passed else POLICY['statuses']['failed']
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            status, returncode = POLICY['statuses']['timeout'], None
    diagnostic = stderr_path.read_text(errors='replace')[-POLICY['diagnostic_characters']:]
    return {'task_id': sample['task_id'], 'dataset': test['dataset'], 'status': status,
            'returncode': returncode, 'elapsed_seconds': time.monotonic() - started,
            'diagnostic': diagnostic, 'artifact_directory': str(path)}


def main():
    parser = argparse.ArgumentParser(description=POLICY['description'])
    types = {'path': Path, 'int': int, 'float': float}
    for argument in POLICY['arguments']:
        parser.add_argument(argument['flag'], type=types[argument['type']], required=argument['required'])
    args = parser.parse_args()
    args.sandbox = args.sandbox.resolve(strict=True)
    args.work_dir = args.work_dir.resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(args.solutions)
    if len({sample['task_id'] for sample in samples}) != len(samples):
        parser.error(TEMPLATES['duplicate_error'])
    tests = {row['task_id']: row for row in read_jsonl(args.tests)}
    unknown = {sample['task_id'] for sample in samples} - tests.keys()
    if unknown:
        parser.error(TEMPLATES['unknown_error'].format(ids=sorted(unknown)))
    program = Path(tempfile.mkdtemp(prefix=POLICY['preflight_prefix'], dir=args.work_dir)) / POLICY['program_name']
    program.write_text(TEMPLATES['preflight'])
    subprocess.run(sandbox_command(args.sandbox, program), check=True,
                   timeout=POLICY['preflight_timeout_seconds'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda sample: evaluate_one(sample, tests[sample['task_id']], args), samples))
    with args.output.open('w') as stream:
        for result in results:
            stream.write(json.dumps(result) + '\n')
    by_dataset = collections.defaultdict(collections.Counter)
    for result in results:
        by_dataset[result['dataset']][result['status']] += True
    passed = POLICY['statuses']['passed']
    summary = {
        'sandbox': TEMPLATES['sandbox_description'], 'evaluation': TEMPLATES['evaluation_description'],
        'counts': dict(collections.Counter(result['status'] for result in results)),
        'datasets': {dataset: {'total': sum(counts.values()), 'passed': counts[passed],
                              'final_solution_pass_rate': counts[passed] / sum(counts.values())}
                     for dataset, counts in by_dataset.items()},
    }
    args.output.with_suffix(POLICY['summary_suffix']).write_text(json.dumps(summary) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
