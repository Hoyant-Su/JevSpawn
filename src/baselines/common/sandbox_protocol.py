import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from jev_spawn.cli.evaluate import POLICY, TEMPLATES, sandbox_command
from jev_spawn.infra.configuration import ROOT
from jev_spawn.infra.prompts import load_prompt


PROTOCOL = json.loads((ROOT / 'configs/baselines/common/candidates/sandbox_protocol_v1.json').read_text())
PROMPTS = load_prompt(PROTOCOL['templates'])


class SandboxInfrastructureError(RuntimeError):
    pass


def bootstrap_command(sandbox, program, marker):
    command = sandbox_command(sandbox, program)
    index = PROTOCOL['bootstrap_argument_index']
    command[index] = PROMPTS['bootstrap'].format(marker=repr(marker), bootstrap=command[index])
    return command


def diagnostic_after_bootstrap(stderr, marker):
    lines = stderr.splitlines(keepends=True)
    started = any(line.rstrip('\r\n') == marker for line in lines)
    diagnostic = ''.join(line for line in lines if line.rstrip('\r\n') != marker)
    if not started:
        raise SandboxInfrastructureError(diagnostic)
    return diagnostic


def preflight(sandbox, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    program = Path(tempfile.mkdtemp(prefix=POLICY['preflight_prefix'], dir=directory)) / POLICY['program_name']
    program.write_text(TEMPLATES['preflight'])
    marker = PROTOCOL['started_marker_prefix'] + uuid.uuid4().hex
    result = subprocess.run(bootstrap_command(sandbox, program, marker), capture_output=True, text=True,
                            timeout=POLICY['preflight_timeout_seconds'])
    diagnostic = diagnostic_after_bootstrap(result.stderr, marker)
    if result.returncode != POLICY['success_returncode']:
        raise SandboxInfrastructureError(diagnostic)


def evaluate_one(sample, test, args):
    started = time.monotonic()
    marker = POLICY['marker_prefix'] + uuid.uuid4().hex
    bootstrap_marker = PROTOCOL['started_marker_prefix'] + uuid.uuid4().hex
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
            result = subprocess.run(bootstrap_command(args.sandbox, program, bootstrap_marker),
                stdout=stdout, stderr=stderr, timeout=args.timeout + POLICY['process_grace_seconds'])
            passed = (result.returncode == POLICY['success_returncode'] and
                      stdout_path.read_text(errors='replace').rstrip().endswith(marker))
            status = POLICY['statuses']['passed'] if passed else POLICY['statuses']['failed']
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            status, returncode = POLICY['statuses']['timeout'], None
    diagnostic = diagnostic_after_bootstrap(stderr_path.read_text(errors='replace'), bootstrap_marker)
    return {'task_id': sample['task_id'], 'dataset': test['dataset'], 'status': status,
            'returncode': returncode, 'elapsed_seconds': time.monotonic() - started,
            'diagnostic': diagnostic[-POLICY['diagnostic_characters']:], 'artifact_directory': str(path)}
