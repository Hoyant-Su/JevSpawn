import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from baselines.common.sandbox_protocol import (
    POLICY, PROMPTS, PROTOCOL, SandboxInfrastructureError, bootstrap_command,
    diagnostic_after_bootstrap, evaluate_one,
)


FIXTURE = json.loads(Path('tests/baselines/common/fixtures/sandbox_protocol_v1.json').read_text())


def cpu_bootstrap(sandbox, program):
    return [sys.executable, '-c', 'import runpy; runpy.run_path(' + repr(str(program)) + ', run_name="__main__")']


class SandboxProtocolTests(unittest.TestCase):
    def test_real_python_bootstrap_preserves_pass_failure_syntax_and_timeout(self):
        directory = Path(FIXTURE['work_directory'])
        directory.mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(work_dir=directory, sandbox=sys.executable,
                               timeout=FIXTURE['timeout_seconds'], memory_mb=FIXTURE['memory_mb'])
        for case in FIXTURE['cases']:
            with self.subTest(task=case['task_id']), patch('baselines.common.sandbox_protocol.sandbox_command', cpu_bootstrap):
                result = evaluate_one(case, {'dataset': 'bootstrap_protocol_cpu_test', 'test_code': case['test_code']}, args)
                self.assertEqual(result['status'], case['status'])
                self.assertNotIn(PROTOCOL['started_marker_prefix'], result['diagnostic'])
                if 'diagnostic' in case:
                    self.assertIn(case['diagnostic'], result['diagnostic'])
                raw = (Path(result['artifact_directory']) / POLICY['stderr_name']).read_text()
                self.assertIn(PROTOCOL['started_marker_prefix'], raw)

    def test_syntax_error_in_candidate_file_cannot_hide_bootstrap_start(self):
        directory = Path(FIXTURE['work_directory'])
        directory.mkdir(parents=True, exist_ok=True)
        program = directory / 'invalid_candidate_file.py'
        program.write_text(next(case['solution'] for case in FIXTURE['cases'] if case['task_id'] == 'bootstrap/syntax'))
        marker = PROTOCOL['started_marker_prefix'] + 'syntax_file_cpu_test'
        with patch('baselines.common.sandbox_protocol.sandbox_command', cpu_bootstrap):
            command = bootstrap_command(sys.executable, program, marker)
        result = subprocess.run(command, capture_output=True, text=True, timeout=POLICY['preflight_timeout_seconds'])
        self.assertIn('SyntaxError', diagnostic_after_bootstrap(result.stderr, marker))
        self.assertNotIn(marker, diagnostic_after_bootstrap(result.stderr, marker))

    def test_actual_unavailable_sandbox_diagnostics_are_infrastructure_errors(self):
        for filename in FIXTURE['unavailable_records']:
            record = json.loads(Path(filename).read_text())
            for action in record['actions']:
                if action['tool'] != 'run_tests':
                    continue
                with self.subTest(record=filename), self.assertRaises(SandboxInfrastructureError):
                    diagnostic_after_bootstrap(action['result']['diagnostic'],
                                               PROTOCOL['started_marker_prefix'] + 'unstarted_replay')


if __name__ == '__main__':
    unittest.main()
