"""Run the original Self-Refine GSM loop through the shared model service."""

import ast
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
from types import SimpleNamespace
import uuid


def original_loop(upstream, provider):
    root = Path(upstream)
    namespace = {
        'traceback': traceback, 'ENGINE': 'Qwen3.5-4B',
        'openai_api': SimpleNamespace(OpenaiAPIWrapper=provider),
        'open': lambda path, mode: builtins.open(root / path, mode),
        'print': lambda *args, **kwargs: None,
    }
    definitions = [
        ('src/utils.py', {'Prompt', 'retry_parse_fail_prone_cmd'}),
        ('src/gsm/task_init.py', {'GSMInit'}),
        ('src/gsm/feedback.py', {'GSMFeedback'}),
        ('src/gsm/run.py', {'iterative_gsm'}),
    ]
    for relative, names in definitions:
        path = root / relative
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
        exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['iterative_gsm']


def execute_solution(source, settings):
    spec = importlib.util.spec_from_file_location('self_refine_sandbox', settings['evaluation_script'])
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    directory = Path(tempfile.mkdtemp(prefix='self-refine-', dir=settings['work_dir']))
    marker = 'SOLUTION_' + uuid.uuid4().hex
    memory = settings['sandbox_memory_mb'] * 1024**2
    seconds = settings['execution_seconds']
    limits = [('RLIMIT_AS', memory), ('RLIMIT_CPU', seconds), ('RLIMIT_FSIZE', 1024**2),
              ('RLIMIT_NPROC', 32), ('RLIMIT_NOFILE', 64), ('RLIMIT_CORE', 0)]
    program = directory / 'check.py'
    program.write_text(
        'import json\nimport resource\n'
        + '\n'.join(f'resource.setrlimit(resource.{key}, ({value}, {value}))' for key, value in limits)
        + "\nnamespace = {'__name__': '__main__'}\n"
        + f"exec(compile({source!r}, 'solution.py', 'exec'), namespace)\n"
        + f"print({marker!r} + json.dumps(namespace['solution']()), flush=True)\n")
    started = time.perf_counter()
    with (directory / 'stdout').open('wb') as stdout, (directory / 'stderr').open('wb') as stderr:
        process = subprocess.run(evaluator.sandbox_command(settings['sandbox'], program),
                                 stdout=stdout, stderr=stderr, timeout=seconds + 2)
    lines = (directory / 'stdout').read_text().splitlines()
    answer = json.loads(lines[-1][len(marker):]) if process.returncode == 0 and lines and lines[-1].startswith(marker) else None
    return {'answer': answer, 'returncode': process.returncode,
            'elapsed_seconds': time.perf_counter() - started,
            'diagnostic': (directory / 'stderr').read_text()[-4000:], 'directory': str(directory)}


def solve(row, complete, settings, prompts):
    calls = []

    class Provider:
        @staticmethod
        def call(prompt, engine, max_tokens, stop_token, temperature):
            cap = min(max_tokens, settings['max_new_tokens'])
            output = complete([{'role': 'system', 'content': prompts['system']},
                               {'role': 'user', 'content': prompt}],
                              cap, temperature, stop=stop_token)
            calls.append({'prompt': prompt, 'upstream_max_tokens': max_tokens,
                          'max_new_tokens': cap, 'temperature': temperature,
                          'stop': stop_token, 'output': output[0]})
            return output

        @staticmethod
        def get_first_response(output):
            return output[0]

    field = row['fields']['q0']
    options = '\n'.join(option['id'] + '. ' + option['description'] for option in field['options'])
    question = prompts['question'].format(state=row['state'], question=field['question'], options=options)
    logs = original_loop(settings['upstream'], Provider)(
        question, settings['max_attempts'], 'rich', settings['temperature'])
    if logs is None:
        return {'task_id': row['task_id'], 'answer': None, 'calls': calls,
                'failure': 'Original parse retry limit exhausted.'}
    source = logs[-1]['solution_fixed']
    execution = execute_solution(source, settings)
    return {'task_id': row['task_id'], 'answer': execution['answer'], 'calls': calls,
            'refinements': logs, 'solution': source, 'execution': execution}
