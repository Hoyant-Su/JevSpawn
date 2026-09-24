import argparse
import json
from pathlib import Path
import time

import torch

from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER
from methods.finite_constraints.adaptive import adaptive_run
from methods.finite_constraints.program import (
    assemble_factors, assignment_rows, compile_scopes, distinct_factors,
    domains, execute, instantiate, parse_puzzle,
)
from methods.finite_constraints.solver import solve
from methods.finite_constraints.symbolic import compile_symbolic
from methods.generated_schema.run import measured_call

from project_paths import ROOT


def write(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summary_record(result):
    return {key: value for key, value in result.items()
            if key not in {'execution', 'execution_records', 'compilation', 'generation', 'scopes', 'waves'}}


def finite_run(backend, state, settings, compilation, mode):
    started = time.perf_counter()
    program = instantiate(state, compilation, settings)
    execution = execute(backend, program, mode)
    factors = assemble_factors(program, execution)
    solution = solve(domains(state), [*distinct_factors(state), *factors], settings['max_solver_nodes'])
    prediction = assignment_rows(state, solution['assignment']) if solution['status'] == 'sat' else None
    return {'status': solution['status'], 'prediction_rows': prediction,
            'scopes': program['scopes'], 'finite_workers': len(program['nodes']),
            'worker_model_depth': 1, 'execution': execution, 'solver': solution,
            'total_seconds': time.perf_counter() - started + compilation['elapsed_seconds'],
            'compilation': compilation}


def symbolic_run(backend, state, settings):
    started = time.perf_counter()
    compilation = compile_symbolic(backend, state, settings)
    solution = solve(domains(state), [*distinct_factors(state), *compilation['factors']], settings['max_solver_nodes'])
    prediction = assignment_rows(state, solution['assignment']) if solution['status'] == 'sat' else None
    return {'status': solution['status'], 'prediction_rows': prediction,
            'compilation': compilation, 'solver': solution,
            'total_seconds': time.perf_counter() - started}


def direct_run(backend, row, state, settings, prompts):
    started = time.perf_counter()
    generation = measured_call(backend, lambda: backend.generate(
        [row['puzzle']], prompts['direct_system'], settings['compiler_tokens']), True)
    result = {'generation': generation, 'prediction_rows': None}
    try:
        prediction = json.loads(generation['result']['texts'][0])
        assert isinstance(prediction, list) and len(prediction) == len(state['positions'])
        assert all(isinstance(values, list) and len(values) == len(state['attributes']) for values in prediction)
        assert all(isinstance(value, str) for values in prediction for value in values)
        result.update(status='completed', prediction_rows=prediction)
    except (ValueError, AssertionError) as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    result['total_seconds'] = time.perf_counter() - started
    return result


def generation_records(record):
    generation = record.get('generation')
    if generation is not None:
        return [generation]
    compilation = record.get('compilation', {})
    return [compilation['generation']] if 'generation' in compilation else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    rows = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()]
    assert len(rows) == settings['task_count']
    prompts = json.loads((ROOT / 'configs/methods/finite_constraints/schema/prompts.json').read_text())
    native = json.loads(Path(settings['native_config']).read_text())
    native.update(batch_size=settings['root_batch_size'], branch_batch_size=settings['branch_batch_size'],
                  run_dir=str(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'settings': settings, 'native': native,
                'task_ids': [row['task_id'] for row in rows], 'prompts': prompts}
    if (args.output / 'protocol.json').exists():
        assert json.loads((args.output / 'protocol.json').read_text()) == protocol
    write(args.output / 'protocol.json', protocol)
    if (args.output / 'completion.json').exists():
        print('All declared task-method records are already complete.', flush=True)
        return
    CONTROLLER['option_template'] = prompts['option_template']
    backend = Backend(native)
    write(args.output / 'backend.json', backend.metadata)
    all_summaries, phase_directories = {}, {}
    for phase in settings['phases']:
        directory = args.output / phase
        if phase == 'warmup' and directory.exists():
            attempt = len(list(args.output.glob('warmup-restart-*'))) + 1
            directory = args.output / f'warmup-restart-{attempt}'
        directory.mkdir(exist_ok=phase == 'measured')
        phase_directories[phase] = directory.name
        summaries = []
        for index, row in enumerate(rows):
            outputs = {method: directory / f'{row["task_id"]}-{method}.json' for method in settings['methods']}
            if all(path.exists() for path in outputs.values()):
                summaries.extend(summary_record(json.loads(path.read_text())) for path in outputs.values())
                continue
            prepare_started = time.perf_counter()
            state = parse_puzzle(row['puzzle'])
            prepare_seconds = time.perf_counter() - prepare_started
            compilation_path = directory / f'{row["task_id"]}-compilation.json'
            if compilation_path.exists():
                compilation = json.loads(compilation_path.read_text())
            else:
                compilation = compile_scopes(backend, state, settings)
                write(compilation_path, compilation)
            order = settings['methods'][index % len(settings['methods']):] + settings['methods'][:index % len(settings['methods'])]
            for method in order:
                if outputs[method].exists():
                    summaries.append(summary_record(json.loads(outputs[method].read_text())))
                    continue
                torch.cuda.reset_peak_memory_stats(backend.device)
                started = time.perf_counter()
                try:
                    if method == 'finite_adaptive':
                        result = adaptive_run(backend, state, settings, compilation)
                    elif method.startswith('finite_'):
                        mode = {'finite_shared': 'tiled_shared', 'finite_independent': 'tiled_independent'}[method]
                        result = finite_run(backend, state, settings, compilation, mode)
                    elif method == 'symbolic':
                        result = symbolic_run(backend, state, settings)
                    else:
                        result = direct_run(backend, row, state, settings, prompts)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as error:
                    result = {'status': 'failed', 'prediction_rows': None,
                              'error': f'{type(error).__name__}: {error}',
                              'total_seconds': time.perf_counter() - started}
                    if method.startswith('finite_'):
                        result['compilation'] = compilation
                        result['total_seconds'] += compilation['elapsed_seconds']
                    elif hasattr(error, 'generation'):
                        result['generation'] = error.generation
                records = generation_records(result)
                peaks = [torch.cuda.max_memory_allocated(backend.device)]
                peaks.extend(record['peak_allocated_bytes'] for record in records)
                if 'execution' in result:
                    peaks.append(result['execution']['peak_allocated_bytes'])
                peaks.extend(record['peak_allocated_bytes'] for record in result.get('execution_records', []))
                intervals = [value for record in records for decoded in record['decode']
                             for value in decoded['inter_token_seconds']]
                result.update(task_id=row['task_id'], method=method, phase=phase,
                              source_preparation_seconds=prepare_seconds,
                              peak_allocated_bytes=max(peaks),
                              max_itl_ms=max(intervals, default=0) * 1000,
                              intervals_over_100ms=sum(value >= .1 for value in intervals))
                result['total_seconds'] += prepare_seconds
                write(outputs[method], result)
                summary = summary_record(result)
                summaries.append(summary)
                write(directory / 'summary.json', summaries)
                print(json.dumps({key: summary[key] for key in ['phase', 'task_id', 'method', 'status', 'total_seconds']}), flush=True)
        write(directory / 'summary.json', summaries)
        all_summaries[phase] = summaries
    write(args.output / 'completion.json', {'phases': list(all_summaries),
                                          'phase_directories': phase_directories,
                                          'task_methods_per_phase': {key: len(value) for key, value in all_summaries.items()}})


if __name__ == '__main__':
    main()
