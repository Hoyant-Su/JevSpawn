import argparse
import json
import os
from pathlib import Path
import statistics
import time

from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER
from methods.decision_program.compiler import compatible_jobs, compile_programs, instantiate, reduce_results
from methods.generated_schema.run import measured_call, write
from methods.program_execution.grouped import score_grouped
from jev_spawn.infra.prompts import resolve_prompts


def execute(backend, jobs, programs, settings, directory):
    valid = [job for job in jobs if job['task_id'] in programs]
    records = {mode: [] for mode in settings['modes']}
    outputs = {mode: {} for mode in settings['modes']}
    for index, batch in enumerate(compatible_jobs(valid, programs, settings['batch_size'])):
        groups = [instantiate(job, programs[job['task_id']]) for job in batch]
        order = settings['modes'][index % len(settings['modes']):] + settings['modes'][:index % len(settings['modes'])]
        for mode in order:
            record = measured_call(backend, lambda: score_grouped(backend, groups, mode))
            started = time.perf_counter()
            reduced = {job['task_id']: reduce_results(programs[job['task_id']], decisions)
                       for job, decisions in zip(batch, record['result']['groups'])}
            record.update(root_ids=[job['task_id'] for job in batch],
                          reduction_seconds=time.perf_counter() - started, method_order=order)
            records[mode].append(record)
            outputs[mode].update(reduced)
            write(directory / (mode + '-calls.json'), records[mode])
            write(directory / (mode + '-outputs.json'), outputs[mode])
    return {mode: {'valid_roots': len(outputs[mode]), 'assigned_roots': len(jobs),
                   'typed_workers': sum(len(job['items']) for job in valid),
                   'execution_seconds': sum(call['elapsed_seconds'] + call['reduction_seconds'] for call in calls),
                   'peak_allocated_bytes': max((call['peak_allocated_bytes'] for call in calls), default=None),
                   'root_batch_sizes': [len(call['root_ids']) for call in calls]}
            for mode, calls in records.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = resolve_prompts(json.loads(args.settings.read_text()))
    config = resolve_prompts(json.loads(Path(settings['native_config']).read_text()))
    jobs = [json.loads(line) for line in Path(settings['jobs']).read_text().splitlines()]
    assert len(jobs) == settings['job_count']
    assert settings['batch_size'] == config['batch_size']
    args.output.mkdir(parents=True)
    CONTROLLER['option_template'] = settings['option_template']
    write(args.output / 'protocol.json', {'settings': settings, 'native': config,
                                        'job_ids': [job['task_id'] for job in jobs],
                                        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']})
    backend = Backend(config)
    write(args.output / 'backend.json', backend.metadata)
    for phase in ['warmup', 'measured']:
        directory = args.output / phase
        directory.mkdir()
        compiled = compile_programs(backend, jobs, settings)
        write(directory / 'compilation.json', compiled)
        summary = execute(backend, jobs, compiled['programs'], settings, directory)
        record = compiled['generation']
        intervals = [value for row in record['decode'] for value in row['inter_token_seconds']]
        for mode, value in summary.items():
            value.update(compiler_seconds=record['elapsed_seconds'],
                         total_model_and_reduction_seconds=record['elapsed_seconds'] + value['execution_seconds'],
                         compiler_output_tokens=record['result']['output_tokens'],
                         compiler_truncated=record['result']['truncated'],
                         compiler_max_itl_ms=1000 * max(intervals),
                         compiler_median_itl_ms=1000 * statistics.median(intervals),
                         total_peak_allocated_bytes=max(record['peak_allocated_bytes'], value['peak_allocated_bytes'] or 0))
        write(directory / 'summary.json', summary)
        print(json.dumps({'phase': phase, 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
