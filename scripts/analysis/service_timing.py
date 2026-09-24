import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean, median

import yaml

from baselines.common.persistence import save


def summarize(values):
    return {'count': len(values), 'mean': mean(values) if values else None,
            'median': median(values) if values else None}


def measure(cell):
    reports = json.loads(Path(cell['evaluation']).read_text())['runs']
    intervals, first_tokens, finite_times, generation_sizes = [], [], [], []
    cohorts = defaultdict(list)
    first_token_cohorts = defaultdict(list)
    finite_cohorts = defaultdict(list)
    counters = Counter()
    sessions = []
    completed_ids, session_seconds, missing_timing = [], [], []
    for report in reports:
        for directory in sorted(Path(report['run']).glob('session-*')):
            completion = directory / 'completion.json'
            if completion.exists():
                timing = json.loads(completion.read_text())
                completed_ids.extend(timing['task_ids'])
                session_seconds.append(timing['elapsed_seconds'])
            else:
                missing_timing.append(str(directory))
        for path in sorted(Path(report['run']).glob('session-*/batches.json')):
            batches = json.loads(path.read_text())
            sessions.append({'path': str(path), 'records': len(batches)})
            counters['logged_records'] += len(batches)
            for batch in batches:
                if batch.get('operation') == 'finite':
                    finite_times.append(batch['elapsed_seconds'])
                    finite_cohorts[batch['batch_size']].append(batch['elapsed_seconds'])
                    counters['finite_rows'] += batch['batch_size']
                    counters['finite_logical_input_tokens'] += batch['structured']['logical_input_tokens']
                    counters['finite_computed_input_tokens'] += batch['structured']['computed_input_tokens']
                if 'messages' in batch:
                    values = [value for row in batch['decode']
                              for value in row['inter_token_seconds']]
                    intervals.extend(values)
                    cohorts[batch['batch_size']].extend(values)
                    first = [row['ttft_seconds'] for row in batch['decode']
                             if row['ttft_seconds'] is not None]
                    first_tokens.extend(first)
                    first_token_cohorts[batch['batch_size']].extend(first)
                    generation_sizes.append(batch['batch_size'])
                    counters['generation_rows'] += batch['batch_size']
                    counters['output_tokens'] += sum(batch['output_tokens'])
                    counters['text_logical_input_tokens'] += sum(batch['input_tokens'])
                    counters['generation_service_seconds'] += batch['elapsed_seconds']
    expected_ids = [row['task_id'] for report in reports for row in report['scores']]
    complete_timing = (not missing_timing and sorted(completed_ids) == sorted(expected_ids)
                       and len(completed_ids) == len(set(completed_ids)))
    throughput = None
    if complete_timing:
        elapsed = sum(session_seconds)
        throughput = {
            'replica_seconds': elapsed,
            'attempted_tasks_per_second': len(expected_ids) / elapsed,
            'correct_tasks_per_second': cell['correct'] / elapsed if cell['accuracy'] is not None else None,
            'text_output_tokens_per_second': counters['output_tokens'] / elapsed,
            'text_logical_input_tokens_per_second': counters['text_logical_input_tokens'] / elapsed,
            'finite_logical_input_tokens_per_second': counters['finite_logical_input_tokens'] / elapsed,
            'model_scored_fields_per_second': counters['finite_rows'] / elapsed,
        }
    return {'method': cell['method'], 'dataset': cell['dataset'],
            'evaluation': cell['evaluation'], 'declared_tasks': cell['declared_tasks'],
            'sessions': sessions, 'counts': dict(counters) | {
                'generation_batches': len(generation_sizes)},
            'lifecycle_throughput': throughput,
            'timing_coverage': {'complete': complete_timing, 'missing_sessions': missing_timing,
                'recorded_task_lifecycles': len(completed_ids), 'expected_tasks': len(expected_ids)},
            'text_inter_token_seconds': summarize(intervals),
            'text_time_to_first_token_seconds': summarize(first_tokens),
            'text_time_to_first_token_seconds_by_admitted_batch_size': {
                size: summarize(values) for size, values in sorted(first_token_cohorts.items())},
            'text_inter_token_seconds_by_admitted_batch_size': {
                size: summarize(values) for size, values in sorted(cohorts.items())},
            'finite_service_batch_seconds': summarize(finite_times),
            'finite_service_seconds_by_admitted_batch_size': {
                size: summarize(values) for size, values in sorted(finite_cohorts.items())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = json.loads(parser.parse_args().config.read_text())
    shared = yaml.safe_load(Path(settings['shared_config']).read_text())
    cells = json.loads(Path(settings['comparison']).read_text())['completed_cells']
    with ProcessPoolExecutor(max_workers=shared['runtime']['cpu_threads']) as pool:
        records = list(pool.map(measure, cells))
    save(settings['output'], {'interpretation': settings['interpretation'], 'records': records})
    print(json.dumps({'completed_cells': len(records), 'output': settings['output']}))


if __name__ == '__main__':
    main()
