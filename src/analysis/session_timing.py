from collections import Counter
import json
from pathlib import Path
from statistics import mean, median


def collect_timing(cell, settings):
    sessions, delivered = [], []
    observed = {size: [] for size in settings['batch_sizes']}
    for directory in sorted(Path(cell['run']).glob(settings['session_glob'])):
        paths = {name: directory / filename for name, filename in settings['files'].items()}
        missing = [name for name, path in paths.items() if not path.exists()]
        session = {'session': directory.name, 'missing_telemetry': missing}
        sessions.append(session)
        if missing:
            continue
        completion = json.loads(paths['completion'].read_text())
        runtime = json.loads(paths['runtime'].read_text())
        batches = json.loads(paths['batches'].read_text())
        assert len(completion['task_ids']) == completion['tasks']
        delivered.extend(completion['task_ids'])
        session['tasks'] = completion['tasks']
        session['elapsed_seconds'] = json.loads(paths['elapsed'].read_text())['seconds']
        session['shared_config'] = runtime['shared_config']
        session['batch_shapes'] = dict(Counter(batch['batch_size'] for batch in batches))
        session['model_batches'] = len(batches)
        if batches:
            session['peak_allocated_bytes'] = max(batch['peak_allocated_bytes'] for batch in batches)
        session['output_tokens'] = sum(sum(batch['output_tokens']) for batch in batches)
        session['decode'] = []
        for size in settings['batch_sizes']:
            intervals = [value for batch in batches if batch['batch_size'] == size
                         for row in batch['decode'] for value in row['inter_token_seconds']]
            observed[size].extend(intervals)
            metric = {'batch_size': size, 'observed_token_intervals': len(intervals)}
            if intervals:
                assert min(intervals) > 0
                metric.update(mean_itl_seconds=mean(intervals), median_itl_seconds=median(intervals))
            session['decode'].append(metric)
    expected = set(cell['task_ids'])
    assert len(delivered) == len(set(delivered)) and set(delivered) <= expected
    complete = bool(sessions) and all(not session['missing_telemetry'] for session in sessions) and set(delivered) == expected
    metrics = {'timing_complete': complete, 'timing_covered_tasks': len(delivered)}
    if complete:
        metrics['active_run_seconds'] = sum(session['elapsed_seconds'] for session in sessions)
        metrics['model_batches'] = sum(session['model_batches'] for session in sessions)
        metrics['model_executed'] = any(session['model_batches'] for session in sessions)
        memory = [session['peak_allocated_bytes'] for session in sessions if session['model_batches']]
        if memory:
            metrics['leader_peak_allocated_bytes'] = max(memory)
        for size, intervals in observed.items():
            if intervals:
                metrics[f'b{size}_median_itl_seconds'] = median(intervals)
                metrics[f'b{size}_mean_itl_seconds'] = mean(intervals)
                metrics[f'b{size}_observed_token_intervals'] = len(intervals)
    return {'method': cell['method'], 'dataset': cell['dataset'], 'metrics': metrics, 'sessions': sessions,
            'timing_scope': settings['scope']}
