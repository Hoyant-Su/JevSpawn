import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

from baselines.lats.adapter import load_upstream, mbpp_task
from baselines.lats.stages import run_stage
from baselines.official.model_service import GenerationService
from jev_spawn.infra.backend import Backend


def save(path, value):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def evaluate(output, tasks, settings):
    commits = [json.loads(path.read_text()) for path in sorted((output / 'blocks').glob('block-*.json'))]
    rows = [row for commit in commits for row in commit['results']]
    assert [row['task_id'] for row in rows] == [row['task_id'] for row in tasks]
    solutions = output / 'solutions.jsonl'
    solutions.write_text(''.join(json.dumps({'task_id': row['task_id'], 'solution': row['solution']}) + '\n'
                                 for row in rows if row['status'] == 'completed'))
    subprocess.run([sys.executable, settings['evaluation_script'], '--solutions', str(solutions),
                    '--tests', settings['heldout_tests'], '--sandbox', settings['sandbox'],
                    '--output', str(output / 'heldout.jsonl'), '--work-dir', str(output / 'heldout-tools'),
                    '--workers', str(settings['root_batch_size']), '--timeout', str(settings['offline_timeout_seconds']),
                    '--memory-mb', str(settings['sandbox_memory_mb'])], check=True)
    scores = read_rows(output / 'heldout.jsonl')
    batches = [batch for commit in commits for batch in json.loads((Path(commit['directory']) / 'batches.json').read_text())]
    intervals = [value for batch in batches for row in batch['decode'] for value in row['inter_token_seconds']]
    seconds = sum(commit['seconds'] for commit in commits)
    passed = sum(row['status'] == 'passed' for row in scores)
    summary = {'tasks': len(rows), 'blocks': len(commits), 'actual_root_batch_sizes': [len(c['results']) for c in commits],
               'status_counts': dict(Counter(row['status'] for row in rows)),
               'heldout_passes': passed, 'accuracy': passed / len(rows),
               'public_passes': sum(row.get('public_test_passed', False) for row in rows),
               'seconds': seconds, 'tasks_per_second': len(rows) / seconds,
               'mean_request_to_answer_seconds': statistics.mean(row['elapsed_seconds'] for row in rows),
               'model_calls': sum(row['model_calls'] for row in rows),
               'actual_model_sequences': sum(batch['batch_size'] for batch in batches),
               'model_batch_size_histogram': dict(Counter(batch['batch_size'] for batch in batches)),
               'input_tokens': sum(sum(batch['input_tokens']) for batch in batches),
               'output_tokens': sum(sum(batch['output_tokens']) for batch in batches),
               'truncated_model_sequences': sum(sum(batch['truncated']) for batch in batches),
               'peak_allocated_bytes': max(batch['peak_allocated_bytes'] for batch in batches),
               'peak_reserved_bytes': max(batch['peak_reserved_bytes'] for batch in batches),
               'itl_median_seconds': statistics.median(intervals),
               'itl_p95_seconds': statistics.quantiles(intervals, n=100)[94],
               'itl_max_seconds': max(intervals), 'itl_over_100ms': sum(v >= .1 for v in intervals),
               'node_event_counts': dict(Counter(event['function'] for row in rows for event in row['node_events'])),
               'max_observed_depth': max((event['depth'] for row in rows for event in row['node_events']), default=0),
               'scope': 'Original LATS core,500 original MBPP test tasks. Hidden tests read only after all terminal outputs committed. Failed tasks remain in the500 denominator.'}
    save(output / 'summary.json', summary)
    print(json.dumps({'event': 'complete', **summary}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    native = json.loads(Path(settings['native_config']).read_text())
    tasks, warm = read_rows(settings['formal_tasks']), read_rows(settings['tasks'])[:8]
    assert len(tasks) == settings['task_count'] == 500
    assert native['batch_size'] == settings['root_batch_size'] == 8
    assert native['dtype'] == 'bfloat16' and native['max_input_tokens'] == 8192
    assert not ({row['task_id'] for row in tasks} & {row['task_id'] for row in warm})
    assert len({row['task_id'] for row in tasks}) == 500
    for row in [*tasks, *warm]:
        mbpp_task(row)
    revision = subprocess.check_output(['git', '-C', settings['upstream'], 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == settings['upstream_revision']
    protocol = {'settings': settings, 'native': native, 'task_ids': [row['task_id'] for row in tasks]}
    if args.resume:
        assert json.loads((args.output / 'protocol.json').read_text()) == protocol
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'blocks').mkdir()
        save(args.output / 'protocol.json', protocol)
    committed = sorted((args.output / 'blocks').glob('block-*.json'))
    assert [p.name for p in committed] == [f'block-{i:03d}.json' for i in range(len(committed))]
    for i, path in enumerate(committed):
        assert [r['task_id'] for r in json.loads(path.read_text())['results']] == [r['task_id'] for r in tasks[i*8:(i+1)*8]]
    if len(committed) == (len(tasks) + 7) // 8:
        evaluate(args.output, tasks, settings)
        return
    attempt = args.output / f'attempt-{len(list(args.output.glob("attempt-*"))):03d}'
    attempt.mkdir()
    started = time.perf_counter()
    core, result_type = load_upstream(settings['upstream'])
    backend = Backend(native)
    save(attempt / 'setup.json', {'model_load_seconds': time.perf_counter() - started,
         'backend': backend.metadata, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
         'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True),
         'resumed_complete_blocks': len(committed)})
    service = GenerationService(backend, native['batch_size'], settings['batch_wait_seconds'])
    try:
        run_stage(core, result_type, service, warm, settings, attempt, 'warmup')
        service.records.clear()
        for start in range(len(committed) * 8, len(tasks), 8):
            name = f'block-{start // 8:03d}'
            results = run_stage(core, result_type, service, tasks[start:start+8], settings, attempt, name)
            directory = attempt / name
            measured = json.loads((directory / 'summary.json').read_text())
            save(args.output / 'blocks' / f'{name}.json', {'directory': str(directory),
                 'seconds': measured['elapsed_seconds'], 'results': results, 'source_offset': start,
                 'timing': measured['timing']})
            print(json.dumps({'event': 'block_committed', 'completed_tasks': start + len(results),
                              'seconds': measured['elapsed_seconds'], 'completed': measured['completed'],
                              'itl_max_ms': measured['timing']['maximum_ms']}), flush=True)
            service.records.clear()
    finally:
        service.close()
    evaluate(args.output, tasks, settings)


if __name__ == '__main__':
    main()
