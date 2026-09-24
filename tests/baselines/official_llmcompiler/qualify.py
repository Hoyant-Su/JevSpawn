import argparse
import asyncio
from functools import partial
import json
import statistics
import time
import traceback
from pathlib import Path

from baselines.official.model_service import GenerationService
from baselines.official_llmcompiler.adapter import solve
from jev_spawn.infra.backend import Backend
from src.utils.logger_utils import enable_logging


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


async def run_stage(service, rows, config):
    jobs = []
    for row in rows:
        field = row['fields']['q0']
        task = {'task_id': row['task_id'], 'state': row['state'], 'question': field['question'],
                'options': '\n'.join(f"{option['id']}: {option['description']}" for option in field['options'])}
        jobs.append(solve(task, partial(service.complete, task_id=row['task_id']), config))
    outputs = await asyncio.gather(*jobs, return_exceptions=True)
    results = []
    for row, output in zip(rows, outputs):
        if isinstance(output, BaseException):
            results.append({'task_id': row['task_id'], 'status': 'error',
                            'error_type': type(output).__name__, 'error': str(output)})
        else:
            output['status'] = 'valid' if output['answer'].strip() in [o['id'] for o in row['fields']['q0']['options']] else 'invalid_answer'
            results.append(output)
    return results


def main():
    parser = argparse.ArgumentParser()
    for name in ['native-config', 'config', 'tasks', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--task-count', type=int, required=True)
    parser.add_argument('--batch-wait-seconds', type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    native = json.loads(args.native_config.read_text())
    config = json.loads(args.config.read_text())
    rows = [json.loads(line) for line in args.tasks.read_text().splitlines()][:args.task_count]
    assert len(rows) == args.task_count == native['batch_size']
    native['run_dir'] = str(args.output)
    save(args.output / 'protocol.json', {'native': native, 'config': config, 'tasks': str(args.tasks),
         'task_ids': [r['task_id'] for r in rows], 'batch_wait_seconds': args.batch_wait_seconds,
         'upstream_core': 'Original LLMCompiler planner, parser, task-fetching scheduler and joiner loop.'})
    enable_logging(False)
    backend = Backend(native)
    service = GenerationService(backend, native['batch_size'], args.batch_wait_seconds)
    try:
        for stage in ['warmup', 'measured']:
            start_record = len(service.records)
            started = time.perf_counter()
            results = asyncio.run(run_stage(service, rows, config))
            duration = time.perf_counter() - started
            calls = service.records[start_record:]
            save(args.output / f'{stage}-results.json', results)
            save(args.output / f'{stage}-batches.json', calls)
            assert not any(row.get('error_type') == 'TimeoutError' for row in results), 'An upstream task timed out.'
            intervals = sorted(t for call in calls for row in call['decode'] for t in row['inter_token_seconds'])
            summary = {'stage': stage, 'tasks': len(rows), 'valid_answers': sum(r['status'] == 'valid' for r in results),
                       'elapsed_seconds': duration, 'actual_batch_sizes': [r['batch_size'] for r in calls],
                       'itl_median_ms': statistics.median(intervals) * 1000,
                       'itl_p95_ms': intervals[int(.95 * (len(intervals) - 1))] * 1000,
                       'itl_max_ms': max(intervals) * 1000,
                       'intervals_over_100ms': sum(t >= .1 for t in intervals),
                       'truncated_calls': sum(sum(r['truncated']) for r in calls),
                       'peak_allocated_gib': max(r['peak_allocated_bytes'] for r in calls) / 2**30}
            save(args.output / f'{stage}-summary.json', summary)
            print(json.dumps(summary), flush=True)
    except Exception as error:
        save(args.output / 'failure.json', {'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        service.close()


if __name__ == '__main__':
    main()
