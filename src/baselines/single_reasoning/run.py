import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tool_agents'))

import torch

from agents import run_batch
from baselines.formal_choices.run import blocks, completed, load_protocol, read, rows, save
from baselines.tool_agents.run import timed_generator
from jev_spawn.infra.backend import Backend


def execute(backend, tasks, protocol, directory, seed):
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.synchronize(backend.device)
    torch.cuda.reset_peak_memory_stats(backend.device)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    results = run_batch(timed_generator(backend), tasks, 'reasoning',
                        protocol['agent'], protocol['prompts'])
    torch.cuda.synchronize(backend.device)
    finished = time.perf_counter()
    assert [row['task_id'] for row in results] == [row['task_id'] for row in tasks]
    assert all(row['model_calls'] == 1 and row['tool_calls'] == 0 for row in results)
    intervals = [value for row in results for event in row['trace']
                 if event['event'] == 'generation' for value in event['decode']['inter_token_seconds']]
    metrics = {
        'elapsed_seconds': finished - started,
        'model_calls': sum(row['model_calls'] for row in results),
        'output_tokens': sum(row['output_tokens'] for row in results),
        'truncated_calls': sum(event['truncated'] for row in results
                               for event in row['trace'] if event['event'] == 'generation'),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(backend.device),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(backend.device),
        'max_itl_ms': max(intervals) * 1000 if intervals else None,
        'intervals_over_100ms': sum(value >= .1 for value in intervals),
    }
    block = {'task_ids': [row['task_id'] for row in tasks], 'seed': seed,
             'results': results, 'metrics': metrics, 'started_utc': started_utc,
             'started_monotonic': started, 'finished_monotonic': finished,
             'answers_available_monotonic': finished}
    save(directory / 'complete.json', block)
    print(json.dumps({'directory': str(directory), 'tasks': len(tasks), **metrics}), flush=True)


def run(protocol, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.writer.lock').open('a') as writer:
        fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'protocol.json').exists():
            assert read(output / 'protocol.json') == protocol
        else:
            save(output / 'protocol.json', protocol)
        pending = [(index, tasks) for index, tasks in enumerate(blocks(protocol))
                   if completed(output / f'block-{index:04d}', [r['task_id'] for r in tasks],
                                protocol['native']['seed'] + index + 1) is None]
        if not pending:
            return
        session = output / f'session-{len(list(output.glob("session-*"))):04d}'
        session.mkdir()
        started = time.perf_counter()
        backend = Backend(protocol['native'])
        save(session / 'backend.json', {'backend': backend.metadata,
             'loading_seconds': time.perf_counter() - started,
             'pending_blocks': [index for index, tasks in pending]})
        execute(backend, protocol['warmup'], protocol, session / 'warmup', protocol['native']['seed'])
        for index, tasks in pending:
            execute(backend, tasks, protocol, output / f'block-{index:04d}',
                    protocol['native']['seed'] + index + 1)


def evaluate(protocol, output):
    assert read(output / 'protocol.json') == protocol
    measured = [completed(output / f'block-{index:04d}', [r['task_id'] for r in tasks],
                          protocol['native']['seed'] + index + 1)
                for index, tasks in enumerate(blocks(protocol))]
    assert all(block is not None for block in measured)
    results = [row for block in measured for row in block['results']]
    assert [row['task_id'] for row in results] == [row['task_id'] for row in protocol['tasks']]
    labels = {row['task_id']: row['labels']['q0'] for row in rows(protocol['settings']['labels'])}
    assert set(labels) == {row['task_id'] for row in results}
    summary = {'tasks': len(results), 'blocks': len(measured),
               'correct': sum(row['status'] == 'complete' and row['choice'] == labels[row['task_id']]
                              for row in results),
               'valid': sum(row['status'] == 'complete' for row in results),
               'failure_reasons': dict(Counter(row['error'] for row in results if row['status'] == 'failed')),
               'measurement_scope': 'Complete measured batches after disjoint development warmup. Includes generation and parsing, excludes startup and output persistence.',
               'budget': protocol['settings']['budget']}
    for key in ['elapsed_seconds', 'model_calls', 'output_tokens', 'truncated_calls', 'intervals_over_100ms']:
        summary[key] = sum(block['metrics'][key] for block in measured)
    for key in ['peak_allocated_bytes', 'peak_reserved_bytes']:
        summary[key] = max(block['metrics'][key] for block in measured)
    summary['max_itl_ms'] = max((block['metrics']['max_itl_ms'] for block in measured
                               if block['metrics']['max_itl_ms'] is not None), default=None)
    summary['accuracy'] = summary['correct'] / summary['tasks']
    save(output / 'evaluation.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'evaluate', 'validate'])
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.settings)
    protocol['agent'] = read(protocol['settings']['agent_config'])
    assert protocol['agent']['batch_size'] == protocol['native']['batch_size']
    assert protocol['agent']['seed'] == protocol['native']['seed']
    assert protocol['agent']['max_model_calls_per_task'] == 1
    assert protocol['agent']['max_new_tokens'] == protocol['agent']['max_output_tokens_per_task'] == 2048
    if args.mode == 'validate':
        print(json.dumps({'tasks': len(protocol['tasks']), 'warmup': len(protocol['warmup'])}))
        return
    assert args.output is not None
    {'run': run, 'evaluate': evaluate}[args.mode](protocol, args.output)


if __name__ == '__main__':
    main()
