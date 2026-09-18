"""Compare identical decision workloads, separating warm latency and GPU memory."""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from jev_spawn.backend import Backend
from jev_spawn.schema import DECISIONS


def measure(backend, observations, fields, method, tokens, clean_allocator):
    gc.collect()
    torch.cuda.synchronize()
    if clean_allocator:
        torch.cuda.empty_cache()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    generators = {'joint_json': backend.generate_fields, 'compact_json': backend.generate_array_fields}
    result = (generators[method](observations, fields, tokens) if method in generators
              else backend.score_fields(observations, fields, mode=method))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    memory = {'resident_allocated_bytes': resident,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'incremental_peak_bytes': torch.cuda.max_memory_allocated() - resident,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'clean_allocator': clean_allocator}
    return {'method': method, 'elapsed_seconds': elapsed, 'memory': memory, 'result': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source-run', type=Path)
    source.add_argument('--tasks', type=Path)
    parser.add_argument('--fields', type=Path)
    parser.add_argument('--methods', nargs='+', default=['shared', 'independent', 'joint_json'],
                        choices=['shared', 'streamed', 'independent', 'joint_json', 'compact_json'])
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--batch-count', type=int, required=True)
    parser.add_argument('--warmups', type=int, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.tasks:
        states = [json.loads(line) for line in args.tasks.read_text().splitlines()]
        observations = [state['state'] for state in states]
        fields = json.loads(args.fields.read_text()) if args.fields else [state['fields'] for state in states]
    else:
        states = [state for p in sorted(args.source_run.glob('chunk-*.json'))
                  for state in json.loads(p.read_text())['states']]
        observations = [json.dumps({'specification': s['prompt'], 'implementation': s['initial_solution']},
                                   ensure_ascii=False) for s in states]
        fields = {name: DECISIONS[name] for name in ('review', 'review_focus')}
    states = states[:args.batch_size * args.batch_count]
    observations = observations[:len(states)]
    if isinstance(fields, list):
        fields = fields[:len(states)]
    assert len(states) == args.batch_size * args.batch_count
    methods = args.methods
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {'config': config, 'task_ids': [s['task_id'] for s in states], 'fields': fields,
           'batch_size': args.batch_size, 'batch_count': args.batch_count,
           'repeats': args.repeats, 'warmups': args.warmups, 'methods': methods}
    (args.output_dir / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    trials = args.output_dir/'trials.jsonl'
    previous = [json.loads(line) for line in trials.read_text().splitlines()] if trials.exists() else []
    completed = {(r['sweep'], r.get('repeat'), r['batch'], r['method']): r for r in previous}
    backend = Backend(config)
    records = []
    for repeat in range(-args.warmups, args.repeats):
        for batch in range(args.batch_count):
            selected = observations[batch * args.batch_size:(batch + 1) * args.batch_size]
            selected_fields = fields[batch * args.batch_size:(batch + 1) * args.batch_size] if isinstance(fields, list) else fields
            offset = (repeat + batch) % len(methods)
            order = methods[offset:] + methods[:offset]
            for method in order:
                key = ('latency', repeat, batch, method)
                if key in completed and repeat >= 0:
                    record = completed[key]
                else:
                    record = measure(backend, selected, selected_fields, method, config['controller_max_new_tokens'], False)
                    record.update(repeat=repeat, batch=batch, order=order, sweep='latency')
                    with trials.open('a') as stream:
                        stream.write(json.dumps(record)+'\n')
                if repeat >= 0:
                    records.append(record)
                print(json.dumps({'repeat': repeat, 'batch': batch, 'method': method,
                                  'seconds': record['elapsed_seconds']}), flush=True)
    memory_records = []
    for batch in range(args.batch_count):
        selected = observations[batch * args.batch_size:(batch + 1) * args.batch_size]
        selected_fields = fields[batch * args.batch_size:(batch + 1) * args.batch_size] if isinstance(fields, list) else fields
        for method in methods:
            key = ('memory', None, batch, method)
            if key in completed:
                record = completed[key]
            else:
                record = measure(backend, selected, selected_fields, method, config['controller_max_new_tokens'], True)
                record.update(batch=batch, sweep='memory')
                with trials.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
            memory_records.append(record)
    summary = {'batch_size': args.batch_size, 'unique_states': len(states),
               'fields_per_state': [len(schema) for schema in fields] if isinstance(fields, list) else [len(fields)] * len(states),
               'backend': backend.metadata, 'methods': {},
               'limits': 'JSON sees the fields jointly and may choose different answers. Peak allocated bytes cover live PyTorch allocations; reserved bytes are allocator holdings, not model size. Do not extrapolate controller timings to a complete coding workflow.'}
    for method in methods:
        times = [r['elapsed_seconds'] for r in records if r['method'] == method]
        memory = [r['memory'] for r in memory_records if r['method'] == method]
        summary['methods'][method] = {'median_seconds': statistics.median(times),
                                      'measured_batches': len(times), 'memory': memory}
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary['methods'], indent=2))


if __name__ == '__main__':
    main()
