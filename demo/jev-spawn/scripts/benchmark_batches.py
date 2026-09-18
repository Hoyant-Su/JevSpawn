"""Repeat exact completed batches with warm, counterbalanced method timings."""

import argparse
import json
import statistics
import time
from pathlib import Path

from benchmark_decisions import measure
from jev_spawn.backend import Backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'source-run', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('rank', 'warmups', 'repeats'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--batch-indices', nargs='+', type=int, required=True)
    parser.add_argument('--methods', nargs='+', required=True, choices=['streamed', 'independent', 'compact_json'])
    args = parser.parse_args()
    assert args.warmups > 0 and args.repeats > 0 and args.rank >= 0
    assert len(set(args.methods)) == len(args.methods) and len(set(args.batch_indices)) == len(args.batch_indices)
    config = json.loads(args.config.read_text())
    source = json.loads((args.source_run / f'run-{args.rank}.json').read_text())
    rows = {row['task_id']: row for row in map(json.loads, Path(source['config']['tasks_path']).read_text().splitlines())}
    cases = []
    for index in args.batch_indices:
        assert 0 <= index < len(source['batch_task_ids'])
        ids = source['batch_task_ids'][index]
        fields = [rows[key]['fields'] for key in ids] if source['fields'] is None else source['fields']
        cases.append({'batch_index': index, 'task_ids': ids, 'states': [rows[key]['state'] for key in ids], 'fields': fields})
    scope = ('Repeated warm timings of selected original batches on one model replica. Includes tokenization, synchronized '
             'model execution and parsing; excludes model load, garbage collection, warmups and file I/O. '
             'Not full-run latency or an extrapolated speedup. Allocator remains warm; memory is not a clean comparison.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = {'config': config, 'rank': args.rank, 'cases': cases, 'methods': args.methods,
           'warmups': args.warmups, 'repeats': args.repeats, 'scope': scope}
    (args.output_dir / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    trials = args.output_dir / 'trials.jsonl'
    previous = [json.loads(line) for line in trials.read_text().splitlines()] if trials.exists() else []
    completed = {(row['repeat'], row['batch_index'], row['method']): row for row in previous if row['repeat'] >= 0}
    backend = Backend(config)
    records = []
    for repeat in range(-args.warmups, args.repeats):
        case_order = list(range(len(cases)))
        offset = repeat % len(cases)
        case_order = case_order[offset:] + case_order[:offset]
        for position in case_order:
            case = cases[position]
            offset = (repeat + position) % len(args.methods)
            order = args.methods[offset:] + args.methods[:offset]
            for method in order:
                key = (repeat, case['batch_index'], method)
                if key in completed:
                    record = completed[key]
                else:
                    started_at = time.time()
                    record = measure(backend, case['states'], case['fields'], method, config['controller_max_new_tokens'], False)
                    record.update(repeat=repeat, batch_index=case['batch_index'], task_ids=case['task_ids'],
                                  method_order=order, case_order=[cases[p]['batch_index'] for p in case_order],
                                  started_at=started_at, ended_at=time.time(), warmup=repeat < 0)
                    with trials.open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                if repeat >= 0:
                    records.append(record)
                print(json.dumps({'repeat': repeat, 'batch': case['batch_index'], 'method': method,
                                  'seconds': record['elapsed_seconds']}), flush=True)
    summaries = []
    for case in cases:
        fields = case['fields']
        counts = [len(f) for f in fields] if isinstance(fields, list) else [len(fields)] * len(case['task_ids'])
        summaries.append({'batch_index': case['batch_index'], 'batch_size': len(case['task_ids']), 'field_counts': counts,
                          'methods': {method: {'median_seconds': statistics.median(r['elapsed_seconds'] for r in records
                                      if r['batch_index'] == case['batch_index'] and r['method'] == method)} for method in args.methods}})
    summary = {'scope': scope, 'backend': backend.metadata, 'cases': summaries,
               'methods': {method: {'pooled_batch_median_seconds': statistics.median(r['elapsed_seconds'] for r in records if r['method'] == method),
                                    'measured_batches': sum(r['method'] == method for r in records)} for method in args.methods},
               'limits': 'Selected batches have different shapes; pooled medians are descriptive.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['methods'], indent=2))


if __name__ == '__main__':
    main()
