import argparse
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path
from statistics import mean

from analysis.semantic_latency import distribution
from analysis.session_timing import collect_timing


def read(path):
    return json.loads(Path(path).read_text())


def analyze(settings):
    run = Path(settings['run'])
    contract = read(run / settings['protocol_file'])
    tasks = contract['tasks']
    completion = read(run / settings['completion_file'])
    identities = [task['task_id'] for task in tasks]
    assert completion['task_ids'] == identities
    assert completion['tasks'] == len(tasks) == contract['specification']['task_count']
    results = [read(run / settings['task_file'].format(index=index)) for index, _ in enumerate(tasks)]
    assert [result['task_id'] for result in results] == identities
    module, name = settings['scorer']['class'].rsplit('.', maxsplit=1)
    scorer = getattr(importlib.import_module(module), name)(read(settings['scorer']['settings']))
    references = settings['references']
    targets = {row['task_id']: row for row in
               [json.loads(line) for line in Path(references).read_text().splitlines()]}
    assert set(targets) == set(identities)
    quality = [scorer.score(task, result, targets[task['task_id']])
               for task, result in zip(tasks, results, strict=True)]
    timing = collect_timing({'run': str(run), 'task_ids': identities,
                            'method': contract['specification']['method'],
                            'dataset': sorted({task['dataset'] for task in tasks})}, settings['timing'])
    assert timing['metrics']['timing_complete']
    groups = defaultdict(list)
    for path in sorted(run.glob(settings['batch_glob'])):
        for batch in read(path):
            groups[(batch.get('operation', settings['text_operation']), batch['batch_size'])].append(batch)
    finite = []
    for (operation, size), batches in groups.items():
        if operation == settings['finite_operation']:
            finite.append({'batch_size': size, 'batches': len(batches),
                'batch_service_latency': distribution([row['elapsed_seconds'] for row in batches]),
                'components': {name: distribution([row['structured']['timings'][name] for row in batches])
                               for name in settings['finite_components']}})
    decisions, intervals = [], []
    for result in results:
        calls = result.get('calls', [])
        native = [call['decision'] for call in calls if 'decision' in call
                  and 'delivered_monotonic' in call['decision']]
        decisions.extend(row['delivered_monotonic'] - row['submitted_monotonic'] for row in native)
        intervals.extend(child['delivered_monotonic'] - parent['delivered_monotonic']
                         for parent, child in zip(native, native[settings['next_index']:]))
    return {'tasks': len(tasks), 'quality': {'mean_score': mean(row['score'] for row in quality),
            'details': quality}, 'status_counts': dict(Counter(row['status'] for row in results)),
            'task_latency': distribution([row['elapsed_seconds'] for row in results]),
            'timing': timing, 'finite_batches': finite,
            'finite_request_to_delivery': distribution(decisions),
            'sequential_inter_decision': distribution(intervals),
            'metric_definitions': settings['metric_definitions']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    args = parser.parse_args()
    settings = read(args.settings)
    report = analyze(settings)
    output = Path(settings['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=settings['indent']) + '\n')
    print(json.dumps({'tasks': report['tasks'], 'mean_score': report['quality']['mean_score']}))


if __name__ == '__main__':
    main()
