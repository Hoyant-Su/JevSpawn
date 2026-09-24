import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    parser.add_argument('--task-count', type=int, required=True)
    args = parser.parse_args()
    methods = ['single', 'all', 'adaptive_direct', 'adaptive_json']
    collected, reference = {}, None
    for method in methods:
        rows = []
        reference_solutions, reference_outcomes = None, None
        for repeat in range(args.repeats):
            directory = args.run / f'repeat-{repeat:02d}-{method}'
            metrics = json.loads((directory / 'metrics.json').read_text())
            states = json.loads((directory / 'states.json').read_text())
            tests = [json.loads(line) for line in (directory / 'evaluation.jsonl').read_text().splitlines()]
            assert metrics['examples'] == len(states) == len(tests) == args.task_count
            assert metrics['eligible_whole_workload_timing']
            assert [s['task_id'] for s in states] == [t['task_id'] for t in tests]
            drafts = [(s['task_id'], s['initial_solution']) for s in states]
            solutions = [(s['task_id'], s['solution']) for s in states]
            outcomes = [(t['task_id'], t['status']) for t in tests]
            if reference_solutions is None:
                reference_solutions, reference_outcomes = solutions, outcomes
            if reference is None:
                reference = drafts
            draft_matches = sum(a == b for a, b in zip(reference, drafts))
            rows.append({**metrics, 'repeat': repeat, 'passed': sum(t['status'] == 'passed' for t in tests),
                         'source_directory': str(directory.resolve()),
                         'timing_block': json.loads((directory / 'run.json').read_text())['config'].get('timing_block'),
                         'final_solution_matches_first_repeat': sum(a == b for a, b in zip(reference_solutions, solutions)),
                         'test_outcome_matches_first_repeat': sum(a == b for a, b in zip(reference_outcomes, outcomes)),
                         'status_counts': dict(Counter(t['status'] for t in tests)),
                         'initial_draft_matches_reference': draft_matches,
                         'roles': dict(Counter(u['role'] for s in states for u in s['usage']))})
        times = [r['elapsed_seconds'] for r in rows]
        collected[method] = {
            'elapsed_mean_seconds': statistics.mean(times),
            'elapsed_stdev_seconds': statistics.stdev(times) if len(times) > 1 else None,
            'passed_each_repeat': [r['passed'] for r in rows],
            'max_itl_ms': max(r['itl_max_ms'] for r in rows),
            'latency_qualified': all(r['itl_max_ms'] < 100 for r in rows),
            'identical_initial_drafts': all(r['initial_draft_matches_reference'] == args.task_count for r in rows),
            'repeats': rows,
        }
    report = {'tasks': args.task_count, 'timing_repeats': args.repeats,
              'scope': 'Frozen-backbone complete coding workflows. Each repeat reruns initial generation and every selected controller and worker call. Repeats do not multiply the number of distinct evaluation tasks.',
              'evaluation': 'Unmodified read-only Bubblewrap evaluator with original held-out assertions.',
              'methods': collected}
    (args.run / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({m: {k: v for k, v in row.items() if k != 'repeats'}
                      for m, row in collected.items()}, indent=2))


if __name__ == '__main__':
    main()
