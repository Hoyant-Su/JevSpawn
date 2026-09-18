"""Report measured quality, cost and paired draft-to-final changes for completed policies."""

import argparse
import json
from pathlib import Path


def verdicts(path):
    return {row['task_id']: row['status'] == 'passed'
            for row in (json.loads(line) for line in path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policies, task_sets, drafts = [], [], []
    for directory in args.run_dirs:
        run = json.loads((directory / 'summary.json').read_text())
        final = verdicts(directory / 'evaluation.jsonl')
        initial = verdicts(directory / 'initial_evaluation.jsonl')
        drafts.append({row['task_id']: row['solution'] for row in
                       (json.loads(line) for line in (directory / 'initial_solutions.jsonl').read_text().splitlines())})
        assert final.keys() == initial.keys() == drafts[-1].keys()
        task_sets.append(set(final))
        correct = sum(final.values())
        policies.append({
            'run_id': run['run_id'], 'policy': run['spawn_policy'],
            'routing_mode': run.get('routing_mode', 'model'),
            'tasks': len(final), 'passed': correct, 'pass_rate': correct / len(final),
            'initial_passed': sum(initial.values()),
            'fixed': sum(final[key] and not initial[key] for key in final),
            'regressed': sum(initial[key] and not final[key] for key in final),
            'elapsed_seconds': run['elapsed_seconds'],
            'correct_tasks_per_second': correct / run['elapsed_seconds'],
            'agents_spawned': run['agents_spawned'], 'additional_agents': run['additional_agents'],
            'worker_output_tokens': run['worker_output_tokens'],
            'controller_computed_input_tokens': run['controller_computed_input_tokens'],
            'truncated_worker_outputs': run['truncated_worker_outputs'],
            'decision_counts': run['decision_counts'],
            'splits': json.loads((directory / 'evaluation.splits.json').read_text()),
        })
    assert all(tasks == task_sets[0] for tasks in task_sets)
    result = {'policies': policies,
              'identical_initial_solutions': all(draft == drafts[0] for draft in drafts),
              'interpretation': 'Descriptive measured runs. Different policies change real work and batch shapes. Timing includes compilation encountered during each run and excludes model loading. No universal speedup or calibrated-confidence claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'splits'} for row in policies], indent=2))


if __name__ == '__main__':
    main()
