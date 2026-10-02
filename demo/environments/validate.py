import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path

from demo.environments.factory import create
from demo.environments.common import ROOT


def validate(path):
    sample = json.loads(path.read_text())
    root = create(sample, path.parent, float('inf'))
    states = {'root': root}
    action_count = 0
    for turn in sample['rounds']:
        for node, steps in turn.get('children', {}).items():
            parent, = [step['parent_id'] for step in steps if 'parent_id' in step]
            environment = states[parent].fork()
            for step in steps:
                for expected in step.get('observations', []):
                    action = expected['action']
                    actual = environment.observe(action['tool'], deepcopy(action['arguments']))
                    assert actual == expected['value'], (sample['id'], node, actual, expected['value'])
                    action_count += 1
            states[node] = environment
    score = float(root.evaluate(sample['answer']))
    assert score == sample['score'], (sample['id'], score, sample['score'])
    return {'id': sample['id'], 'task_id': sample['task_id'], 'score': score,
            'rounds': len(sample['rounds']), 'branch_actions': action_count,
            'initial_context_matches': True, 'all_observations_match': True}


def main():
    parser = argparse.ArgumentParser(description='Reexecute every recorded branch in native CPU environments.')
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads((ROOT / 'sources.json').read_text())['validation']
    with ProcessPoolExecutor(configuration['workers']) as pool:
        results = list(pool.map(validate, sorted(arguments.data.glob('*.json'))))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
