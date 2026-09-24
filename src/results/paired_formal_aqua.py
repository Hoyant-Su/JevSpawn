import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replicates', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--confidence', type=float, required=True)
    args = parser.parse_args()
    table = read(ROOT / 'manuscript/tables/published_aqua.json')['records']
    label_path = Path(read(ROOT / 'runs/official-react-aqua-test-001/protocol.json')['settings']['labels'])
    labels = {row['task_id']: row['labels']['q0']
              for row in map(json.loads, label_path.read_text().splitlines())}
    ids = list(labels)
    scores = {}
    for row in table:
        directory = (ROOT / row['source']).parent
        if row['method'] == 'LatentMAS':
            predictions = [r for path in sorted(directory.glob('block-*.json')) for r in read(path)['predictions']]
        elif row['method'] == 'AgentPrune':
            predictions = [r for path in sorted(directory.glob('block-*/complete.json'))
                           for r in read(path)['result']['records']]
        else:
            predictions = [r for path in sorted(directory.glob('block-*/complete.json')) for r in read(path)['results']]
        answers = {r['task_id']: r.get('answer') for r in predictions}
        assert len(predictions) == len(answers) == row['tasks'] and answers.keys() == labels.keys()
        scores[row['method']] = np.array([answers[i] == labels[i] for i in ids], dtype=np.int8)
        assert int(scores[row['method']].sum()) == row['correct']
    indices = np.random.default_rng(args.seed).integers(len(ids), size=(args.replicates, len(ids)))
    tail = (1 - args.confidence) / 2
    pairs = []
    for left, right in combinations(scores, 2):
        delta = scores[left] - scores[right]
        pairs.append({'left': left, 'right': right, 'difference_percentage_points': float(delta.mean() * 100),
                      'paired_interval_percentage_points': (np.quantile(delta[indices].mean(axis=1),
                                                                       [tail, 1-tail]) * 100).tolist(),
                      'left_only_correct': int((delta == 1).sum()),
                      'right_only_correct': int((delta == -1).sum())})
    result = {'settings': vars(args), 'tasks': len(ids), 'task_ids': ids, 'pairs': pairs,
              'methods': {method: {'accuracy_percentage': float(values.mean() * 100),
                                   'interval_percentage': (np.quantile(values[indices].mean(axis=1),
                                                                        [tail, 1-tail]) * 100).tolist()}
                          for method, values in scores.items()},
              'scope': 'Paired question bootstrap of the complete fixed AQuA test split. Unadjusted descriptive intervals across method pairs. Different intrinsic algorithm budgets are retained. These comparisons do not identify equal-compute effects.'}
    (ROOT / 'results/paired_formal_aqua.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
