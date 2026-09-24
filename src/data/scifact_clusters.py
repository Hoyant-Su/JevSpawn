"""Preserve shared-claim and shared-document dependence during resampling."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.labels.read_text().splitlines()]
    adjacency = defaultdict(set)
    by_claim = defaultdict(list)
    for row in rows:
        for question_id in row['question_ids'].values():
            claim_id, document_id = question_id.split('/')
            by_claim[claim_id].append(row['task_id'])
    for documents in by_claim.values():
        for document in documents:
            adjacency[document].update(documents)
    clusters = {}
    for task in sorted(adjacency):
        if task in clusters:
            continue
        pending = [task]
        while pending:
            current = pending.pop()
            if current in clusters:
                continue
            clusters[current] = task
            pending.extend(adjacency[current] - clusters.keys())
    result = {'unit': 'Connected component of the original claim-document graph',
              'clusters': clusters, 'count': len(set(clusters.values()))}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'documents': len(clusters), 'clusters': result['count']}))


if __name__ == '__main__':
    main()
