"""Select original paragraphs and cyclically reorder existing candidate lists."""

import argparse
import copy
import json
from pathlib import Path
import random


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--squad-source', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--squad-count', type=int, required=True)
    parser.add_argument('--ablation-count', type=int, required=True)
    args = parser.parse_args()
    tasks = read(args.squad_source / 'tasks.jsonl')
    random.Random(args.seed).shuffle(tasks)
    selected = tasks[:args.squad_count]
    labels = {row['task_id']: row for row in read(args.squad_source / 'evaluation/labels.jsonl')}
    gold = [dict(labels[row['task_id']], labels={key: 'yes' if value else 'no'
            for key, value in labels[row['task_id']]['labels'].items()}) for row in selected]
    directory = args.data / 'squad2_answerability'
    write(directory / 'evaluation/tasks.jsonl', selected)
    write(directory / 'evaluation/labels.jsonl', gold)
    write(directory / 'ablation/tasks.jsonl', selected[:args.ablation_count])
    write(directory / 'ablation/labels.jsonl', gold[:args.ablation_count])
    for dataset in ['aqua', 'race_middle', 'pubmedqa', 'scifact_cited', 'squad2_answerability']:
        source = args.data / dataset / 'evaluation'
        rows = copy.deepcopy(read(source / 'tasks.jsonl'))
        for row in rows:
            for field in row['fields'].values():
                options = field['options']
                field['options'] = options[1:] + options[:1]
        write(args.data / dataset / 'rotated/tasks.jsonl', rows)
        write(args.data / dataset / 'rotated/labels.jsonl', read(source / 'labels.jsonl'))
    manifest = {'seed': args.seed, 'source': str(args.squad_source),
                'selection': 'Seeded shuffle of original paragraphs; retain every original question.',
                'tasks': len(selected), 'decisions': sum(len(row['fields']) for row in selected),
                'ablation_tasks': args.ablation_count,
                'rotation': 'One cyclic left shift per candidate list; semantic option IDs and gold IDs unchanged.'}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
