import argparse
import json
import random
from pathlib import Path

from jev_spawn.infra.prompts import load_prompt
from project_paths import ROOT


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def prepare(settings):
    instruction = load_prompt(settings['prompt'])['instruction']
    reports = []
    for specification in settings['datasets']:
        original = read_rows(ROOT / specification['tasks'])
        gold = {row['id']: row for row in read_rows(ROOT / specification['labels'])}
        assert len(original) == len(gold) == specification['source_count']
        assert {row['id'] for row in original} == gold.keys()
        random.Random(settings['seed']).shuffle(original)
        assert sum(settings['splits'].values()) == len(original)
        remaining = iter(original)
        output = ROOT / specification['output']
        splits = {}
        for split, count in settings['splits'].items():
            selected = [next(remaining) for _ in range(count)]
            tasks = [{'task_id': specification['dataset'] + '/' + row['id'],
                      'dataset': specification['dataset'], 'kind': 'completion',
                      'instruction': instruction,
                      'input': {'question': row['question'], 'function': row['function']},
                      'answer_schema': settings['answer_schema'], 'source': row}
                     for row in selected]
            labels = [{'task_id': task['task_id'], 'source_id': row['id'],
                       'category': specification['category'],
                       'ground_truth': gold[row['id']]['ground_truth']}
                      for task, row in zip(tasks, selected, strict=True)]
            task_path, label_path = output / split / 'tasks.jsonl', output / split / 'labels.jsonl'
            write_rows(task_path, tasks)
            write_rows(label_path, labels)
            splits[split] = {'tasks': str(task_path), 'labels': str(label_path), 'count': count,
                             'task_ids': [task['task_id'] for task in tasks]}
        reports.append({'dataset': specification['dataset'], 'category': specification['category'],
                        'source_count': len(original), 'seed': settings['seed'], 'splits': splits})
    report = ROOT / settings['report']
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(reports, indent=settings['indent']) + '\n')
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    print(json.dumps(prepare(settings), indent=settings['indent']))


if __name__ == '__main__':
    main()
