import argparse
import json
from pathlib import Path


def records(path):
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return value if isinstance(value, list) else [value]


def resolve(root, path):
    return root / path


def validate(manifest, root):
    reports = []
    for dataset in manifest['datasets']:
        task_path = resolve(root, Path(dataset['tasks']))
        label_path = resolve(root, Path(dataset['labels']))
        assert task_path.is_file(), task_path
        assert label_path.is_file(), label_path
        tasks = records(task_path)
        labels = records(label_path)
        required = set(dataset['task_fields'])
        assert all(required <= set(task) for task in tasks), dataset['id']
        assert len(tasks) > 0 and len(labels) > 0, dataset['id']
        evaluator = dataset['evaluator']
        assert evaluator is None or resolve(root, Path(evaluator)).is_file(), dataset['id']
        reports.append({'id': dataset['id'], 'status': dataset['status'],
                        'tasks': len(tasks), 'labels': len(labels),
                        'evaluator': evaluator})
    return {'datasets': reports, 'network_access': manifest['network_access'],
            'seed': manifest['seed']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text())
    report = validate(manifest, arguments.root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
