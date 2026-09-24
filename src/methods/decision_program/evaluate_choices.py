import argparse
from collections import defaultdict
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--scope', choices=['development', 'heldout'], required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    protocol = json.loads((args.run / 'protocol.json').read_text())
    tasks, labels = {}, {}
    for source in inputs['sources']:
        selected = read_rows(source['tasks'])[source['offset']:source['offset'] + source['count']]
        tasks.update({row['task_id']: row for row in selected})
        labels.update({row['task_id']: row['labels'] for row in read_rows(source['labels'])})
    assert set(tasks) == set(protocol['job_ids'])
    phases = {}
    for phase in ['warmup', 'measured']:
        directory = args.run / phase
        compiled = json.loads((directory / 'compilation.json').read_text())
        reports = {}
        for mode in protocol['settings']['modes']:
            outputs = json.loads((directory / (mode + '-outputs.json')).read_text())
            groups = defaultdict(list)
            for identity, task in tasks.items():
                predictions = {row['id']: row['choice'] for row in outputs.get(identity, [])}
                for field_id, field in task['fields'].items():
                    answer = predictions.get(field_id)
                    valid = answer in [option['id'] for option in field['options']]
                    groups[task['dataset']].append({'task_id': identity, 'field': field_id,
                        'prediction': answer, 'valid': valid,
                        'correct': valid and answer == labels[identity][field_id]})
            reports[mode] = {name: {'fields': len(rows), 'valid': sum(row['valid'] for row in rows),
                'correct': sum(row['correct'] for row in rows),
                'accuracy': sum(row['correct'] for row in rows) / len(rows), 'rows': rows}
                for name, rows in groups.items()}
        phases[phase] = {'programs': compiled['programs'], 'failures': compiled['failures'],
                         'compiled_program_count': compiled['compiled_program_count'],
                         'quality': reports}
    report = {'scope': args.scope, 'phases': phases,
        'warm_measured_programs_equal': phases['warmup']['programs'] == phases['measured']['programs'],
        'warm_measured_quality_equal': phases['warmup']['quality'] == phases['measured']['quality']}
    (args.run / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({mode: {name: {k: v for k, v in scores.items() if k != 'rows'}
        for name, scores in datasets.items()} for mode, datasets in phases['measured']['quality'].items()}))


if __name__ == '__main__':
    main()
