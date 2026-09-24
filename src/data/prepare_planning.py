import argparse
import json
import random
from pathlib import Path


def keyed_records(source, specification):
    return source.items()


def listed_records(source, specification):
    return [(row[specification['identity_field']], row)
            for row in source[specification['records_field']]]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def prepare(specification, settings):
    source = json.loads(Path(specification['source']).read_text())
    loaders = {'mapping': keyed_records, 'list': listed_records}
    records = list(loaders[specification['layout']](source, specification))
    assert len(records) == specification['source_count']
    assert len(dict(records)) == len(records)
    random.Random(settings['seed']).shuffle(records)
    assert sum(settings['splits'].values()) <= len(records)
    remaining = iter(records)
    output = Path(specification['output'])
    splits = {}
    for split, count in settings['splits'].items():
        chosen = [next(remaining) for _ in range(count)]
        tasks = [{'task_id': specification['identity_template'].format(identity=identity),
                  'query': row[specification['input_field']]}
                 for identity, row in chosen]
        labels = [{'task_id': task['task_id'], 'source_id': identity,
                   **{key: row[field] for key, field in specification['label_fields'].items()}}
                  for task, (identity, row) in zip(tasks, chosen, strict=True)]
        write_rows(output / split / 'tasks.jsonl', tasks)
        write_rows(output / split / 'labels.jsonl', labels)
        splits[split] = {'count': count, 'task_ids': [task['task_id'] for task in tasks],
                         'tasks': str(output / split / 'tasks.jsonl'),
                         'labels': str(output / split / 'labels.jsonl')}
    result = {'dataset': specification['dataset'], 'kind': specification['kind'],
              'seed': settings['seed'], 'source': specification['source'],
              'source_count': len(records), 'source_urls': specification['source_urls'],
              'input_field': specification['input_field'], 'splits': splits,
              'evaluation': specification['evaluation']}
    (output / 'manifest.json').write_text(json.dumps(result, indent=settings['indent']) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    arguments = parser.parse_args()
    settings = json.loads(arguments.config.read_text())
    reports = [prepare(specification, settings) for specification in settings['datasets']]
    report = Path(settings['report'])
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(reports, indent=settings['indent']) + '\n')


if __name__ == '__main__':
    main()
