import argparse
import json
from pathlib import Path


def prepare(configuration):
    for source in configuration['sources']:
        records = json.loads(Path(source['path']).read_text())
        schema = json.loads(Path(source['environment']).read_text())['answer_schema']
        tasks = [{'task_id': source['task_id'].format(index=index),
                  'dataset': source['dataset'], 'answer_schema': schema,
                  'source': {name: records[index][name] for name in source['public_fields']}}
                 for index in source['indices']]
        output = Path(source['output'])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(''.join(json.dumps(task, ensure_ascii=False) + '\n' for task in tasks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.configuration.read_text()))
