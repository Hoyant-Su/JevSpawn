import argparse
import json
import shutil
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ['run', 'labels', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    (args.output / 'trials').mkdir(parents=True, exist_ok=True)
    for path in (args.run / 'trials').glob('*.json'):
        shutil.copyfile(path, args.output / 'trials' / path.name)
    for path in (args.run / 'retrieval').glob('*.json'):
        source = json.loads(path.read_text())
        compact = {k: source[k] for k in ['task_ids', 'status', 'batch', 'rank',
                   'component_elapsed_seconds', 'reconstructed_retrieval']}
        compact['reconstruction_seconds'] = source['reconstruction_seconds'] if source['reconstructed_retrieval'] else None
        compact['retrieval'] = {'elapsed_seconds': source['retrieval']['elapsed_seconds']}
        compact['generation'] = [{k: row[k] for k in ['elapsed_seconds', 'input_tokens', 'output_tokens', 'decode']}
                                 for row in source['generation']]
        save(args.output / 'retrieval' / path.name, compact)
    paths = list(args.run.glob('retrieve-rank-*/run.json')) + list((args.run / 'recovery').glob('initial-rank-*-run.json'))
    for path in paths:
        source = json.loads(path.read_text())
        save(args.output / path.relative_to(args.run),
             {'embedding_load_and_index_seconds': source['embedding_load_and_index_seconds']})
    shutil.copyfile(args.labels, args.output / 'labels.jsonl')
    failures = json.loads((args.run / 'qualification.json').read_text())['failures']
    save(args.output / 'retrieval_failures.json', failures)
    print(json.dumps({'output': str(args.output), 'trials': len(list((args.output / 'trials').glob('*.json')))}))


if __name__ == '__main__':
    main()
