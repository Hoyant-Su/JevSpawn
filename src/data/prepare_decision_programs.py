import argparse
import json
from pathlib import Path

from jev_spawn.infra.prompts import resolve_prompts


def value_at(record, path):
    for key in path:
        record = record[key]
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = resolve_prompts(json.loads(args.settings.read_text()))
    rows = [json.loads(line) for line in Path(settings['source']).read_text().splitlines()]
    jobs = []
    for row in rows:
        items = value_at(row, settings['items_path'])
        jobs.append({
            'task_id': value_at(row, settings['task_id_path']),
            'instruction': settings['instruction'],
            'shared_input': {name: value_at(row, path) for name, path in settings['shared_fields'].items()},
            'shared_interface': settings['shared_interface'],
            'item_interface': settings['item_interface'],
            'items': [{'id': value_at(item, settings['item_id_path']),
                       'input': {name: value_at(item, path) for name, path in settings['item_fields'].items()}}
                      for item in items],
            'output_contract': settings['output_contract']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(job, ensure_ascii=False) + '\n' for job in jobs))


if __name__ == '__main__':
    main()
