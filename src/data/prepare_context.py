import argparse
import json
from pathlib import Path

from data.task_context import render, rows


def prepare(settings):
    records, sources = [], []
    for source in settings['sources']:
        for index, row in enumerate(rows(source)):
            context = render(row)
            records.append({'task_id': row['task_id'], 'context': context})
            sources.append({'task_id': row['task_id'], 'source': source, 'row_index': index})
    assert len({row['task_id'] for row in records}) == len(records)
    for name, values in [('output', records), ('provenance', sources)]:
        path = Path(settings[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(row, **settings['serialization']) + '\n' for row in values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()))
