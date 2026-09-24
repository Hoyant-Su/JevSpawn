import argparse
import json
from pathlib import Path

from jev_spawn.infra.prompts import resolve_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    settings = resolve_prompts(json.loads(args.settings.read_text()))
    contract = resolve_prompts(json.loads(Path(settings['contract']).read_text()))
    jobs = []
    for source in settings['sources']:
        rows = [json.loads(line) for line in Path(source['tasks']).read_text().splitlines()]
        selected = rows[source['offset']:source['offset'] + source['count']]
        assert len(selected) == source['count']
        jobs.extend({**contract, 'task_id': task['task_id'],
                     'shared_input': {'context': task['state']},
                     'items': [{'id': identity, 'input': {'question': field['question'],
                                                        'options': field['options']}}
                               for identity, field in task['fields'].items()]}
                    for task in selected)
    assert len({job['task_id'] for job in jobs}) == len(jobs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(job, ensure_ascii=False) + '\n' for job in jobs))


if __name__ == '__main__':
    main()
