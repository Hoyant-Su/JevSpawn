import argparse
import json
from pathlib import Path

from baselines.common.evaluate import score_task, summarize, save
from data.task_context import rows
from jev_spawn.infra.configuration import resolve_symbol


def read(path):
    return json.loads(Path(path).read_text())


def evaluate(run):
    contract = read(run / 'protocol.json')
    scores = []
    for index, entry in enumerate(contract['entries']):
        source = read(entry['source_experiment_specification'])
        task, = [task for task in rows(source['tasks']) if task['task_id'] == entry['task_id']]
        definition = source['environment_execution']
        score = score_task(task, run / f'task-{index:05d}.json', resolve_symbol(definition['class']),
            definition['parameters'], read(source['environment']), run / 'evaluation' / f'task-{index:05d}')
        scores.append(score)
    report = {'scope': contract['scope'], 'run': str(run), 'summary': summarize(scores), 'scores': scores}
    save(run / 'evaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = [evaluate(run) for run in args.run]
    save(args.output, {'runs': reports})
    print(json.dumps([report['summary'] for report in reports], indent=2))


if __name__ == '__main__':
    main()
