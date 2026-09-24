import argparse
import csv
import json
from pathlib import Path

from baselines.common.persistence import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    matrix = json.loads(Path(settings['matrix']).read_text())
    records = []
    for cell in matrix['cells']:
        path = Path(cell['evaluation'])
        if not path.exists():
            continue
        report, = json.loads(path.read_text())['runs']
        specification = json.loads(Path(cell['specification']).read_text())
        assert report['stage'] == matrix['stage']
        assert Path(report['run']).resolve() == Path(cell['run']).resolve()
        assert report['task_source'] == specification['tasks']
        assert report['method'] == specification['method']
        assert report['environment_execution'] == specification['environment_execution']
        summary = report['summary']
        assert summary['declared_tasks'] == cell['tasks']
        if summary['artifacts'] != summary['declared_tasks']:
            continue
        tasks = [json.loads(line) for line in Path(specification['tasks']).read_text().splitlines()]
        assert [row['task_id'] for row in report['scores']] == [row['task_id'] for row in tasks]
        records.append({'method': cell['method'], 'dataset': cell['dataset'],
                        'evaluation': cell['evaluation'], **summary})
    save(settings['summary'], {'matrix': settings['matrix'], 'completed_cells': records})
    datasets = list(dict.fromkeys(cell['dataset'] for cell in matrix['cells']))
    lookup = {(row['method'], row['dataset']): row for row in records}
    for table in settings['tables']:
        path = Path(table['output'])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['method', *datasets])
            for method in matrix['methods']:
                writer.writerow([method, *[
                    lookup[method, dataset][table['metric']] if (method, dataset) in lookup
                    else None for dataset in datasets]])
    print(json.dumps({'completed_cells': len(records), 'declared_cells': len(matrix['cells'])}))


if __name__ == '__main__':
    main()
