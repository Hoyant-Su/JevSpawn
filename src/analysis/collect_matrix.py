import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def collect(config):
    manifest = json.loads(Path(config['manifest']).read_text())
    stage = json.loads(Path(config['stage']).read_text())
    declared = {(method, entry['dataset']): entry for method, entries in manifest['methods'].items()
                for entry in entries}
    assert {(cell['method'], cell['dataset']) for cell in stage['cells']} == set(declared)
    cells, completed = [], []
    for source in stage['cells']:
        method, dataset = source['method'], source['dataset']
        assert source['specification'] == declared[(method, dataset)]['specification']
        specification = json.loads(Path(source['specification']).read_text())
        assert source['task_count'] == specification['task_count']
        path = Path(source['evaluation'])
        cell = dict(source, evaluated=path.exists())
        cells.append(cell)
        if cell['evaluated']:
            result = json.loads(path.read_text())
            metrics = result['datasets'][dataset]
            assert result['stage'] == specification['stage']
            assert metrics['tasks'] == result['tasks'] == specification['task_count']
            assert len(result['scores']) == len({row['task_id'] for row in result['scores']}) == result['tasks']
            assert all(row['dataset'] == dataset for row in result['scores'])
            cell['metrics'] = {key: value for key, value in metrics.items()
                               if value is None or isinstance(value, (str, int, float, bool))}
            cell['status_counts'] = metrics['status_counts']
            completed.append({key: cell[key] for key in config['identity_columns']} | cell['metrics'])
    report = {'updated_at': datetime.now(timezone.utc).isoformat(), 'manifest': config['manifest'],
              'completed_cells': len(completed), 'declared_cells': len(cells), 'cells': cells}
    Path(config['json_output']).write_text(json.dumps(report, indent=2) + '\n')
    columns = config['identity_columns'] + sorted({key for row in completed for key in row} - set(config['identity_columns']))
    with Path(config['csv_output']).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(completed)
    print(json.dumps({'completed_cells': len(completed), 'declared_cells': len(cells)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    collect(json.loads(args.config.read_text()))
