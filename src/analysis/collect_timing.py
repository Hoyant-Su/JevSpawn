import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
import json
from pathlib import Path

from session_timing import collect_timing


def main(config):
    matrix = json.loads(Path(config['matrix']).read_text())
    cells = []
    for cell in matrix['cells']:
        if cell['evaluated']:
            evaluation = json.loads(Path(cell['evaluation']).read_text())
            cells.append(dict(cell, task_ids=[row['task_id'] for row in evaluation['scores']]))
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        results = list(pool.map(partial(collect_timing, settings=config['timing']), cells))
    shared = {json.dumps(session['shared_config'], sort_keys=True) for result in results
              for session in result['sessions'] if not session['missing_telemetry']}
    assert len(shared) == 1, 'Completed sessions used different shared inference configurations.'
    common, = shared
    Path(config['json_output']).write_text(json.dumps({'config': config,
        'shared_config': json.loads(common), 'cells': results}, indent=2) + '\n')
    rows = [dict(method=result['method'], dataset=result['dataset'], **result['metrics']) for result in results]
    columns = config['identity_columns'] + sorted({key for row in rows for key in row} - set(config['identity_columns']))
    with Path(config['csv_output']).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'cells': len(results), 'complete_timing': sum(r['metrics']['timing_complete'] for r in results)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    main(json.loads(args.config.read_text()))
