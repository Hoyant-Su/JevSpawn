import argparse
import ast
from collections import Counter
from itertools import product
import json
from pathlib import Path
import re

from data.evaluate_zebralogic import score_rows
from methods.finite_constraints.soft import soft_table_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, required=True)
    args = parser.parse_args()
    settings = json.loads((args.source / 'protocol.json').read_text())['settings']
    tasks = [json.loads(line) for line in Path(settings['tasks']).read_text().splitlines()]
    assert len(tasks) == settings['task_count']
    source_path = Path(__file__).with_name('program.py')
    helpers = dict(globals())
    names = {'parse_puzzle', 'domains', 'assignment_rows'}
    definitions = [node for node in ast.parse(source_path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == names
    # Reuse the unchanged source-format functions without importing the CUDA backend.
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), 'exec'), helpers)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'source': str(args.source), 'phase': 'measured', 'method': 'finite_shared',
        'time_limit_seconds_per_puzzle': args.time_limit,
        'task_ids': [row['task_id'] for row in tasks],
        'objective': 'Maximize the sum of log original p(satisfied) over one selected tuple per saved factor.',
        'constraints': 'Original finite variable domains and all-different within each original attribute.',
        'scope': 'CPU counterfactual diagnostic on frozen probability tables. No model call, scope repair, calibration, or conditional-spawn speed claim.',
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    outputs = []
    for task in tasks:
        saved = json.loads((args.source / 'measured' / f'{task["task_id"]}-finite_shared.json').read_text())
        record = {'task_id': task['task_id'], 'source_status': saved['status'], 'prediction_rows': None}
        if 'execution' not in saved:
            record.update(status='source_failure', error=saved['error'])
        else:
            state = helpers['parse_puzzle'](task['puzzle'])
            domain = helpers['domains'](state)
            decisions = saved['execution']['result']['groups'][0]
            factors, offset = [], 0
            for index, scope in enumerate(saved['scopes']):
                tuples = list(product(*(domain[variable] for variable in scope)))
                rows = decisions[offset:offset + len(tuples)]
                assert [row['id'] for row in rows] == [f'c{index}/a{entry}' for entry in range(len(tuples))]
                probabilities = [row['probabilities'][row['option_ids'].index('satisfied')] for row in rows]
                factors.append({'scope': scope, 'tuples': tuples, 'probabilities': probabilities})
                offset += len(tuples)
            assert offset == len(decisions)
            solution = soft_table_map(domain, factors, state['attributes'], args.time_limit)
            record.update(status=solution['status'], solver=solution)
            if solution['assignment'] is not None:
                record['prediction_rows'] = helpers['assignment_rows'](state, solution['assignment'])
        outputs.append(record)
        (args.output / f'{task["task_id"]}.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'task_id': record['task_id'], 'status': record['status']}), flush=True)
    (args.output / 'predictions.json').write_text(json.dumps(outputs, indent=2) + '\n')
    labels = {row['task_id']: row for row in map(json.loads, Path(settings['labels']).read_text().splitlines())}
    scores = [{'task_id': task['task_id'], **score_rows(task['puzzle'], labels[task['task_id']]['solution'],
               output['prediction_rows'])} for task, output in zip(tasks, outputs)]
    report = {
        'tasks': len(outputs), 'statuses': dict(Counter(row['status'] for row in outputs)),
        'solved': sum(row['solved'] for row in scores),
        'puzzle_accuracy': sum(row['solved'] for row in scores) / len(scores),
        'cell_accuracy': sum(row['correct_cells'] for row in scores) / sum(row['total_cells'] for row in scores),
        'cpu_seconds': sum(row['solver']['total_seconds'] for row in outputs if 'solver' in row),
        'scores': scores,
    }
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
