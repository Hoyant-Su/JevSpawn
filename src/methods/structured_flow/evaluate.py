import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import jsonschema

from data.evaluate_zebralogic import score_rows


def read(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def evaluate(run, reference, sandbox, timeout_seconds, memory_mb):
    completion = read(run / 'completion.json')
    protocol = read(run / 'protocol.json')
    tasks = protocol['tasks']
    assert completion['tasks'] == len(tasks)
    methods = protocol['settings']['methods']
    results = {method: [read(run / method / f'task-{index:04d}' / 'complete.json')
                        for index in range(len(tasks))] for method in methods}
    assert all([row['task']['task_id'] for row in rows] == [task['task_id'] for task in tasks]
               for rows in results.values())
    references = {row['task_id']: row for row in read(reference)['tasks']}
    labels = {path: {row['task_id']: row for row in read_rows(path)}
              for path in {ref['path'] for ref in references.values()
                           if ref['kind'] != 'original_sandbox_tests'}}
    report = {}
    for method, rows in results.items():
        scores, coding = [], []
        for task, result in zip(tasks, rows):
            identifier = task['task_id']
            ref = references[identifier]
            score = {'task_id': identifier, 'dataset': task['dataset'], 'valid': False, 'correct': False}
            scores.append(score)
            if result['status'] != 'completed':
                score['error'] = result['error']
                continue
            try:
                output = result['output']
                prediction = json.loads(output['text']) if isinstance(output, dict) and set(output) == {'text'} else output
                jsonschema.validate(prediction, task['answer_contract'])
            except (ValueError, TypeError, jsonschema.ValidationError) as error:
                score['error'] = type(error).__name__ + ': ' + str(error)
                continue
            score['valid'] = True
            if ref['kind'] == 'original_sandbox_tests':
                coding.append({'task_id': identifier, 'solution': prediction['code']})
            elif ref['kind'] == 'grid_cell_and_puzzle':
                score.update(score_rows(task['input']['puzzle'], labels[ref['path']][identifier]['solution'], prediction['rows']))
                score['correct'] = score['solved']
            else:
                target = labels[ref['path']][identifier]['labels']
                score.update(correct=prediction == target,
                             correct_fields=sum(prediction[key] == value for key, value in target.items()),
                             total_fields=len(target))
        if coding:
            directory = run / method / 'offline'
            directory.mkdir(exist_ok=True)
            solutions = directory / 'solutions.jsonl'
            solutions.write_text(''.join(json.dumps(row) + '\n' for row in coding))
            tests = {references[row['task_id']]['path'] for row in coding}
            assert len(tests) == 1
            evaluator = Path(__file__).parents[5] / 'research/jevspawn-paper/jevspawn/src/jev_spawn/cli/evaluate.py'
            subprocess.run([sys.executable, str(evaluator), '--solutions', str(solutions),
                            '--tests', tests.pop(), '--sandbox', str(sandbox),
                            '--output', str(directory / 'tests.jsonl'), '--work-dir', str(directory / 'sandbox'),
                            '--workers', str(len(coding)), '--timeout', str(timeout_seconds),
                            '--memory-mb', str(memory_mb)], check=True)
            outcomes = {row['task_id']: row for row in read_rows(directory / 'tests.jsonl')}
            for score in scores:
                if score['task_id'] in outcomes:
                    score['sandbox'] = outcomes[score['task_id']]
                    score['correct'] = score['sandbox']['status'] == 'passed'
        datasets = defaultdict(list)
        for score in scores:
            datasets[score['dataset']].append(score)
        report[method] = {
            'tasks': len(scores), 'valid': sum(score['valid'] for score in scores),
            'correct': sum(score['correct'] for score in scores),
            'datasets': {name: {'tasks': len(group), 'valid': sum(score['valid'] for score in group),
                                'correct': sum(score['correct'] for score in group)} for name, group in datasets.items()},
            'compute_wall_seconds': sum(row['compute_wall_seconds'] for row in rows),
            'peak_allocated_bytes': max(row['profile']['peak_allocated_bytes'] or 0 for row in rows),
            'instantiated_nodes': [row['profile']['instantiated_nodes'] for row in rows],
            'completed_model_requests': [sum(record['result'].get('logical_field_count',
                                                    record['result']['batch_size'])
                                             for record in row['records']) for row in rows],
            'model_request_count_scope': 'Count completed backend requests from raw records, including outputs rejected by subsequent parsing or graph validation. Host nodes are excluded. Failed backend calls without a returned record are not counted as completed requests.',
            'scores': scores,
        }
    output = {'scope': 'Eight development tasks, two per task family. All inference finished before label access. No aggregate generalization or speedup claim from this qualification.',
              'methods': report}
    (run / 'evaluation.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({method: {key: value for key, value in result.items() if key != 'scores'}
                      for method, result in report.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--sandbox', type=Path, required=True)
    parser.add_argument('--timeout-seconds', type=float, required=True)
    parser.add_argument('--memory-mb', type=int, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.reference, args.sandbox, args.timeout_seconds, args.memory_mb)


if __name__ == '__main__':
    main()
