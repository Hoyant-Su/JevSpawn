import argparse
import json
from pathlib import Path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def save_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser()
    for name in ['run', 'layout', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'summary.json').read_text())
    layout = json.loads(args.layout.read_text())
    blocks = {row['repeat']: {k: row[k] for k in ['instance', 'gpu', 'repeat']} for row in layout['blocks']}
    reference = json.loads((args.run / 'repeat-00-single/states.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for method, result in report['methods'].items():
        method_reference = json.loads((args.run / f'repeat-00-{method}/states.json').read_text())
        for row in result['repeats']:
            name = f"repeat-{row['repeat']:02d}-{method}"
            source, target = args.run / name, args.output / name
            target.mkdir(exist_ok=True)
            states = json.loads((source / 'states.json').read_text())
            assert [s['task_id'] for s in states] == [s['task_id'] for s in reference]
            records = []
            for state, first, same_method in zip(states, reference, method_reference):
                records.append({'task_id': state['task_id'], 'usage': state['usage'],
                                'decisions': [{k: decision[k] for k in ['stage', 'choice', 'probabilities']}
                                              for decision in state['decisions']],
                                'initial_draft_matches_reference': state['initial_solution'] == first['initial_solution'],
                                'final_solution_matches_first_repeat': state['solution'] == same_method['solution']})
            save_rows(target / 'tasks.jsonl', records)
            save_rows(target / 'evaluation.jsonl', [{k: test[k] for k in ['task_id', 'status']}
                                                   for test in jsonl(source / 'evaluation.jsonl')])
            save_rows(target / 'generation.jsonl', [{k: call[k] for k in ['input_tokens', 'output_tokens',
                      'elapsed_seconds', 'truncated', 'batch_size', 'decode']} for call in jsonl(source / 'generation.jsonl')])
            (target / 'metrics.json').write_text((source / 'metrics.json').read_text())
            row.update(source_directory=name, timing_block=blocks[row['repeat']])
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'blocks.json').write_text(json.dumps(list(blocks.values()), indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'tasks': report['tasks']}))


if __name__ == '__main__':
    main()
