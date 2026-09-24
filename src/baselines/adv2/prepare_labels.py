import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.source.read_text())
    problems = {problem: index for index, problem in enumerate(dict.fromkeys(r['problem'] for r in rows))}
    labels = [{'task_id': f"processbench/{r['id']}", 'problem_group': problems[r['problem']],
               'acceptable': r['label'] == -1, 'first_error': r['label']} for r in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in labels))
    print(json.dumps({'solutions': len(labels), 'problem_groups': len(problems),
                      'first_error_free': sum(r['acceptable'] for r in labels)}))


if __name__ == '__main__':
    main()
