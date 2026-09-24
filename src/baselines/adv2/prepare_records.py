import argparse
import json
from pathlib import Path


def processbench_records(source):
    return [{'task_id': f"processbench/{row['id']}", 'task': row['problem'],
             'role': 'Mathematical solver', 'solution': '\n\n'.join(row['steps'])}
            for row in json.loads(Path(source).read_text())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = processbench_records(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row) + '\n' for row in records))
    print(json.dumps({'examples': len(records), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
