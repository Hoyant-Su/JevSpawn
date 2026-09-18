"""Bind a structured-workload template to local inputs, weights and output paths."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--fields', type=Path)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', choices=['shared', 'streamed', 'independent', 'compact_json'], required=True)
    parser.add_argument('--task-count', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--branch-batch-size', type=int)
    parser.add_argument('--world-size', type=int)
    args = parser.parse_args()
    config = json.loads(args.template.read_text())
    config.update(model_path=str(args.model.resolve()), tasks_path=str(args.tasks.resolve()),
                  fields_path=str(args.fields.resolve()) if args.fields else None,
                  run_dir=str(args.run_dir.resolve()), run_id=args.run_dir.name, method=args.method,
                  task_count=len(args.tasks.read_text().splitlines()) - config['task_offset'])
    for name in ('task_count', 'batch_size', 'branch_batch_size', 'world_size'):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2)+'\n')
    print(json.dumps({'config': str(args.output), 'tasks': config['task_count'], 'method': args.method}))


if __name__ == '__main__':
    main()
