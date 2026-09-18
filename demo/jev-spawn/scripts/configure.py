"""Adapt an experiment configuration to local model, dataset and result paths."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spawn-policy", choices=["single", "adaptive", "all"], required=True)
    parser.add_argument("--field-mode", choices=["shared", "independent"], required=True)
    args = parser.parse_args()
    config = json.loads(args.template.read_text())
    config.update(model_path=str(args.model.resolve()), tasks_path=str(args.tasks.resolve()),
                  run_dir=str(args.run_dir.resolve()), run_id=args.run_dir.name,
                  spawn_policy=args.spawn_policy, field_mode=args.field_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
