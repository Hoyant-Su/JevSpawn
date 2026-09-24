import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    layout = json.loads(args.layout.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    reference = None
    for block in layout['blocks']:
        for method in layout['methods']:
            name = f"repeat-{block['repeat']:02d}-{method}"
            source = Path(block['directory']) / name
            run = json.loads((source / 'run.json').read_text())
            assert run['config']['timing_block'] == {k: block[k] for k in ['instance', 'gpu', 'repeat']}
            assert len(run['task_ids']) == layout['task_count']
            if reference is None:
                reference = run['task_ids']
            assert run['task_ids'] == reference
            assert (source / 'metrics.json').exists() and (source / 'evaluation.jsonl').exists()
            target = args.output / name
            if target.is_symlink():
                assert target.resolve() == source.resolve()
            else:
                target.symlink_to(source.resolve(), target_is_directory=True)
    (args.output / 'blocks.json').write_text(json.dumps(layout, indent=2) + '\n')


if __name__ == '__main__':
    main()
