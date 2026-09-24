import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feasibility-count', type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / 'manifest.json').read_text())
    tasks = [json.loads(line) for line in (args.source / 'tasks.jsonl').read_text().splitlines()]
    tests = [json.loads(line) for line in (args.source / 'evaluation/tests.jsonl').read_text().splitlines()]
    split_ids = manifest['mbpp_split_task_ids']
    assert len(split_ids['test']) == 500 and set(split_ids['test']) == {f'MBPP/{i}' for i in range(11, 511)}
    groups = {'humaneval': [r['task_id'] for r in tasks if r['dataset'] == 'humaneval'],
              'mbpp_test': split_ids['test'], 'feasibility': split_ids['validation'][:args.feasibility_count]}
    assert len(groups['humaneval']) == 164
    assert not set(groups['feasibility']) & set(groups['mbpp_test'])
    for name, ids in groups.items():
        directory = args.output / name
        directory.mkdir(parents=True, exist_ok=True)
        for filename, rows in [('tasks', tasks), ('tests', tests)]:
            lookup = {r['task_id']: r for r in rows}
            (directory / f'{filename}.jsonl').write_text(''.join(json.dumps(lookup[i]) + '\n' for i in ids))
    metadata = {'source_manifest': str(args.source / 'manifest.json'), 'sources': manifest['sources'],
                'mbpp_examples': manifest['mbpp_examples'], 'protocol': manifest['protocol'],
                'split_task_ids': groups, 'counts': {name: len(ids) for name, ids in groups.items()},
                'quality_unit': 'One final solution per distinct task; timing repeats are not new task samples.'}
    (args.output / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata['counts']))


if __name__ == '__main__':
    main()
