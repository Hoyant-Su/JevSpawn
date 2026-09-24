import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from urllib.request import urlopen
from jev_spawn.infra.prompts import resolve_prompts


def download(url):
    with urlopen(url, timeout=30) as response:
        return response.read().decode()


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = resolve_prompts(json.loads(args.config.read_text()))
    repository = config['repository']
    revision = json.loads(download(f'https://api.github.com/repos/{repository}/commits/{config["revision"]}'))['sha']
    base = f'https://raw.githubusercontent.com/{repository}/{revision}/'
    names = ['dataset/source_info.jsonl', 'dataset/response.jsonl',
             'baseline/predict_and_evaluate.py', 'baseline/prepare_dataset.py', 'baseline/dataset.py', 'LICENSE']
    with ThreadPoolExecutor(max_workers=2) as pool:
        contents = dict(zip(names, pool.map(download, [base + name for name in names])))
    sources = {row['source_id']: row for row in map(json.loads, contents[names[0]].splitlines())}
    grouped = defaultdict(list)
    for row in map(json.loads, contents[names[1]].splitlines()):
        if row['quality'] == config['quality']:
            key = (row['split'], sources[row['source_id']]['task_type'], row['source_id'])
            grouped[key].append(row)
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    raw = root / 'sources'
    raw.mkdir()
    for name in names[2:]:
        (raw / Path(name).name).write_text(contents[name])
    rng = random.Random(config['seed'])
    selected, split_sources, summary = [], {}, {}
    for split in ['development', 'evaluation']:
        original_split = config[f'{split}_source_split']
        chosen = []
        for task_type in config['task_types']:
            candidates = sorted(key for key in grouped if key[:2] == (original_split, task_type))
            keys = rng.sample(candidates, config[f'{split}_per_type'])
            chosen.extend(rng.choice(sorted(grouped[key], key=lambda row: row['id'])) for key in keys)
        selected.extend(chosen)
        split_sources[split] = {row['source_id'] for row in chosen}
        directory = root / split
        directory.mkdir()
        tasks, labels = [], []
        for row in chosen:
            source = sources[row['source_id']]
            identity = f'ragtruth/{row["id"]}'
            tasks.append({'task_id': identity, 'source_id': row['source_id'], 'task_type': source['task_type'],
                          'source_info': source['source_info'], 'response': row['response']})
            labels.append({'task_id': identity, 'source_id': row['source_id'], 'labels': row['labels'],
                           'model': row['model'], 'original_split': row['split'], 'quality': row['quality']})
        write_jsonl(directory / 'tasks.jsonl', tasks)
        write_jsonl(directory / 'labels.jsonl', labels)
        summary[split] = {'responses': len(chosen), 'sources': len(split_sources[split]),
                          'by_type': {kind: sum(row['task_type'] == kind for row in tasks)
                                      for kind in config['task_types']}}
    assert split_sources['development'].isdisjoint(split_sources['evaluation'])
    write_jsonl(raw / 'response.jsonl', selected)
    write_jsonl(raw / 'source_info.jsonl', [sources[identity] for identity in sorted(set.union(*split_sources.values()))])
    for row in selected:
        for label in row['labels']:
            assert row['response'][label['start']:label['end']] == label['text'], (row['id'], label)
    metadata = {'repository': repository, 'revision': revision, 'config': config, 'splits': summary,
                'annotation_offsets_valid': True, 'upstream_urls': [base + name for name in names],
                'scope': 'Only the configured response subset and its source records are retained. Original task and labels are unchanged.'}
    (root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(summary)


if __name__ == '__main__':
    main()
