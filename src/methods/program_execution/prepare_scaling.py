import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from data.prepare_recent import retrieve, save_jsonl


def main():
    paper = Path(__file__).resolve().parents[3]
    data = paper.parents[1] / 'data/research_v1'
    directory = data / 'decision_program'
    config = json.loads((paper / 'configs/data/recent_datasets.json').read_text())['bright_pony']
    config['top_k'] = 1024
    source = data / 'sources/recent2025/bright_pony/examples.parquet'
    examples = pq.read_table(source, columns=['id', 'query', 'excluded_ids']).to_pylist()
    examples = [examples[index] for index in config['development_indices']]
    documents = [json.loads(line) for line in (data / 'bright_pony/corpus.jsonl').read_text().splitlines()]
    records = retrieve(examples, documents, config)
    old = [json.loads(line) for line in (directory / 'bright-development-128-collections.jsonl').read_text().splitlines()]
    assert [row['task_id'] for row in records] == [row['task_id'] for row in old]
    assert all(row['candidates'][:128] == previous['candidates'] for row, previous in zip(records, old))
    assert all(len({item['document_id'] for item in row['candidates']}) == 1024 for row in records)
    native = json.loads((paper / 'configs/native.json').read_text())
    native['batch_size'] = 1
    native_path = paper / 'configs/methods/program_execution/native-single.json'
    native_path.write_text(json.dumps(native, indent=2) + '\n')
    template = json.loads((paper / 'configs/methods/program_execution/tiled-development-128.json').read_text())
    for count in (512, 1024):
        collections = directory / f'bright-development-{count}-collections.jsonl'
        jobs = directory / f'bright-development-{count}-jobs.jsonl'
        save_jsonl(collections, [{**row, 'candidates': row['candidates'][:count]} for row in records])
        mapping = json.loads((paper / 'configs/methods/decision_program/bright-input.json').read_text())
        mapping['source'] = str(collections)
        mapping_path = directory / f'bright-development-{count}-input.json'
        mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
        subprocess.run([sys.executable, str(paper / 'src/data/prepare_decision_programs.py'),
                        '--settings', str(mapping_path), '--output', str(jobs)], check=True)
        settings = {**template, 'native_config': str(native_path), 'jobs': str(jobs),
                    'collections': str(collections), 'batch_size': 1, 'items_per_root': count,
                    'modes': ['tiled_independent', 'tiled_shared'], 'task_execution': 'sequential_single_root',
                    'compiler_policy': 'Fresh compilation charged once per root and execution arm.'}
        (paper / f'methods/program_execution/single-development-{count}.json').write_text(json.dumps(settings, indent=2) + '\n')
    stage = {'roots': [row['task_id'] for row in records], 'candidate_counts': [512, 1024],
             'corpus_documents': len(documents), 'retriever_config': config,
             'source_columns': ['id', 'query', 'excluded_ids'], 'gold_injection': False,
             'preserved_previous_128_exactly': True, 'root_batch_size': 1, 'live_leaf_cap': 128,
             'program': 'Frozen S5 contract program reproduced by fresh compilation per root.',
             'larger_size_modes': ['tiled_independent', 'tiled_shared'],
             'streamed': 'Not scheduled at these sizes. Completed ablation uses 32 and 128 documents.',
             'scope': 'Eight independent development queries evaluated sequentially. Runtime fans out input items. The model generates the worker decision program.'}
    (directory / 'single-task-scaling-stage.json').write_text(json.dumps(stage, indent=2) + '\n')
    print(json.dumps({'roots': len(records), 'documents_per_root': 1024, 'exact_previous_prefix': True}))


if __name__ == '__main__':
    main()
