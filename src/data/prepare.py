import argparse
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from project_paths import ROOT
from jev_spawn.infra.prompts import resolve_prompts


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def field(index, question, options):
    name = f'q{index}'
    return name, {'id': name, 'question': question, 'options': options}


def write_split(output, dataset, split, groups):
    directory = output / dataset / split
    directory.mkdir(parents=True, exist_ok=True)
    tasks, labels = [], []
    for identity, state, questions in groups:
        fields, answers, question_ids = {}, {}, {}
        for index, (question_id, question, options, answer) in enumerate(questions):
            name, definition = field(index, question, options)
            fields[name], answers[name], question_ids[name] = definition, answer, question_id
        task_id = f'{dataset}/{identity}'
        tasks.append({'task_id': task_id, 'dataset': dataset, 'state': state, 'fields': fields})
        labels.append({'task_id': task_id, 'labels': answers, 'question_ids': question_ids})
    for name, rows in [('tasks', tasks), ('labels', labels)]:
        (directory / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return {
        'tasks': len(tasks), 'decisions': sum(len(task['fields']) for task in tasks),
        'fields_per_task': dict(sorted(Counter(len(task['fields']) for task in tasks).items())),
        'class_counts': dict(Counter(value for row in labels for value in row['labels'].values())),
        'tasks_file': str(directory / 'tasks.jsonl'), 'labels_file': str(directory / 'labels.jsonl'),
    }


def prepare_aqua(source, output, schema):
    splits = {}
    for original, split in [('dev', 'feasibility'), ('test', 'evaluation')]:
        groups = []
        for index, row in enumerate(read_jsonl(source / f'aqua_{original}.json')):
            options = [{'id': label, 'description': text.split(')', 1)[1].strip()}
                       for label, text in zip(schema['option_ids'], row['options'], strict=True)]
            identity = f'{original}/{index}'
            groups.append((identity, row['question'], [(identity, schema['question'], options, row['correct'])]))
        splits[split] = write_split(output, 'aqua', split, groups)
    return splits


def prepare_race(source, output, schema):
    articles = defaultdict(list)
    for row in pq.read_table(source / 'race_validation.parquet').to_pylist():
        articles[row['example_id']].append(row)
    splits = {}
    for split, level in [('evaluation', schema['evaluation_level']), ('feasibility', schema['feasibility_level'])]:
        identities = sorted(identity for identity in articles if identity.startswith(level))
        if split == 'feasibility':
            identities = identities[:schema['feasibility_articles']]
        groups = []
        for identity in identities:
            rows = articles[identity]
            assert len({row['article'] for row in rows}) == 1, identity
            questions = []
            for index, row in enumerate(rows):
                options = [{'id': label, 'description': text}
                           for label, text in zip(schema['option_ids'], row['options'], strict=True)]
                questions.append((f'{identity}/{index}', schema['question'].format(question=row['question']), options, row['answer']))
            groups.append((identity, rows[0]['article'], questions))
        splits[split] = write_split(output, 'race_middle', split, groups)
    return splits


def prepare_pubmedqa(source, output, schema):
    labelled = json.loads((source / 'pubmedqa_labelled.json').read_text())
    test_labels = json.loads((source / 'pubmedqa_test_ids.json').read_text())
    splits = {}
    for split, identities in [('evaluation', set(test_labels)), ('feasibility', set(labelled) - set(test_labels))]:
        groups = []
        for identity in sorted(identities, key=int):
            row = labelled[identity]
            answer = row['final_decision']
            if split == 'evaluation':
                assert answer == test_labels[identity], identity
            question = schema['question'].format(question=row['QUESTION'])
            groups.append((identity, '\n\n'.join(row['CONTEXTS']), [(identity, question, schema['options'], answer)]))
        splits[split] = write_split(output, 'pubmedqa', split, groups)
    return splits


def prepare_scifact(source, output, schema):
    with tarfile.open(source / 'scifact.tar.gz') as archive:
        corpus = {row['doc_id']: row for row in map(json.loads, archive.extractfile('data/corpus.jsonl'))}
        claims = {split: list(map(json.loads, archive.extractfile(f'data/claims_{split}.jsonl')))
                  for split in ['dev', 'train']}
    splits = {}
    for original, split in [('dev', 'evaluation'), ('train', 'feasibility')]:
        groups = defaultdict(list)
        for claim in sorted(claims[original], key=lambda row: row['id']):
            assert set(claim['evidence']).issubset(set(map(str, claim['cited_doc_ids'])))
            for document_id in sorted(claim['cited_doc_ids']):
                answer = 'NOINFO'
                if str(document_id) in claim['evidence']:
                    document_labels = {item['label'] for item in claim['evidence'][str(document_id)]}
                    assert len(document_labels) == 1, (claim['id'], document_id)
                    answer = document_labels.pop()
                question = schema['question'].format(question=claim['claim'])
                groups[document_id].append((f"{claim['id']}/{document_id}", question, schema['options'], answer))
        prepared = [(f'{original}/{identity}', corpus[identity]['title'] + '\n\n' + '\n'.join(corpus[identity]['abstract']), questions)
                    for identity, questions in sorted(groups.items())]
        splits[split] = write_split(output, 'scifact_cited', split, prepared)
        splits[split]['unique_claims'] = len(claims[original])
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schema', type=Path, required=True)
    args = parser.parse_args()
    schema = resolve_prompts(json.loads(args.schema.read_text()))
    summaries = {
        'aqua': prepare_aqua(args.source, args.output, schema['aqua']),
        'race_middle': prepare_race(args.source, args.output, schema['race']),
        'pubmedqa': prepare_pubmedqa(args.source, args.output, schema['pubmedqa']),
        'scifact_cited': prepare_scifact(args.source, args.output, schema['scifact']),
    }
    metadata = json.loads((ROOT / 'configs/data/datasets.json').read_text())
    for name, summary in summaries.items():
        summary.update(metadata[name])
    (args.output / 'dataset_summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
