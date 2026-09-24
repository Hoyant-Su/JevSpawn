import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from prepare import read_jsonl, write_split
from jev_spawn.infra.prompts import resolve_prompts


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def prepare_med(source, output, config):
    data = {split: read_jsonl(source / 'medxpertqa_text' / f'{split}.jsonl') for split in ['dev', 'test']}
    assert len(data['test']) == config['test_total']
    splits = {}
    for split, original, indices in [('evaluation', 'test', config['evaluation_indices']),
                                     ('feasibility', 'dev', range(len(data['dev'])))]:
        groups = []
        for index in indices:
            row = data[original][index]
            options = [{'id': key, 'description': value} for key, value in row['options'].items()]
            assert row['label'] in row['options']
            groups.append((row['id'], row['question'], [(row['id'], config['question'], options, row['label'])]))
        splits[split] = write_split(output, 'medxpertqa_text', split, groups)
    return {'splits': splits, 'original_test_questions': len(data['test']),
            'selection': 'Seed 0 uniform sample without replacement, indices fixed before source retrieval.',
            'scope': 'Original text-only medical multiple-choice questions with all original choices and embedded choice text retained.'}


def prepare_super(source, output, config):
    splits = {}
    source_metadata = {}
    for split in ['evaluation', 'feasibility']:
        groups, metadata = [], []
        for index in config[f'{split}_indices']:
            response = json.loads((source / 'supergpqa' / f'{index}.json').read_text())
            assert response['num_rows_total'] == config['total']
            assert len(response['rows']) == 1 and response['rows'][0]['row_idx'] == index
            assert response['rows'][0]['truncated_cells'] == []
            row = response['rows'][0]['row']
            options = [{'id': config['option_ids'][choice], 'description': value}
                       for choice, value in enumerate(row['options'])]
            assert dict((item['id'], item['description']) for item in options)[row['answer_letter']] == row['answer']
            groups.append((row['uuid'], row['question'], [(row['uuid'], config['question'], options, row['answer_letter'])]))
            metadata.append({key: row[key] for key in ['uuid', 'discipline', 'field', 'subfield', 'difficulty', 'is_calculation']})
        splits[split] = write_split(output, 'supergpqa', split, groups)
        splits[split]['option_counts'] = dict(Counter(len(group[2][0][2]) for group in groups))
        source_metadata[split] = {'disciplines': dict(Counter(row['discipline'] for row in metadata)),
                                  'fields': dict(Counter(row['field'] for row in metadata))}
        save_jsonl(output / 'supergpqa' / split / 'source_metadata.jsonl', metadata)
    downloads = read_jsonl(source / 'downloads.jsonl')
    revisions = {row['headers']['x-revision'] for row in downloads if '/supergpqa' in row['file']}
    assert revisions == {config['revision']}, revisions
    return {'splits': splits, 'source_metadata': source_metadata, 'viewer_revisions': sorted(revisions),
            'selection': 'Seed 0 uniform sample of 272 original row indices without replacement. First 256 sampled indices evaluate, remaining 16 develop. Each split is stored in source order.',
            'scope': 'Graduate-level questions with every original option retained, including questions with fewer than ten options. This small unstratified subset does not estimate performance for every original discipline.'}


def retrieve(examples, documents, config):
    pattern = re.compile(config['token_pattern'])
    terms = [Counter(pattern.findall(row['content'].lower())) for row in documents]
    lengths = [sum(row.values()) for row in terms]
    average = sum(lengths) / len(lengths)
    postings = defaultdict(list)
    for index, document in enumerate(terms):
        for term, count in document.items():
            postings[term].append((index, count))
    records = []
    for example in examples:
        scores = [0.0] * len(documents)
        for term, query_count in Counter(pattern.findall(example['query'].lower())).items():
            matches = postings[term]
            idf = math.log(1 + (len(documents) - len(matches) + 0.5) / (len(matches) + 0.5))
            for index, count in matches:
                denominator = count + config['bm25_k1'] * (1 - config['bm25_b'] + config['bm25_b'] * lengths[index] / average)
                scores[index] += query_count * idf * count * (config['bm25_k1'] + 1) / denominator
        excluded = set(example['excluded_ids'])
        eligible = [i for i, row in enumerate(documents) if row['id'] not in excluded]
        ordered = sorted(eligible, key=lambda i: (-scores[i], documents[i]['id']))[:config['top_k']]
        assert len(ordered) == config['top_k']
        records.append({'task_id': f"bright_pony/{example['id']}", 'dataset': 'bright_pony',
                        'query_id': example['id'], 'query': example['query'],
                        'candidates': [{'document_id': documents[index]['id'], 'text': documents[index]['content'],
                                        'retrieval_rank': rank + 1, 'retrieval_score': scores[index]}
                                       for rank, index in enumerate(ordered)]})
    return records


def prepare_bright(source, output, config):
    examples = pq.read_table(source / 'bright_pony/examples.parquet').to_pylist()
    documents = pq.read_table(source / 'bright_pony/documents.parquet').to_pylist()
    assert len(examples) == config['original_queries']
    document_ids = {row['id'] for row in documents}
    assert len(document_ids) == len(documents)
    assert all(set(row['gold_ids']) <= document_ids for row in examples)
    records = retrieve(examples, documents, config)
    save_jsonl(output / 'bright_pony/corpus.jsonl', documents)
    save_jsonl(output / 'bright_pony/retrieved_pools.jsonl', records)
    splits = {}
    for split, indices, count in [('feasibility', config['development_indices'], config['development_prefix_k']),
                                   ('evaluation', config['evaluation_indices'], config['top_k'])]:
        groups, collections, relevance, coverage = [], [], [], []
        for index in indices:
            example, record = examples[index], records[index]
            candidates = record['candidates'][:count]
            collections.append({**record, 'candidates': candidates})
            gold = set(example['gold_ids'])
            questions = [(candidate['document_id'], config['question'].format(document=candidate['text']),
                          config['options'], 'yes' if candidate['document_id'] in gold else 'no')
                         for candidate in candidates]
            groups.append((example['id'], example['query'], questions))
            relevance.append({'task_id': record['task_id'], 'relevant_document_ids': example['gold_ids'],
                              'excluded_document_ids': example['excluded_ids']})
            selected = [candidate['document_id'] for candidate in candidates]
            hits = len(set(selected) & gold)
            cutoff = config['metric_cutoff']
            dcg = sum(int(doc in gold) / math.log2(rank + 2) for rank, doc in enumerate(selected[:cutoff]))
            ideal = sum(1 / math.log2(rank + 2) for rank in range(min(cutoff, len(gold))))
            coverage.append({'task_id': record['task_id'], 'gold_documents': len(gold), 'retrieved_gold': hits,
                             'recall': hits / len(gold), 'any_gold': hits > 0, 'bm25_ndcg_at_10': dcg / ideal})
        splits[split] = write_split(output, 'bright_pony', split, groups)
        directory = output / 'bright_pony' / split
        save_jsonl(directory / 'collections.jsonl', collections)
        save_jsonl(directory / 'relevance.jsonl', relevance)
        save_jsonl(directory / 'retrieval_coverage.jsonl', coverage)
        splits[split].update(candidate_documents=count, collections_file=str(directory / 'collections.jsonl'),
                             macro_candidate_recall=sum(row['recall'] for row in coverage) / len(coverage),
                             queries_with_relevant_candidate=sum(row['any_gold'] for row in coverage),
                             bm25_ndcg_at_10=sum(row['bm25_ndcg_at_10'] for row in coverage) / len(coverage))
    lengths = [len(row['content']) for row in documents]
    return {'splits': splits, 'original_queries': len(examples), 'corpus_documents': len(documents),
            'document_characters': {'minimum': min(lengths), 'maximum': max(lengths), 'mean': sum(lengths) / len(lengths)},
            'selection': 'All Pony queries partitioned with seed 0 shuffle. Eight development queries and 104 held-out evaluation queries. This is a study-specific split.',
            'retriever': {'name': 'BM25 with positive Robertson IDF', 'k1': config['bm25_k1'], 'b': config['bm25_b'],
                          'tokenization': config['token_pattern'], 'lowercase': True, 'tie_break': 'lexicographic document ID',
                          'excluded_ids': 'Original per-query exclusions applied before ranking.', 'gold_injection': False},
            'scope': 'Relevance classification and reranking within BM25 candidates from the full original Pony corpus. Original gold IDs evaluate nDCG@10 and candidate recall. Unlisted documents are benchmark negatives, not guaranteed exhaustive semantic irrelevance. Each query is one root, each candidate is one document worker. Binary fields are an additional fixed-schema adapter.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = resolve_prompts(json.loads(args.config.read_text()))
    summary = {'medxpertqa_text': prepare_med(args.source, args.output, config['medxpertqa_text']),
               'supergpqa': prepare_super(args.source, args.output, config['supergpqa']),
               'bright_pony': prepare_bright(args.source, args.output, config['bright_pony'])}
    for name, entry in summary.items():
        entry['repository'] = config[name]['repository']
        entry['revision'] = config[name]['revision']
        entry['paper_url'] = config[name]['paper_url']
        entry['dataset_url'] = config[name]['dataset_url']
        (args.output / name / 'dataset_summary.json').write_text(json.dumps(entry, indent=2) + '\n')
    (args.output / 'recent_dataset_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
