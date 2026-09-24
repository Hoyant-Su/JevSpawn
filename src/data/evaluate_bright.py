import argparse
import json
import math
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def ranking_metrics(ranked_ids, gold_ids, cutoff):
    gold = set(gold_ids)
    gains = [int(identity in gold) for identity in ranked_ids[:cutoff]]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(cutoff, len(gold))))
    return {'ndcg': dcg / ideal, 'recall': sum(gains) / len(gold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collections', type=Path, required=True)
    parser.add_argument('--relevance', type=Path, required=True)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collections = {row['task_id']: row for row in read_jsonl(args.collections)}
    relevance = {row['task_id']: row for row in read_jsonl(args.relevance)}
    predictions = {row['task_id']: row for row in read_jsonl(args.rankings)}
    assert collections.keys() == relevance.keys() == predictions.keys()
    results = []
    for identity, collection in collections.items():
        ranked = predictions[identity]['ranked_document_ids']
        candidates = {row['document_id'] for row in collection['candidates']}
        assert len(ranked) == len(candidates) and set(ranked) == candidates
        metrics = ranking_metrics(ranked, relevance[identity]['relevant_document_ids'], args.cutoff)
        results.append({'task_id': identity, **metrics})
    summary = {'queries': len(results), 'cutoff': args.cutoff,
               'ndcg': sum(row['ndcg'] for row in results) / len(results),
               'recall': sum(row['recall'] for row in results) / len(results), 'per_query': results}
    args.output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
