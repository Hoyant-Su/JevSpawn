import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--development-count", type=int, required=True)
    parser.add_argument("--evaluation-count", type=int, required=True)
    args = parser.parse_args()
    records = pq.read_table(args.output / "sources/grid.parquet").to_pylist()
    sizes = sorted({row["size"] for row in records})
    rng = random.Random(args.seed)
    pools = {size: [i for i, row in enumerate(records) if row["size"] == size] for size in sizes}
    for pool in pools.values():
        rng.shuffle(pool)
    size_order = sizes.copy()
    rng.shuffle(size_order)
    splits = {}
    for split, count in (("development", args.development_count), ("evaluation", args.evaluation_count)):
        indices = [pools[size_order[i % len(sizes)]].pop() for i in range(count)]
        splits[split] = sorted(indices)
    splits["remaining"] = sorted(index for pool in pools.values() for index in pool)
    assert len({row["id"] for row in records}) == len(records)
    for split, indices in splits.items():
        write_jsonl(args.output / split / "tasks.jsonl",
                    [{"task_id": records[i]["id"], "puzzle": records[i]["puzzle"]} for i in indices])
        write_jsonl(args.output / split / "labels.jsonl",
                    [{"task_id": records[i]["id"], "size": records[i]["size"], "solution": records[i]["solution"]} for i in indices])
    manifest = {
        "source": json.loads((args.output / "sources/source.json").read_text()),
        "seed": args.seed, "size_order": size_order, "source_indices": splits,
        "selection": "Shuffle source-index pools within sorted original size strata with one Python Random instance, shuffle the stratum order, then draw round-robin from that order separately for development and evaluation. No solution contents or model outputs enter selection.",
        "development_count": args.development_count, "evaluation_count": args.evaluation_count,
        "source_size_counts": dict(Counter(row["size"] for row in records)),
        "splits": {name: {"count": len(indices), "sizes": dict(Counter(records[i]["size"] for i in indices))}
                   for name, indices in splits.items()},
        "model_visible_fields": ["task_id", "puzzle"],
        "labels": "Original solution and original size remain evaluation-only. Input variable names and domains must be derived from puzzle text, never from solution headers.",
        "scope": "Original finite logic grids with natural-language clues. The release does not provide a formal constraint AST or gold clue scopes.",
        "metric": "Single-output case-normalized cell agreement excluding the House index column. Puzzle success requires all gold cells. Missing outputs count as failures. No best-of-N or gold-dependent selection.",
        "limits": ["These are study-specific disjoint subsets of the original test split.",
                   "Eight development puzzles cannot cover all25 original size strata. The64 evaluation puzzles cover each stratum at least twice.",
                   "All-different and local clue consistency are task-specific structural assumptions, not a learned general reasoning guarantee."]}
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
