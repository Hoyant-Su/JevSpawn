import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--development-per-type", type=int, required=True)
    parser.add_argument("--evaluation-per-type", type=int, required=True)
    args = parser.parse_args()
    source = args.output / "sources"
    corpus = json.loads((source / "corpus.json").read_text())
    questions = json.loads((source / "MultiHopRAG.json").read_text())
    documents = [{"id": f"doc{i:04d}", **row} for i, row in enumerate(corpus)]
    url_ids = {row["url"]: row["id"] for row in documents}
    assert len(url_ids) == len(documents)
    units = []
    for doc in documents:
        start = 0
        for index, text in enumerate(doc["body"].split("\n\n")):
            end = start + len(text)
            assert doc["body"][start:end] == text
            units.append({"id": f"{doc['id']}/p{index:04d}", "doc_id": doc["id"],
                          "start": start, "end": end, "text": text})
            start = end + 2
    write_jsonl(args.output / "corpus.jsonl", documents)
    write_jsonl(args.output / "source_units.jsonl", units)
    rng = random.Random(args.seed)
    splits = {"development": [], "evaluation": [], "remaining": []}
    categories = sorted({row["question_type"] for row in questions})
    for category in categories:
        indices = [i for i, row in enumerate(questions) if row["question_type"] == category]
        rng.shuffle(indices)
        ndev, neval = args.development_per_type, args.evaluation_per_type
        assert len(indices) >= ndev + neval
        splits["development"].extend(indices[:ndev])
        splits["evaluation"].extend(indices[ndev:ndev + neval])
        splits["remaining"].extend(indices[ndev + neval:])
    for split, indices in splits.items():
        indices.sort()
        tasks, labels = [], []
        for index in indices:
            row = questions[index]
            task_id = f"multihoprag/{index}"
            tasks.append({"task_id": task_id, "query": row["query"]})
            labels.append({"task_id": task_id, "answer": row["answer"],
                           "question_type": row["question_type"], "evidence_list": row["evidence_list"],
                           "evidence_document_ids": [url_ids[e["url"]] for e in row["evidence_list"]]})
        write_jsonl(args.output / split / "tasks.jsonl", tasks)
        write_jsonl(args.output / split / "labels.jsonl", labels)
    manifest = {
        "source": json.loads((source / "source.json").read_text()),
        "evaluation_source": json.loads((source / "evaluation_source.json").read_text()),
        "seed": args.seed, "indices": splits,
        "selection": "Shuffle original source indices within lexicographically sorted original question types using one Python Random instance. Equal counts per type. Development and evaluation are disjoint study-specific splits. No model outputs enter selection.",
        "development_per_type": args.development_per_type,
        "evaluation_per_type": args.evaluation_per_type,
        "question_type_hidden": True,
        "shared_corpus": "Every split uses the complete original corpus. Gold answers, evidence, and question types are evaluation-only."}
    (args.output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lengths = [len(row["body"]) for row in documents]
    unit_lengths = [len(row["text"]) for row in units]
    summary = {
        "corpus_documents": len(documents), "questions": len(questions), "source_units": len(units),
        "source_bytes": {name: (source / name).stat().st_size for name in ("corpus.json", "MultiHopRAG.json")},
        "question_types": dict(Counter(row["question_type"] for row in questions)),
        "evidence_documents_per_question": dict(Counter(len(row["evidence_list"]) for row in questions)),
        "splits": {name: {"count": len(indices), "question_types": dict(Counter(questions[i]["question_type"] for i in indices))}
                   for name, indices in splits.items()},
        "body_characters": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)},
        "unit_characters": {"min": min(unit_lengths), "median": statistics.median(unit_lengths), "max": max(unit_lengths)},
        "units": "Exact original body.split('\\n\\n') segments, including source headings and lists. These are source-separated text units, not asserted grammatical paragraphs. Offsets are Python Unicode character offsets into the unchanged body. No sentence split, merging, truncation, or gold-driven boundary selection.",
        "model_visible": {"tasks": ["task_id", "query"], "corpus": ["id"] + list(corpus[0]),
                          "source_units": ["id", "doc_id", "start", "end", "text"]},
        "task": "Answer an original query after obtaining evidence through shared search and read tools over the original corpus. Initial input contains no document text or gold source identifiers. Preserve title, date, author, source, category, and URL in readable documents.",
        "official_evaluation": {
            "qa": "Pinned qa_evaluate.py extracts the optional original answer phrase and counts any lowercase whitespace-token intersection with the gold answer as success. Its reported precision, recall, F1, and accuracy are the same scalar, not strict exact match or token F1.",
            "retrieval": "Pinned retrieval_evaluate.py excludes null_query and matches gold fact strings inside retrieved texts after removing spaces and newlines. It reports Hits@4, Hits@10, MRR@10 and a source-specific MAP@10 implementation. Its AP adds newly found facts divided by rank rather than cumulative precision, so it is not standard document AP.",
            "recommendation": "Report explicit normalized exact match and token F1 alongside the named upstream token-overlap metric. Evaluate source-document evidence recall separately using original URL-aligned IDs. Keep null queries in answer scoring and report their abstention accuracy separately."},
        "limits": ["Balanced category sampling changes original category weights. Report per-category and balanced aggregate metrics.",
                   "Some queries require parallel comparison rather than a serial entity bridge. Report observed evidence-dependent execution edges without claiming every item needs serial search.",
                   "Original body lengths can exceed one model context. Source units must be read without silently truncating documents.",
                   "Question and corpus overlap across development and evaluation is inherited from one shared corpus. Query identities are disjoint."],
        "sources": ["https://huggingface.co/datasets/yixuantt/MultiHopRAG", "https://github.com/yixuantt/MultiHop-RAG"]}
    (args.output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
