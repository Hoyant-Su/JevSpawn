import argparse
import collections
import concurrent.futures
import datetime
import json
from pathlib import Path
import statistics
import urllib.request


COMMIT = "240e165ab706d95bd4323653bb92421f446208cf"
UPSTREAM = f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{COMMIT}"
SOURCES = {
    "dev-v2.0.json": f"{UPSTREAM}/dataset/dev-v2.0.json",
    "source-readme.txt": f"{UPSTREAM}/README.md",
    "source-index.pug": f"{UPSTREAM}/views/index.pug",
}


def download(item, directory):
    name, url = item
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    (directory / name).write_bytes(content)
    return {"file": name, "url": url, "bytes": len(content)}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def describe(rows):
    sizes = [len(row["labels"]) for row in rows]
    questions = sum(sizes)
    positive = sum(sum(row["labels"].values()) for row in rows)
    assert 0 < positive < questions
    return {
        "paragraphs": len(rows), "questions": questions,
        "answerable": positive, "unanswerable": questions - positive,
        "questions_per_paragraph": {"minimum": min(sizes), "maximum": max(sizes), "mean": statistics.mean(sizes), "median": statistics.median(sizes), "histogram": dict(sorted(collections.Counter(sizes).items()))},
        "baselines": {
            "always_yes": {"accuracy": positive / questions, "balanced_accuracy": 0.5, "paragraph_exact_match": sum(all(row["labels"].values()) for row in rows) / len(rows)},
            "always_no": {"accuracy": (questions - positive) / questions, "balanced_accuracy": 0.5, "paragraph_exact_match": sum(not any(row["labels"].values()) for row in rows) / len(rows)},
            "majority_class": {"choice": "yes" if positive > questions - positive else "no", "accuracy": max(positive, questions - positive) / questions, "balanced_accuracy": 0.5},
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SQuAD2.0 answerability fields grouped by original paragraph, without answer spans in model input.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "data/squad2_answerability")
    parser.add_argument("--template", type=Path, default=Path(__file__).resolve().parents[1] / "src/jev_spawn/schema/squad2_answerability.json")
    parser.add_argument("--feasibility-paragraphs", type=int, required=True)
    args = parser.parse_args()
    raw = args.output / "sources"
    evaluation = args.output / "evaluation"
    raw.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        sources = list(pool.map(lambda item: download(item, raw), SOURCES.items()))
    source = json.loads((raw / "dev-v2.0.json").read_text())
    template = json.loads(args.template.read_text())
    tasks, labels = [], []
    for article_index, article in enumerate(source["data"]):
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            task_id = f"SQuAD2/{article_index}/{paragraph_index}"
            fields = {f"q{index}": {**template, "id": f"q{index}", "question": template["question"].format(question=qa["question"])} for index, qa in enumerate(paragraph["qas"])}
            assert fields
            tasks.append({"task_id": task_id, "dataset": "squad2_answerability", "state": paragraph["context"], "fields": fields})
            labels.append({"task_id": task_id, "dataset": "squad2_answerability", "labels": {f"q{index}": not qa["is_impossible"] for index, qa in enumerate(paragraph["qas"])}, "question_ids": {f"q{index}": qa["id"] for index, qa in enumerate(paragraph["qas"])}, "article_index": article_index, "paragraph_index": paragraph_index, "article_title": article["title"]})
    question_ids = [question_id for row in labels for question_id in row["question_ids"].values()]
    assert len(question_ids) == len(set(question_ids))
    assert len(tasks) == len({row["task_id"] for row in tasks})
    assert 0 < args.feasibility_paragraphs <= len(tasks)
    sample_name = f"first{args.feasibility_paragraphs}"
    write_jsonl(args.output / "tasks.jsonl", tasks)
    write_jsonl(args.output / f"tasks.{sample_name}.jsonl", tasks[:args.feasibility_paragraphs])
    write_jsonl(evaluation / "labels.jsonl", labels)
    write_jsonl(evaluation / f"labels.{sample_name}.jsonl", labels[:args.feasibility_paragraphs])
    (args.output / "field-template.json").write_bytes(args.template.read_bytes())
    summaries = {sample_name: describe(labels[:args.feasibility_paragraphs]), "full_dev": describe(labels)}
    manifest = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dataset": "SQuAD2.0 answerability", "source_version": source["version"], "split": "official development", "source_commit": COMMIT, "sources": sources,
        "official_site": "https://rajpurkar.github.io/SQuAD-explorer/",
        "official_download_link": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
        "license": {"spdx": "CC-BY-SA-4.0", "url": "https://creativecommons.org/licenses/by-sa/4.0/", "pinned_statement_source": SOURCES["source-index.pug"], "attribution": "SQuAD2.0 by Pranav Rajpurkar, Robin Jia, Percy Liang and collaborators; source passages derive from Wikipedia. Derived dataset records retain CC-BY-SA-4.0."},
        "task": "Binary answerability detection only. No answer span is generated or evaluated; this is not the full SQuAD QA benchmark or its official answer-span EM/F1 score.",
        "input_boundary": "State is the original paragraph text. Fields contain original question text and the same fixed yes/no template. No is_impossible value, gold answer, plausible answer, answer offset, or evaluation label is included in runtime records.",
        "grouping": "One bounded worker per original paragraph, with all its original questions represented as q0...qN in source order. No paragraph concatenation, field truncation, fabricated questions, or extra neural coordinator.",
        "question_id_mapping": {"path": "evaluation/labels.jsonl", "field": "question_ids", "description": "Maps each per-paragraph q-key back to the original official question ID."},
        "template": {"path": str(args.template), "options": ["yes", "no"], "tuning": "Fixed before model inference; no example selection, threshold tuning, or prompt revision using labels."},
        "stages": [{"name": "feasibility", "selection": f"First {args.feasibility_paragraphs} paragraphs in official article/paragraph order, including every associated question; declared before model inference.", "tasks_path": f"tasks.{sample_name}.jsonl", "labels_path": f"evaluation/labels.{sample_name}.jsonl"}, {"name": "full_dev", "selection": "All official development paragraphs and all associated questions.", "tasks_path": "tasks.jsonl", "labels_path": "evaluation/labels.jsonl"}],
        "counts_and_baselines": summaries,
        "evaluation": {"target": "yes iff not is_impossible", "primary_metrics": ["question_accuracy", "balanced_accuracy"], "secondary_metrics": ["answerable_precision", "answerable_recall", "answerable_f1", "unanswerable_recall", "paragraph_exact_match", "invalid_output_rate"], "aggregation": "Question metrics count each original question once, rather than averaging paragraph accuracies. Balanced accuracy is the mean of answerable and unanswerable recall.", "baseline_scope": "Class counts and majority baselines are descriptive for each fixed subset; they are not learned model outputs.", "invalid_output_policy": "Report invalid output separately; never turn an invalid response into a default yes/no label or remove its workload from latency accounting."},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries))


if __name__ == "__main__":
    main()
