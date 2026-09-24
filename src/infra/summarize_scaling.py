import argparse
import json
from pathlib import Path

import numpy as np


def read_predictions(directory, summary, method, repeat):
    predictions, task_ids, records = {}, [], []
    for rank in range(summary["world_size"]):
        record = json.loads((directory / f"{method}-{repeat}-rank{rank}.json").read_text())
        records.append(record)
        for batch in record["results"]:
            task_ids.extend(batch["task_ids"])
            for field_name, field in batch["result"]["fields"].items():
                for task_id, choice in zip(batch["task_ids"], field["choices"], strict=True):
                    key = (task_id, field_name)
                    assert key not in predictions
                    predictions[key] = choice
    assert len(task_ids) == len(set(task_ids)) == len(summary["task_ids"])
    assert set(task_ids) == set(summary["task_ids"])
    return predictions, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    reference = json.loads((args.reference / "summary.json").read_text())
    labels = {row["task_id"]: row["labels"] for row in map(json.loads, args.labels.read_text().splitlines())}
    truth = {(identity, field): choice for identity in summary["task_ids"] for field, choice in labels[identity].items()}
    assert set(summary["task_ids"]) == set(reference["task_ids"])
    assert len({row["backend"]["gpu_uuid"] for row in summary["backends"]}) == summary["world_size"]
    assert sorted(int(row["backend"]["cuda_visible_devices"]) for row in summary["backends"]) == sorted(summary["gpus"])
    methods = list(dict.fromkeys(row["method"] for row in summary["trials"]))
    repeats = sorted({row["repeat"] for row in summary["trials"]})
    first, _ = read_predictions(args.run, summary, "independent", 0)
    result = {"world_size": summary["world_size"], "gpus": summary["gpus"],
              "batch_policy": summary["batch_policy"], "batch_size_per_gpu": summary["batch_size_per_gpu"],
              "tasks": len(summary["task_ids"]), "field_decisions": len(truth),
              "gpu_uuids": [row["backend"]["gpu_uuid"] for row in summary["backends"]],
              "reference": str(args.reference), "methods": {}}
    for method in methods:
        rows = [row for row in summary["trials"] if row["method"] == method]
        assert len(rows) == len(repeats)
        predictions, records = read_predictions(args.run, summary, method, 0)
        reference_predictions, _ = read_predictions(args.reference, reference, method, 0)
        assert predictions.keys() == truth.keys() == reference_predictions.keys()
        interval_ms, repeated_differences = [], []
        for repeat in repeats:
            repeated, repeat_records = read_predictions(args.run, summary, method, repeat)
            assert repeated.keys() == predictions.keys()
            repeated_differences.append(sum(repeated[key] != predictions[key] for key in predictions))
            if method == "compact_json":
                interval_ms.extend(value * 1000 for row in repeat_records for batch in row["results"]
                                   for sequence in batch["result"]["decode"] for value in sequence["inter_token_seconds"])
        result["methods"][method] = {
            "dispatch_seconds": [row["dispatch_seconds"] for row in rows],
            "dispatch_seconds_median": float(np.median([row["dispatch_seconds"] for row in rows])),
            "compute_span_seconds_median": float(np.median([row["compute_span_seconds"] for row in rows])),
            "peak_allocated_gib_per_gpu": np.max([row["peak_allocated_gib_per_gpu"] for row in rows], axis=0).tolist(),
            "actual_tasks_per_rank": [row["tasks"] for row in records],
            "actual_batches_per_rank": [row["batches"] for row in records],
            "accuracy": sum(predictions[key] == truth[key] for key in truth) / len(truth),
            "field_disagreements_vs_independent": sum(predictions[key] != first[key] for key in first),
            "field_disagreements_vs_reference": sum(predictions[key] != reference_predictions[key] for key in predictions),
            "field_disagreements_across_repeats": repeated_differences,
            "inter_token_ms": {"median": float(np.median(interval_ms)),
                               "p95": float(np.percentile(interval_ms, 95)), "maximum": max(interval_ms)} if interval_ms else None,
            "inter_token_at_least_100ms": sum(value >= 100 for value in interval_ms),
        }
    shared, _ = read_predictions(args.run, summary, "shared", 0)
    streamed, _ = read_predictions(args.run, summary, "streamed", 0)
    result["shared_streamed_disagreements"] = sum(shared[key] != streamed[key] for key in shared)
    (args.run / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
