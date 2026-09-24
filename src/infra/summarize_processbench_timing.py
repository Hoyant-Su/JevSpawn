import argparse
import json
from pathlib import Path

import numpy as np


def distribution(values):
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "maximum": max(values)} if values else None


def predictions(records, method, repeat):
    return {(task_id, name): choice for row in records
            if row["method"] == method and row["repeat"] == repeat
            for name, field in row["result"]["fields"].items()
            for task_id, choice in zip(row["task_ids"], field["choices"], strict=True)}


def summarize(directory):
    metadata = json.loads((directory / "run.json").read_text())
    records = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
    required = {(repeat, batch, method) for repeat in range(metadata["repeats"])
                for batch in range(len(metadata["batch_task_ids"])) for method in metadata["methods"]}
    assert len(records) == len(required)
    assert {(row["repeat"], row["batch"], row["method"]) for row in records} == required
    reference = predictions(records, "independent", 0)
    result = {"timing_scope": metadata["timing_scope"], "repeats": metadata["repeats"],
              "root_tasks": sum(map(len, metadata["batch_task_ids"])),
              "field_decisions": len(reference), "methods": {}}
    for method in metadata["methods"]:
        selected = [row for row in records if row["method"] == method]
        total_seconds = [sum(row["elapsed_seconds"] for row in selected if row["repeat"] == repeat)
                         for repeat in range(metadata["repeats"])]
        intervals = [interval * 1000 for row in selected for sequence in row["decode"]
                     for interval in sequence["inter_token_seconds"]]
        ttft = [sequence["ttft_seconds"] * 1000 for row in selected for sequence in row["decode"]
                if sequence["ttft_seconds"] is not None]
        first = predictions(records, method, 0)
        assert first.keys() == reference.keys()
        repeated_differences = []
        for repeat in range(1, metadata["repeats"]):
            choices = predictions(records, method, repeat)
            assert choices.keys() == first.keys()
            repeated_differences.append(sum(choices[key] != first[key] for key in first))
        result["methods"][method] = {
            "repeat_total_seconds": total_seconds, "total_seconds": distribution(total_seconds),
            "ttft_ms": distribution(ttft), "inter_token_ms": distribution(intervals),
            "inter_token_samples": len(intervals), "inter_token_at_least_100ms": sum(value >= 100 for value in intervals),
            "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in selected) / 2**30,
            "peak_reserved_gib": max(row["peak_reserved_bytes"] for row in selected) / 2**30,
            "field_disagreements_vs_independent_repeat0": sum(first[key] != reference[key] for key in first),
            "repeat_field_disagreements_vs_repeat0": repeated_differences,
            "output_tokens_per_repeat": [sum(sum(row["result"]["output_tokens"]) for row in selected
                                              if row["repeat"] == repeat) for repeat in range(metadata["repeats"])],
        }
    (directory / "processbench_timing.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.run)
