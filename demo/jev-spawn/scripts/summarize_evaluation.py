import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize frozen evaluation verdicts and completed worker usage by official dataset split.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    evaluation_path = args.run_dir / "evaluation.jsonl"
    results = [json.loads(line) for line in evaluation_path.read_text().splitlines()]
    manifest = json.loads(args.manifest.read_text())
    run = json.loads((args.run_dir / "summary.json").read_text())
    states = [state for path in sorted(args.run_dir.glob("chunk-*.json")) for state in json.loads(path.read_text())["states"]]
    results_by_id = {row["task_id"]: row for row in results}
    assert len(results_by_id) == len(results) == len(states)
    assert set(results_by_id) == {state["task_id"] for state in states}
    groups = {
        "all_tasks": set(results_by_id),
        "humaneval": {row["task_id"] for row in results if row["dataset"] == "humaneval"},
        "mbpp_all": {row["task_id"] for row in results if row["dataset"] == "mbpp"},
        **{f"mbpp_{split}": set(ids) for split, ids in manifest["mbpp_split_task_ids"].items()},
    }
    summary = {}
    for name, ids in groups.items():
        selected = [state for state in states if state["task_id"] in ids]
        if not selected:
            continue
        counts = collections.Counter(results_by_id[state["task_id"]]["status"] for state in selected)
        usage = [row for state in selected for row in state["usage"]]
        truncated_ids = {state["task_id"] for state in selected if any(row["truncated"] for row in state["usage"])}
        summary[name] = {
            "tasks": len(selected), "passed": counts["passed"], "failed": counts["failed"], "timeout": counts["timeout"],
            "pass_rate": counts["passed"] / len(selected),
            "workers_completed": len(usage),
            "worker_input_tokens": sum(row["input_tokens"] for row in usage),
            "worker_output_tokens": sum(row["output_tokens"] for row in usage),
            "truncated_worker_outputs": sum(row["truncated"] for row in usage),
            "tasks_with_truncation": len(truncated_ids),
            "tasks_with_truncation_passed": sum(results_by_id[task_id]["status"] == "passed" for task_id in truncated_ids),
        }
    summary["run_id"] = run["run_id"]
    summary["recorded_timing"] = {key: run[key] for key in ["elapsed_seconds", "controller_gpu_seconds", "worker_gpu_seconds", "time_scope"]}
    summary["timing_interpretation"] = "Observed run timings are not matched-order or matched-cache measurements; they do not establish a causal end-to-end speedup."
    summary["protocol"] = {key: manifest["protocol"][key] for key in ["humaneval", "mbpp", "workload", "evaluation"]}
    (args.run_dir / "evaluation.splits.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
