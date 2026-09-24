import argparse
import json
import re
import statistics
import time
from pathlib import Path

import torch

from baselines.latentmas.adapter import method
from baselines.latentmas.io import read_jsonl, save
from baselines.latentmas.adapter import task_item
from jev_spawn.infra.backend import Backend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    native = json.loads(Path(config["native_config"]).read_text())
    assert native["batch_size"] == config["root_batch_size"] == config["task_count"]
    tasks = read_jsonl(config["tasks"])[:config["task_count"]]
    items = [task_item(task) for task in tasks]
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "config.json", config)
    backend = Backend(native)
    runner = method(backend, config, tasks)
    torch.cuda.synchronize()
    started = time.perf_counter()
    before = torch.cuda.memory_allocated()
    runner.model._ensure_latent_realign_matrix(runner.model.model, backend.device, runner.args)
    torch.cuda.synchronize()
    save(args.output / "setup.json", {
        "model": {key: value for key, value in backend.metadata.items() if key != "controller"},
        "alignment_seconds": time.perf_counter() - started,
        "alignment_resident_bytes": torch.cuda.memory_allocated() - before,
        "upstream_revision": "9a9e4d331eb11430bd9e64754c6b252b06d73031",
        "roles": [agent.name for agent in runner.agents],
        "choice_protocol": "Last boxed content must be an original option ID and agree with the original extractor. Numeric fallback outputs remain invalid.",
        "numerical_diagnostic": config["numerical_diagnostic"],
        "timing_scope": "Full four-role dispatch after separate alignment construction. Warmup is excluded from measured metrics."})
    stages = {}
    with torch.inference_mode():
        for stage in ("warm", "measured"):
            runner.model.reset(capture=False)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            predictions = runner.run_batch(items)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            records = []
            for task, prediction, tokens in zip(tasks, predictions, runner.model.output_ids):
                boxes = re.findall(r"\\boxed\{([^}]*)\}", prediction["raw_prediction"])
                options = {option["id"] for option in task["fields"]["q0"]["options"]}
                boxed = boxes[-1].strip() if boxes else None
                upstream = (prediction["prediction"] or "").upper()
                answer = boxed if boxed in options and boxed == upstream else None
                records.append({"task_id": task["task_id"], **prediction, "answer": answer,
                                "output_tokens": len(tokens),
                                "truncated": len(tokens) == config["final_tokens"] and tokens[-1] not in backend.eos_ids})
            intervals = [value for row in runner.model.itl for value in row]
            payload = {"stage": stage, "seconds": elapsed, "batch_size": len(items),
                       "role_seconds": runner.model.role_times,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "output_ids": runner.model.output_ids, "per_sample_itl_seconds": runner.model.itl,
                       "itl_max_seconds": max(intervals), "itl_median_seconds": statistics.median(intervals),
                       "itl_over_100ms": sum(value >= .1 for value in intervals),
                       "forward_calls": runner.model.model.records, "predictions": records}
            save(args.output / f"{stage}.json", payload)
            stages[stage] = payload
            print(json.dumps({"stage": stage, "seconds": elapsed, "valid": sum(r["answer"] is not None for r in records),
                              "truncated": sum(r["truncated"] for r in records), "max_itl": max(intervals)}), flush=True)
    labels = {row["task_id"]: row["labels"]["q0"] for row in read_jsonl(config["labels"])}
    measured = stages["measured"]
    results = measured["predictions"]
    save(args.output / "summary.json", {
        "scope": "AQuA development only", "tasks": len(tasks), "final_token_budget": config["final_tokens"],
        "latent_roles": len(runner.agents) - 1, "latent_steps_per_role": config["latent_steps"],
        "valid": sum(row["answer"] is not None for row in results),
        "correct": sum(row["answer"] == labels[row["task_id"]] for row in results),
        "truncated": sum(row["truncated"] for row in results),
        "generated_tokens": sum(row["output_tokens"] for row in results),
        "warm_measured_token_agreement": sum(a == b for a, b in zip(stages["warm"]["output_ids"], measured["output_ids"])),
        "warm_measured_answer_agreement": sum(a["answer"] == b["answer"] for a, b in zip(stages["warm"]["predictions"], results)),
        "numerical_diagnostic": config["numerical_diagnostic"],
        **{key: measured[key] for key in ("seconds", "role_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
                                        "itl_max_seconds", "itl_median_seconds", "itl_over_100ms")}})


if __name__ == "__main__":
    main()
