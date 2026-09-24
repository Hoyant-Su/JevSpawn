from baselines.latentmas.io import read_jsonl, save
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from baselines.latentmas.adapter import SOURCE, method, task_item
from jev_spawn.infra.backend import Backend






def differences(batched, single, row):
    report = []
    for role, (batch_state, state) in enumerate(zip(batched, single)):
        pairs = [("hidden", batch_state["hidden"][row], state["hidden"][0])]
        for index, (batch_layer, layer) in enumerate(zip(batch_state["layers"], state["layers"])):
            for name, value in layer.items():
                batch_value = batch_layer[name][row]
                value = value[0]
                if name in ("keys", "values"):
                    batch_value = batch_value[:, batch_state["mask"][row].bool()]
                    value = value[:, state["mask"][0].bool()]
                pairs.append((f"layer{index}.{name}", batch_value, value))
        metrics = []
        for name, left, right in pairs:
            assert left.shape == right.shape
            delta = left.float() - right.float()
            metrics.append({"tensor": name, "max_absolute": float(delta.abs().max()),
                            "relative_l2": float(delta.norm() / right.float().norm().clamp_min(1e-12)),
                            "exact_fraction": float((left == right).float().mean())})
        report.append({"role": role, "valid_history_tokens": int(state["mask"].sum()), "tensors": metrics})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    native = json.loads(Path(config["native_config"]).read_text())
    assert native["batch_size"] == config["root_batch_size"] == config["task_count"] == 8
    tasks = read_jsonl(config["tasks"])[:config["task_count"]]
    items = [task_item(task) for task in tasks]
    save(args.output / "config.json", config)
    backend = Backend(native)
    runner = method(backend, config, tasks)
    torch.cuda.synchronize()
    alignment_start = time.perf_counter()
    alignment_before = torch.cuda.memory_allocated()
    runner.model._ensure_latent_realign_matrix(runner.model.model, backend.device, runner.args)
    torch.cuda.synchronize()
    save(args.output / "setup.json", {"model": {key: value for key, value in backend.metadata.items() if key != "controller"},
         "alignment_seconds": time.perf_counter() - alignment_start,
         "alignment_resident_bytes": torch.cuda.memory_allocated() - alignment_before,
         "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True),
         "upstream_revision": "9a9e4d331eb11430bd9e64754c6b252b06d73031",
         "source": str(SOURCE), "full_memory": True,
         "original_bodies": ["LatentMASMethod.run_batch", "ModelWrapper.generate_latent_batch",
                              "ModelWrapper._build_latent_realign_matrix", "ModelWrapper._apply_latent_realignment"],
         "interfaces": ["Qwen3.5 text trunk and complete hybrid cache transport",
                        "Historical per-example masks and positions with exact inactive recurrent-row restoration",
                        "AQuA fifth choice in the original MCQ prompt",
                        "Greedy256-token final decoding with native EOS and correct batched output slicing",
                        "Original function bodies loaded from AST without importing unused vLLM engines",
                        "Original unscored gold and solution fields are empty and evaluation is offline"]})
    snapshots = None
    stages = {}
    with torch.inference_mode():
        for stage in ("warm", "measured"):
            runner.model.reset(capture=stage == "warm")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            predictions = runner.run_batch(items)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            intervals = [value for row in runner.model.itl for value in row]
            payload = {"stage": stage, "seconds": elapsed, "batch_size": len(items),
                       "role_seconds": runner.model.role_times,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                       "restored_padding_rows": runner.model.model.restored_rows,
                       "output_ids": runner.model.output_ids, "per_sample_itl_seconds": runner.model.itl,
                       "itl_max_seconds": max(intervals), "itl_median_seconds": statistics.median(intervals),
                       "itl_over_100ms": sum(value >= .1 for value in intervals),
                       "calls": runner.model.model.records,
                       "predictions": [{"task_id": task["task_id"], **prediction}
                                       for task, prediction in zip(tasks, predictions)]}
            save(args.output / f"{stage}.json", payload)
            stages[stage] = payload
            if stage == "warm":
                snapshots = runner.model.snapshots
            print(json.dumps({"stage": stage, "seconds": elapsed, "itl_max_seconds": max(intervals)}), flush=True)
        qualification_start = time.perf_counter()
        parity = []
        for row, item in enumerate(items):
            runner.model.reset(capture=True)
            prediction = runner.run_batch([item])[0]
            parity.append({"task_id": tasks[row]["task_id"],
                           "prediction": prediction["prediction"], "raw_prediction": prediction["raw_prediction"],
                           "output_ids": runner.model.output_ids[0],
                           "batch_prediction": stages["measured"]["predictions"][row]["prediction"],
                           "cache_comparison": differences(snapshots, runner.model.snapshots, row)})
            save(args.output / "padding-qualification.json", {"completed": len(parity), "records": parity,
                                                              "seconds": time.perf_counter() - qualification_start})
            print(json.dumps({"qualification_row": row, "prediction": prediction["prediction"]}), flush=True)
    labels = {record["task_id"]: record["labels"]["q0"] for record in read_jsonl(config["labels"])}
    results = stages["measured"]["predictions"]
    accuracy = sum((record["prediction"] or "").upper() == labels[record["task_id"]] for record in results) / len(tasks)
    save(args.output / "summary.json", {"stage": "development_only", "tasks": len(tasks), "accuracy": accuracy,
         "valid_answers": sum((record["prediction"] or "").upper() in {option["id"] for option in task["fields"]["q0"]["options"]}
                              for record, task in zip(results, tasks)),
         "warm_measured_token_agreement": sum(a == b for a, b in zip(stages["warm"]["output_ids"], stages["measured"]["output_ids"])),
         "single_batch_answer_agreement": sum(record["prediction"] == record["batch_prediction"] for record in parity),
         "single_batch_token_agreement": sum(record["output_ids"] == stages["measured"]["output_ids"][i] for i, record in enumerate(parity)),
         "seconds": stages["measured"]["seconds"], "peak_allocated_bytes": stages["measured"]["peak_allocated_bytes"],
         "itl_max_seconds": stages["measured"]["itl_max_seconds"], "itl_over_100ms": stages["measured"]["itl_over_100ms"]})


if __name__ == "__main__":
    main()
