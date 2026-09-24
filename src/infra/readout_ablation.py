import argparse
import json
import string
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jev_spawn.infra.backend import Backend


def timed(operation, device):
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, {"cuda_ms": start.elapsed_time(end), "wall_ms": (time.perf_counter() - wall) * 1000}


def statistics(rows):
    return {name: {"median": float(np.median([row[name] for row in rows])),
                   "p95": float(np.percentile([row[name] for row in rows], 95))}
            for name in ["cuda_ms", "wall_ms"]}


def compare(left, right):
    difference = (left.float() - right.float()).abs()
    return {"logit_elements": left.numel(), "unequal_logit_elements": int((difference != 0).sum()),
            "max_absolute_logit_difference": float(difference.max()),
            "mean_absolute_logit_difference": float(difference.mean()),
            "argmax_disagreements": int((left.argmax(-1) != right.argmax(-1)).sum()),
            "decisions": left.shape[0]}


@torch.inference_mode()
def run(config):
    torch.set_float32_matmul_precision(config["float32_matmul_precision"])
    rows = [json.loads(line) for line in Path(config["tasks_path"]).read_text().splitlines()]
    signature = lambda task: tuple((name, tuple(option["id"] for option in field["options"]))
                                   for name, field in task["fields"].items())
    first_signature = signature(rows[0])
    tasks = [task for task in rows if signature(task) == first_signature][:config["batch_size"]]
    assert len(tasks) == config["batch_size"]
    backend = Backend(config)
    captured = []

    def capture(module, arguments, output):
        captured.append(output[:, -1].detach().clone())

    hook = backend.model.model.language_model.norm.register_forward_hook(capture)
    source = backend.score_fields([task["state"] for task in tasks],
                                  [task["fields"] for task in tasks], mode="independent")
    hook.remove()
    assert len(captured) == 1
    hidden = captured.pop()
    fields = tasks[0]["fields"]
    counts = [len(field["options"]) for field in fields.values()]
    assert len(set(counts)) == 1
    labels = list(string.ascii_uppercase[:counts[0]])
    ids = backend.tokenizer(labels, add_special_tokens=False)["input_ids"]
    assert all(len(row) == 1 for row in ids)
    candidate_ids = torch.tensor([row[0] for row in ids], device=backend.device)
    assert hidden.shape[0] == len(tasks) * len(fields)
    weight = backend.model.lm_head.weight
    result = {
        "scope": "Readout component only; actual last-token activations from independent full-prompt inference.",
        "config": config, "backend": backend.metadata,
        "task_ids": [task["task_id"] for task in tasks], "field_names": list(fields),
        "root_batch_size": len(tasks), "fields_per_root": len(fields),
        "hidden_shape": list(hidden.shape), "full_weight_shape": list(weight.shape),
        "candidate_labels": labels, "candidate_token_ids": candidate_ids.tolist(),
        "candidate_counts_per_field": counts,
        "source_input_tokens": source["logical_input_tokens"],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "precisions": {},
    }
    predictions = {}
    for precision in config["precisions"]:
        dtype = getattr(torch, precision)
        torch.cuda.synchronize(backend.device)
        resident_before = torch.cuda.memory_allocated(backend.device)
        typed_hidden, hidden_setup = timed(lambda: hidden.to(dtype), backend.device)
        full_weight, full_setup = timed(lambda: weight.to(dtype), backend.device)
        full_resident_extra = torch.cuda.memory_allocated(backend.device) - resident_before
        before_candidate = torch.cuda.memory_allocated(backend.device)
        candidate_weight, candidate_setup = timed(lambda: weight.index_select(0, candidate_ids).to(dtype), backend.device)
        candidate_resident_extra = torch.cuda.memory_allocated(backend.device) - before_candidate
        operations = {
            "candidate_rows": lambda: F.linear(typed_hidden, candidate_weight),
            "full_vocabulary_then_gather": lambda: F.linear(typed_hidden, full_weight).index_select(1, candidate_ids),
        }
        for operation in operations.values():
            for _ in range(config["warmup_repeats"]):
                operation()
        torch.cuda.synchronize(backend.device)
        timings = {name: [] for name in operations}
        for repeat in range(config["repeats"]):
            order = list(operations) if repeat % 2 == 0 else list(reversed(operations))
            for name in order:
                output, measurement = timed(operations[name], backend.device)
                timings[name].append(measurement)
                del output
        peaks, outputs = {}, {}
        for name, operation in operations.items():
            resident = torch.cuda.memory_allocated(backend.device)
            torch.cuda.reset_peak_memory_stats(backend.device)
            outputs[name] = operation()
            torch.cuda.synchronize(backend.device)
            peaks[name] = torch.cuda.max_memory_allocated(backend.device) - resident
        result["precisions"][precision] = {
            "setup": {"hidden_cast": hidden_setup, "full_head_cast": full_setup,
                      "candidate_selection_and_cast": candidate_setup},
            "full_head_and_hidden_resident_extra_bytes": full_resident_extra,
            "candidate_head_resident_extra_bytes": candidate_resident_extra,
            "incremental_projection_peak_bytes": peaks,
            "timing_summary": {name: statistics(rows) for name, rows in timings.items()},
            "timing_samples": timings,
            "equivalence": compare(outputs["candidate_rows"], outputs["full_vocabulary_then_gather"]),
            "candidate_logits": outputs["candidate_rows"].float().cpu().tolist(),
        }
        predictions[precision] = outputs["candidate_rows"].float().cpu()
        del outputs, operations, typed_hidden, full_weight, candidate_weight
    result["cross_precision"] = compare(predictions["bfloat16"], predictions["float32"])
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({precision: value["timing_summary"] for precision, value in result["precisions"].items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()))
