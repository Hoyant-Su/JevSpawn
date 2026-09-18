"""Combine completed GPU shards without counting interrupted work as results."""

import argparse
import collections
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    directory = Path(config["run_dir"])
    runs = [json.loads((directory / f"run-{rank}.json").read_text()) for rank in range(config["world_size"])]
    done = [json.loads((directory / f"done-{rank}.json").read_text()) for rank in range(config["world_size"])]
    chunks = [json.loads(path.read_text()) for path in sorted(directory.glob("chunk-*.json"))]
    states = [state for chunk in chunks for state in chunk["states"]]
    assert len(states) == config["task_count"]
    expected = {task for run in runs for task in run["task_ids"]}
    assert len({state["task_id"] for state in states}) == len(states)
    assert {state["task_id"] for state in states} == expected
    plans = [json.loads(path.read_text()) for path in sorted(directory.glob("plan-*.json"))]
    events = sorted([event for part in plans + chunks for event in part["events"]], key=lambda event: (event["timestamp"], event["event_id"]))
    running = queued = peak_running = peak_queued = 0
    for event in events:
        kind = event["type"]
        queued += int(kind == "agent_spawned") - int(kind == "agent_started")
        running += int(kind == "agent_started") - int(kind == "agent_completed")
        peak_running, peak_queued = max(peak_running, running), max(peak_queued, queued)
        assert min(running, queued) >= 0
    assert running == queued == 0
    usage = [row for state in states for row in state["usage"]]
    batches = [dict(event["payload"], role=event["role"]) for event in events if event["type"] == "batch_completed"]
    summary = {
        "run_id": config["run_id"], "model": "Qwen/Qwen3.5-4B", "mode": config["mode"],
        "task_count": len(states), "datasets": dict(collections.Counter(state["dataset"] for state in states)),
        "schedule": config.get("schedule", "chunk"),
        "agents_spawned": len(usage), "agents_completed": len(usage),
        "agent_definition": "One bounded role worker with its own task, input, model generation and output; model weights are shared.",
        "roles": dict(collections.Counter(row["role"] for row in usage)),
        "peak_running_agents": peak_running, "peak_queued_agents": peak_queued,
        "gpu_replicas": config["world_size"], "configured_batch_size_per_gpu": config["batch_size"],
        "observed_batch_sizes": sorted({batch["batch_size"] for batch in batches}),
        "worker_input_tokens": sum(row["input_tokens"] for row in usage),
        "worker_output_tokens": sum(row["output_tokens"] for row in usage),
        "truncated_worker_outputs": sum(row["truncated"] for row in usage),
        "controller_input_tokens": sum(batch["input_tokens"] for batch in batches if batch["role"] == "controller"),
        "controller_output_tokens": sum(batch["output_tokens"] for batch in batches if batch["role"] == "controller"),
        "controller_gpu_seconds": sum(batch["duration_seconds"] for batch in batches if batch["role"] == "controller"),
        "worker_gpu_seconds": sum(batch["duration_seconds"] for batch in batches if batch["role"] != "controller"),
        "elapsed_seconds": events[-1]["timestamp"] - events[0]["timestamp"],
        "time_scope": "First recorded routing/dispatch/worker event through last task completion. Includes scheduling and host work; excludes initial model loading. Includes gaps after interruption.",
        "probability_status": "Uncalibrated conditional option probabilities.",
        "evaluation_status": "Not evaluated by this collector; see separate evaluation summary.",
    }
    if 'routing_mode' in config:
        summary['routing_mode'] = config['routing_mode']
    if any('cuda_memory' in rank for rank in done):
        assert all('cuda_memory' in rank for rank in done), 'Every rank must report CUDA memory.'
        assert len({rank['memory_scope'] for rank in done}) == 1
        summary['cuda_memory'] = {
            'per_gpu': [{'gpu_id': rank, **record['cuda_memory']} for rank, record in enumerate(done)],
            'scope': done[0]['memory_scope'],
            'measurement': 'PyTorch CUDA allocator allocated/reserved bytes; not whole-device nvidia-smi usage.',
            'peak_reset': 'Immediately after model loading on each rank.',
            'coordinator_extra_model_replicas': 0,
        }
    if plans:
        summary["spawn_elapsed_seconds"] = max(e["timestamp"] for e in events if e["type"] == "agent_spawned") - min(e["timestamp"] for e in events if e["type"] in {"routing_started", "dispatch_started"})
        summary["controller_mode"] = config["controller_mode"]
        summary["job"] = config["job"]
    if "spawn_policy" in config:
        summary["spawn_policy"] = config["spawn_policy"]
        summary["decision_counts"] = {
            stage: dict(collections.Counter(decision["choice"] for state in states
                                             for decision in state["decisions"] if decision["stage"] == stage))
            for stage in sorted({decision["stage"] for state in states for decision in state["decisions"]})
        }
        summary["initial_agents"] = len(states)
        summary["additional_agents"] = len(usage) - len(states)
        summary["controller_computed_input_tokens"] = sum(batch.get("computed_input_tokens", batch["input_tokens"])
                                                         for batch in batches if batch["role"] == "controller")
    metadata = {key: summary[key] for key in ("run_id", "model", "mode", "agent_definition")}
    events.insert(0, {"event_id": "start", "timestamp": events[0]["timestamp"], "type": "run_started", "payload": metadata})
    events.append({"event_id": "end", "timestamp": events[-1]["timestamp"], "type": "run_completed", "payload": summary})
    (directory / "events.jsonl").write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events))
    for filename, field in (("solutions.jsonl", "solution"), ("initial_solutions.jsonl", "initial_solution")):
        (directory / filename).write_text("".join(json.dumps({"task_id": state["task_id"], "solution": state[field]}, ensure_ascii=False) + "\n" for state in states))
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "run.json").write_text(json.dumps({'config': config, 'task_ids': [state['task_id'] for state in states]}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
