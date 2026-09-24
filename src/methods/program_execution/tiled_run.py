import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from data.evaluate_bright import ranking_metrics
from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER
from methods.decision_program.compiler import compile_programs
from methods.decision_program.run import execute
from methods.generated_schema.run import measured_call, write
from methods.program_execution import score_grouped
from jev_spawn.infra.prompts import load_prompt, resolve_prompts


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def memory(backend):
    torch.cuda.synchronize(backend.device)
    return {"allocated_bytes": torch.cuda.memory_allocated(backend.device),
            "reserved_bytes": torch.cuda.memory_reserved(backend.device)}


def flattened(calls):
    return {(root, node["id"]): node for call in calls
            for root, group in zip(call["root_ids"], call["result"]["groups"]) for node in group}


def parity(values, reference):
    assert values.keys() == reference.keys()
    return {"decisions": len(values), "choice_agreement": sum(
        value["choice"] == reference[key]["choice"] for key, value in values.items()),
        "maximum_probability_difference": max(abs(a - b) for key, value in values.items()
                                               for a, b in zip(value["probabilities"], reference[key]["probabilities"])),
        "maximum_logit_difference": max(abs(a - b) for key, value in values.items()
                                         for a, b in zip(value["option_logits"], reference[key]["option_logits"]))}


def evaluate_rankings(collections, gold, rankings, cutoff):
    rows = []
    for root in collections:
        identifier = root["task_id"]
        ranked = rankings[identifier]
        assert len(ranked) == len(root["candidates"])
        assert set(ranked) == {document["document_id"] for document in root["candidates"]}
        rows.append({"task_id": identifier, "ndcg": ranking_metrics(ranked, gold[identifier], cutoff)["ndcg"]})
    return {"queries": rows, "macro_ndcg": statistics.mean(row["ndcg"] for row in rows)}


def direct(backend, collections, settings, directory):
    prompts = load_prompt(settings["direct_prompt_schema"])
    groups = [[{"id": document["document_id"],
                "state": json.dumps({"query": root["query"], "document_id": document["document_id"],
                                     "document": document["text"], "worker_messages": []}, ensure_ascii=False),
                "question": prompts["relevance_question"], "options": prompts["relevance_options"]}
               for document in root["candidates"]] for root in collections]
    summaries = {}
    for mode in settings["direct_modes"]:
        call = measured_call(backend, lambda: score_grouped(backend, groups, mode))
        call["root_ids"] = [root["task_id"] for root in collections]
        started = time.perf_counter()
        outputs = {root["task_id"]: [node["id"] for node in sorted(
            group, key=lambda node: -node["probabilities"][node["option_ids"].index("yes")])]
            for root, group in zip(collections, call["result"]["groups"])}
        reduction_seconds = time.perf_counter() - started
        write(directory / f"direct-{mode}-calls.json", [call])
        write(directory / f"direct-{mode}-outputs.json", outputs)
        summaries[mode] = {"execution_seconds": call["elapsed_seconds"] + reduction_seconds,
                           "reduction_seconds": reduction_seconds,
                           "peak_allocated_bytes": call["peak_allocated_bytes"],
                           "peak_reserved_bytes": call["peak_reserved_bytes"],
                           "root_batch_size": call["result"]["root_batch_size"],
                           "peak_field_concurrency": call["result"]["peak_field_concurrency"]}
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = resolve_prompts(read(args.settings))
    config = resolve_prompts(read(settings["native_config"]))
    jobs, collections = jsonl(settings["jobs"]), jsonl(settings["collections"])
    frozen = read(settings["frozen_compilation"])
    assert len(jobs) == settings["job_count"] == settings["batch_size"] == config["batch_size"] == 8
    assert config["branch_batch_size"] == settings["live_leaf_cap"] == 128
    assert [job["task_id"] for job in jobs] == [root["task_id"] for root in collections]
    assert all(len(job["items"]) == settings["items_per_root"] for job in jobs)
    for job, root in zip(jobs, collections):
        assert [(item["id"], item["input"]["document"]) for item in job["items"]] == [
            (document["document_id"], document["text"]) for document in root["candidates"]]
        assert len({item["id"] for item in job["items"]}) == len(job["items"])
    assert len({json.dumps(value, sort_keys=True) for value in frozen["programs"].values()}) == 1
    CONTROLLER["option_template"] = settings["option_template"]
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", {"settings": settings, "native": config,
                                         "controller": dict(CONTROLLER), "frozen_programs": frozen["programs"]})
    backend = Backend(config)
    write(args.output / "model-load-memory.json", memory(backend))
    for phase in ("warmup", "measured"):
        directory = args.output / phase
        directory.mkdir()
        if settings["compilation"] == "execute_and_match":
            before_compiler = memory(backend)
            compiled = compile_programs(backend, jobs, settings)
            write(directory / "compiler-memory.json", {"before": before_compiler, "after": memory(backend),
                  "peak_allocated_bytes": compiled["generation"]["peak_allocated_bytes"],
                  "peak_reserved_bytes": compiled["generation"]["peak_reserved_bytes"]})
            write(directory / "compilation.json", compiled)
            assert not compiled["failures"] and compiled["programs"] == frozen["programs"], "Compiler must reproduce the frozen S5 program."
            compiler_seconds = compiled["generation"]["elapsed_seconds"]
            programs = compiled["programs"]
        else:
            assert settings["compilation"] == "frozen_component_qualification"
            programs, compiler_seconds = frozen["programs"], 0
        summary = execute(backend, jobs, programs, settings, directory)
        for value in summary.values():
            value.update(compiler_seconds=compiler_seconds,
                         compiler_executed=settings["compilation"] == "execute_and_match",
                         total_seconds=compiler_seconds + value["execution_seconds"])
        if settings["direct_modes"]:
            summary["direct"] = direct(backend, collections, settings, directory)
        write(directory / "summary.json", summary)
        print(json.dumps({"phase": phase, "summary": summary}), flush=True)
    gold = {root["task_id"]: root["relevant_document_ids"] for root in jsonl(settings["relevance"])}
    directory = args.output / "measured"
    current = {mode: flattened(read(directory / f"{mode}-calls.json")) for mode in settings["modes"]}
    metrics = {}
    for mode, decisions in current.items():
        metrics[mode] = evaluate_rankings(collections, gold, read(directory / f"{mode}-outputs.json"), settings["ranking_cutoff"])
        metrics[mode]["versus_tiled_independent"] = parity(decisions, current["tiled_independent"])
        metrics[mode]["warm_measured"] = parity(decisions, flattened(read(args.output / "warmup" / f"{mode}-calls.json")))
        if settings["compare_frozen32"]:
            prior = flattened(read(Path(settings["frozen_compilation"]).parent / "streamed-calls.json"))
            metrics[mode]["versus_frozen_S5"] = parity(decisions, prior)
    metrics["direct"] = {mode: evaluate_rankings(collections, gold,
                          read(directory / f"direct-{mode}-outputs.json"), settings["ranking_cutoff"])
                         for mode in settings["direct_modes"]}
    write(args.output / "evaluation.json", metrics)


if __name__ == "__main__":
    main()
