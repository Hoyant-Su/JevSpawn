import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from jev_spawn.infra.backend import Backend
from jev_spawn.schema import CONTROLLER, controller_prompts
from methods.generated_map.run import evaluate
from methods.generated_schema.run import measured_call, write
from methods.program_execution import score_grouped


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def ranking(collections, predictions, probabilistic):
    outputs = {}
    for root, group in zip(collections, predictions):
        if probabilistic:
            order = sorted(range(len(group)), key=lambda i: (-group[i]["restricted_p_yes"], root["candidates"][i]["retrieval_rank"]))
        else:
            order = sorted(range(len(group)), key=lambda i: (group[i]["choice"] != "yes", root["candidates"][i]["retrieval_rank"]))
        outputs[root["task_id"]] = [group[i]["document_id"] for i in order]
    return outputs


def comparison(left, right):
    assert [(x["task_id"], x["document_id"]) for x in left] == [(x["task_id"], x["document_id"]) for x in right]
    return {"count": len(left), "choice_agreement": sum(x["choice"] == y["choice"] for x, y in zip(left, right)),
            "maximum_probability_difference": max(abs(x["restricted_p_yes"] - y["restricted_p_yes"]) for x, y in zip(left, right)),
            "disagreements": [{"task_id": x["task_id"], "document_id": x["document_id"],
                               "left": x["choice"], "right": y["choice"],
                               "left_p_yes": x["restricted_p_yes"], "right_p_yes": y["restricted_p_yes"]}
                              for x, y in zip(left, right) if x["choice"] != y["choice"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    config = json.loads(Path(settings["native_config"]).read_text())
    source = Path(settings["prior_s3e"])
    prior = json.loads((source / "protocol.json").read_text())
    prompts = json.loads(Path(settings["prompt_schema"]).read_text())
    collections = jsonl(settings["collections"])
    assert len(collections) == settings["query_count"] == config["batch_size"]
    assert all(len(row["candidates"]) == settings["candidate_count"] for row in collections)
    CONTROLLER["option_template"] = prompts["option_template"]
    assert dict(CONTROLLER) == prior["controller"]
    field = {"question": prompts["relevance_question"], "options": prompts["relevance_options"]}
    assert field == prior["field"]
    groups = [[{"id": document["document_id"],
                "state": json.dumps({"query": row["query"], "document_id": document["document_id"],
                                     "document": document["text"], "worker_messages": []}, ensure_ascii=False),
                **field} for document in row["candidates"]] for row in collections]
    prior_calls = json.loads((source / "measured/finite-calls.json").read_text())
    prior_items = [item for call in prior_calls for item in call["items"]]
    assert [node["state"] for group in groups for node in group] == [item["state"] for item in prior_items]
    for group in groups:
        for node in group:
            canonical = [{**option, "id": f"option{index}"} for index, option in enumerate(node["options"])]
            assert controller_prompts([node["state"]], node["question"], node["options"], ["A", "B"], CONTROLLER["output_instruction"]) == controller_prompts([node["state"]], node["question"], canonical, ["A", "B"], CONTROLLER["output_instruction"])
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", {"settings": settings, "native": config, "controller": dict(CONTROLLER),
          "groups": groups, "task_ids": [root["task_id"] for root in collections],
          "exact_s3e_prompt_equality": True, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
          "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"], text=True),
          "comparison_scope": "Identical256 decisions and prompts. Independent and shared execute256 fields together, streamed uses128-field tiles. Old S3e executes8 fields per call. No concurrency-matched memory claim."})
    backend = Backend(config)
    outputs = {}
    for phase in ("warmup", "measured"):
        directory = args.output / phase
        directory.mkdir()
        outputs[phase] = {}
        for mode in settings["methods"]:
            call = measured_call(backend, lambda: score_grouped(backend, groups, mode))
            write(directory / f"{mode}-call.json", call)
            predictions = [[{"task_id": root["task_id"], "document_id": node["id"], "choice": node["choice"],
                             "restricted_p_yes": node["probabilities"][node["option_ids"].index("yes")],
                             "option_ids": node["option_ids"], "probabilities": node["probabilities"],
                             "option_logits": node["option_logits"]} for node in group]
                           for root, group in zip(collections, call["result"]["groups"])]
            outputs[phase][mode] = [node for group in predictions for node in group]
            started = time.perf_counter()
            binary, probability = ranking(collections, predictions, False), ranking(collections, predictions, True)
            reduction_seconds = time.perf_counter() - started
            write(directory / f"{mode}-predictions.json", outputs[phase][mode])
            write(directory / f"{mode}-rankings.json", {"binary": binary, "probability": probability})
            summary = {key: call[key] for key in ("elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes")}
            summary.update({key: call["result"][key] for key in ("root_batch_size", "logical_field_count", "peak_field_concurrency", "branch_tile_size", "logical_input_tokens", "computed_input_tokens", "padded_input_tokens", "prefix_tokens", "timings")})
            summary["reduction_seconds"] = reduction_seconds
            summary["output_tokens"] = 0
            summary["itl_applicability"] = "No autoregressive decoding"
            write(directory / f"{mode}-summary.json", summary)
            print(json.dumps({"phase": phase, "mode": mode, **summary}), flush=True)
    relevance = jsonl(settings["relevance"])
    prior_predictions = json.loads((source / "measured/finite-predictions.json").read_text())
    prior_rankings = {kind: json.loads((source / f"measured/finite-{kind}-rankings.json").read_text())
                      for kind in ("binary", "probability")}
    metrics = {}
    for mode in settings["methods"]:
        ranks = json.loads((args.output / "measured" / f"{mode}-rankings.json").read_text())
        metrics[mode] = {kind: evaluate(collections, relevance, values, settings["ranking_cutoff"])
                         for kind, values in ranks.items()}
        metrics[mode]["versus_grouped_independent"] = comparison(outputs["measured"][mode], outputs["measured"]["independent"])
        metrics[mode]["versus_prior_s3e"] = comparison(outputs["measured"][mode], prior_predictions)
        metrics[mode]["prior_ranking_exact_roots"] = {
            kind: sum(values[root] == prior_rankings[kind][root] for root in values)
            for kind, values in ranks.items()}
        metrics[mode]["warm_measured"] = comparison(outputs["warmup"][mode], outputs["measured"][mode])
    write(args.output / "evaluation.json", metrics)


if __name__ == "__main__":
    main()
