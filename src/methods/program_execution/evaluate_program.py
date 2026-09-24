import argparse
import json
import statistics
from pathlib import Path

from data.evaluate_bright import ranking_metrics


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--collections", type=Path, required=True)
    parser.add_argument("--relevance", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, required=True)
    args = parser.parse_args()
    collections = jsonl(args.collections)
    gold = {row["task_id"]: row["relevant_document_ids"] for row in jsonl(args.relevance)}
    settings = read(args.run / "protocol.json")["settings"]
    jobs = jsonl(Path(settings["jobs"]))
    assert [job["task_id"] for job in jobs] == [root["task_id"] for root in collections]
    for job, root in zip(jobs, collections):
        assert [(item["id"], item["input"]["document"]) for item in job["items"]] == [
            (document["document_id"], document["text"]) for document in root["candidates"]]
    results = {}
    for phase in ("warmup", "measured"):
        directory = args.run / phase
        compilation = read(directory / "compilation.json")
        summary = read(directory / "summary.json")
        modes, decisions = {}, {}
        for mode in settings["modes"]:
            outputs = read(directory / f"{mode}-outputs.json")
            assert set(outputs) <= {root["task_id"] for root in collections}
            rows = []
            for root in collections:
                identifier = root["task_id"]
                ranking = outputs.get(identifier, [])
                completed = identifier in outputs
                if completed:
                    assert len(ranking) == len(root["candidates"])
                    assert set(ranking) == {document["document_id"] for document in root["candidates"]}
                metrics = ranking_metrics(ranking, gold[identifier], args.cutoff)
                rows.append({"task_id": identifier, "completed": completed, "ndcg": metrics["ndcg"]})
            calls = read(directory / f"{mode}-calls.json")
            decisions[mode] = {(root, node["id"]): node for call in calls
                               for root, group in zip(call["root_ids"], call["result"]["groups"])
                               for node in group}
            modes[mode] = {**summary[mode], "macro_ndcg": statistics.mean(row["ndcg"] for row in rows),
                           "queries": rows, "valid_decisions": len(decisions[mode]),
                           "actual_peak_field_concurrency": [call["result"]["peak_field_concurrency"] for call in calls],
                           "computed_input_tokens": sum(call["result"]["computed_input_tokens"] for call in calls),
                           "logical_input_tokens": sum(call["result"]["logical_input_tokens"] for call in calls)}
        for mode, values in decisions.items():
            reference = decisions["independent"]
            assert values.keys() == reference.keys()
            modes[mode]["independent_choice_agreement"] = sum(node["choice"] == reference[key]["choice"] for key, node in values.items())
            modes[mode]["independent_max_probability_difference"] = max(
                (abs(a - b) for key, node in values.items() for a, b in zip(node["probabilities"], reference[key]["probabilities"])), default=None)
        results[phase] = {"programs": compilation["programs"], "failures": compilation["failures"], "modes": modes,
                          "compiler_batch_size": len(compilation["inputs"]),
                          "unique_program_values": len({json.dumps(program, sort_keys=True)
                                                        for program in compilation["programs"].values()})}
    prior = read(args.prior / "measured/evaluation.json")
    prior_summary = read(args.prior / "measured/finite-summary.json")
    results["comparison"] = {"prior_direct_probability_ndcg": prior["finite_probability_ranking"]["macro_ndcg_at_10"],
                             "prior_direct_binary_ndcg": prior["finite"]["macro_ndcg_at_10"],
                             "prior_direct_seconds": prior_summary["model_seconds"] + prior_summary["aggregation_seconds"],
                             "warm_measured_programs_equal": results["warmup"]["programs"] == results["measured"]["programs"],
                             "scope": "Development only. All assigned roots enter the quality denominator. Invalid programs have no ranking and zero gain. Scores use the original full relevance set in ideal DCG."}
    (args.run / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"measured": results["measured"]["modes"], "comparison": results["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
