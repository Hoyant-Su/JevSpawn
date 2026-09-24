import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from build_processbench_table import FORMULATIONS, METHODS


def scores(matches, erroneous, weights):
    error_count, correct_count = weights @ erroneous, weights @ ~erroneous
    assert np.all(error_count > 0) and np.all(correct_count > 0)
    error_accuracy = (weights @ (matches & erroneous)) / error_count
    correct_accuracy = (weights @ (matches & ~erroneous)) / correct_count
    assert np.all(error_accuracy + correct_accuracy > 0)
    return {"first_error_accuracy": (weights @ matches) / weights.sum(axis=-1),
            "processbench_f1": 2 * error_accuracy * correct_accuracy / (error_accuracy + correct_accuracy)}


def interval(estimate, samples, confidence):
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail], method="linear")
    return {"estimate": float(estimate * 100), "lower": float(lower * 100), "upper": float(upper * 100)}


def main():
    parser = argparse.ArgumentParser()
    for name in ["runs", "source", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resamples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--confidence", type=float, required=True)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    identities = ["processbench_gsm8k/" + row["id"] for row in source]
    assert len(identities) == len(set(identities))
    gold = np.array([row["label"] for row in source])
    erroneous = gold != -1
    problems = list(dict.fromkeys(row["problem"] for row in source))
    cluster_lookup = {problem: index for index, problem in enumerate(problems)}
    membership = np.array([cluster_lookup[row["problem"]] for row in source])
    sampled = np.random.default_rng(args.seed).integers(0, len(problems), (args.resamples, len(problems)))
    cluster_weights = np.zeros((args.resamples, len(problems)), dtype=np.int32)
    np.add.at(cluster_weights, (np.arange(args.resamples)[:, None], sampled), 1)
    weights = cluster_weights[:, membership]
    result = {
        "method": "Paired problem-cluster percentile bootstrap",
        "resamples": args.resamples, "seed": args.seed, "confidence": args.confidence,
        "metric_scale": "percent; paired differences in percentage points",
        "solutions": len(source), "problem_clusters": len(problems),
        "cluster_size_distribution": dict(sorted(Counter(Counter(membership).values()).items())),
        "resampled_solution_count_range": [int(weights.sum(axis=1).min()), int(weights.sum(axis=1).max())],
        "pairing": "The same sampled original-problem clusters and their complete solution sets are used for every method and both formulations.",
        "scope": "Pointwise sampling intervals over benchmark problems; execution repeats are not additional observations. No labels are assigned to steps after the first error.",
        "sources": {"source": str(args.source), "runs": str(args.runs)},
        "formulations": {},
    }
    for formulation in FORMULATIONS:
        quality = json.loads((args.runs / formulation / "processbench_metrics.json").read_text())
        point, draws = {}, {}
        for method in METHODS:
            values = quality["methods"][method]
            assert set(values["predicted_first_errors"]) == set(identities)
            predicted = np.array([values["predicted_first_errors"][identity] for identity in identities])
            matches = predicted == gold
            point[method] = scores(matches, erroneous, np.ones(len(source), dtype=np.int32))
            draws[method] = scores(matches, erroneous, weights)
            assert all(np.isclose(point[method][metric], values[metric]) for metric in point[method])
        result["formulations"][formulation] = {
            "methods": {method: {metric: interval(point[method][metric], draws[method][metric], args.confidence)
                                 for metric in point[method]} for method in METHODS},
            "paired_differences": {
                "streamed_minus_" + reference: {
                    metric: interval(point["streamed"][metric] - point[reference][metric],
                                     draws["streamed"][metric] - draws[reference][metric], args.confidence)
                    for metric in point["streamed"]}
                for reference in ["independent", "compact_json"]},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({form: value["paired_differences"] for form, value in result["formulations"].items()}, indent=2))


if __name__ == "__main__":
    main()
