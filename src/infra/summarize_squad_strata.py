import argparse
import csv
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


METHOD_NAMES = {"independent": "Independent", "shared": "Shared", "streamed": "Streamed", "compact_json": "Compact JSON"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / "run.json").read_text())
    tasks = {row["task_id"]: row for row in map(json.loads, Path(metadata["tasks"]).read_text().splitlines())}
    trials = [json.loads(line) for line in (args.run / "trials.jsonl").read_text().splitlines()]
    indexed = {(row["batch"], row["method"], row["repeat"]): row for row in trials}
    batches = metadata["batch_task_ids"]
    methods, repeats = metadata["methods"], metadata["repeats"]
    assert len(indexed) == len(trials) == len(batches) * len(methods) * repeats
    identities = [identity for batch in batches for identity in batch]
    assert len(set(identities)) == len(identities)
    tokenizer = AutoTokenizer.from_pretrained(metadata["config"]["model_path"], local_files_only=True)
    token_ids = tokenizer([tasks[identity]["state"] for identity in identities], add_special_tokens=False)["input_ids"]
    lengths = dict(zip(identities, map(len, token_ids), strict=True))
    features = []
    for batch, ids in enumerate(batches):
        counts = {len(tasks[identity]["fields"]) for identity in ids}
        assert len(counts) == 1
        count = counts.pop()
        for method in methods:
            for repeat in range(repeats):
                trial = indexed[batch, method, repeat]
                assert trial["task_ids"] == ids
                assert trial["batch_size"] == len(ids) and trial["field_count"] == count
        features.append({"batch": batch, "task_ids": ids, "roots": len(ids), "fields_per_root": count,
                         "decisions": len(ids) * count, "max_context_tokens": max(lengths[identity] for identity in ids),
                         "mean_context_tokens": float(np.mean([lengths[identity] for identity in ids]))})
    rows, strata = [], {}
    for feature in ["fields_per_root", "max_context_tokens"]:
        values = np.array([batch[feature] for batch in features])
        quantiles = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        boundaries = np.unique(quantiles)
        groups = np.searchsorted(boundaries, values, side="left")
        strata[feature] = {"quantile_probabilities": [0.25, 0.5, 0.75], "quantiles": quantiles.tolist(),
                           "unique_boundaries": boundaries.tolist(), "interval_rule": "Upper boundary inclusive; ties remain together.",
                           "quantile_unit": "Original measured batch, equally weighted", "bins": []}
        for group in sorted(set(groups)):
            selected = [batch for batch, assignment in zip(features, groups, strict=True) if assignment == group]
            ids = [batch["batch"] for batch in selected]
            lower = float(boundaries[group - 1]) if group else None
            upper = float(boundaries[group]) if group < len(boundaries) else None
            label = f"({lower if lower is not None else '-inf'}, {upper if upper is not None else '+inf'}]"
            strata[feature]["bins"].append({"stratum": label, "lower_exclusive": lower, "upper_inclusive": upper, "batch_ids": ids})
            times = {method: np.array([[indexed[batch, method, repeat]["elapsed_seconds"] for repeat in range(repeats)]
                                      for batch in ids]) for method in methods}
            for method in methods:
                row = {"feature": feature, "stratum": label, "method": method,
                       "batches": len(ids), "roots": sum(batch["roots"] for batch in selected),
                       "decisions": sum(batch["decisions"] for batch in selected),
                       "actual_batch_size_min": min(batch["roots"] for batch in selected),
                       "actual_batch_size_max": max(batch["roots"] for batch in selected),
                       "field_count_min": min(batch["fields_per_root"] for batch in selected),
                       "field_count_max": max(batch["fields_per_root"] for batch in selected),
                       "batch_max_context_tokens_min": min(batch["max_context_tokens"] for batch in selected),
                       "batch_max_context_tokens_max": max(batch["max_context_tokens"] for batch in selected),
                       "batch_max_context_tokens_median": float(np.median([batch["max_context_tokens"] for batch in selected])),
                       "warm_total_seconds_median": float(np.median(times[method].sum(axis=0))),
                       "batch_seconds_median": float(np.median(np.median(times[method], axis=1))),
                       "peak_allocated_gib": max(indexed[batch, method, repeat]["peak_allocated_bytes"]
                                                 for batch in ids for repeat in range(repeats)) / 2**30}
                for baseline in ["independent", "compact_json"]:
                    paired = np.median(times[baseline] / times[method], axis=1)
                    row.update({f"paired_batch_speedup_vs_{baseline}_median": float(np.median(paired)),
                                f"paired_batch_speedup_vs_{baseline}_q25": float(np.quantile(paired, 0.25)),
                                f"paired_batch_speedup_vs_{baseline}_q75": float(np.quantile(paired, 0.75)),
                                f"total_speedup_vs_{baseline}": float(np.median(times[baseline].sum(axis=0)) / row["warm_total_seconds_median"])})
                rows.append(row)
    result = {"source": str(args.run), "roots": len(identities), "batches": len(batches),
              "decisions": sum(batch["decisions"] for batch in features), "repeats": repeats,
              "definitions": {"scope": "Descriptive strata of the original workload; field count, context length and actual batch size covary. Not a causal field-count or context-length ablation.",
                              "context_length": "Raw original paragraph token count using the frozen model tokenizer, add_special_tokens=False; each batch is stratified by its maximum context length. Questions and chat formatting excluded.",
                              "field_count": "Original question count per paragraph, identical within each schema-compatible measured batch; no fields dropped or duplicated.",
                              "time": metadata["timing_scope"],
                              "warm_total_seconds_median": "Sum elapsed seconds over the same selected batches within each repeat, then median across the three repeat totals.",
                              "paired_batch_speedup": "Baseline/method elapsed seconds for each matched batch and repeat; median across repeats within each batch, then median and quartiles across unique batches.",
                              "memory": "Maximum measured PyTorch allocated GPU memory, including weights, over all selected batches and repeats.",
                              "replication": "Repetitions measure execution variability; they are not additional independent tasks or quality samples. No quality inference or causal confidence interval is computed."},
              "config": {key: metadata["config"][key] for key in ["model_path", "dtype", "batch_size", "branch_batch_size", "seed", "controller_max_new_tokens"]},
              "strata": strata, "batch_features": features, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    latex = [r"\begin{table*}[t]", r"\centering\small",
             r"\caption{Descriptive SQuAD workload strata. Each method processes the same batches in each stratum; field count, context length and batch size covary.}",
             r"\label{tab:squad-strata}", r"\resizebox{\linewidth}{!}{", r"\begin{tabular}{llrrrrrrr}", r"\toprule",
             r"Stratum & Method & Batches & Roots & Decisions & Time (s) & Peak (GiB) & vs. Indep. & vs. JSON \\", r"\midrule"]
    for feature in strata:
        title = "Original fields per root" if feature == "fields_per_root" else "Maximum raw context tokens per batch"
        latex.append(r"\multicolumn{9}{l}{\textit{" + title + r"}} \\")
        for row in [row for row in rows if row["feature"] == feature]:
            label = row["stratum"].replace("-inf", r"$-\infty$").replace("+inf", r"$+\infty$")
            cells = [label, METHOD_NAMES[row["method"]], str(row["batches"]), str(row["roots"]), str(row["decisions"]),
                     f"{row['warm_total_seconds_median']:.2f}", f"{row['peak_allocated_gib']:.2f}",
                     f"{row['paired_batch_speedup_vs_independent_median']:.2f}", f"{row['paired_batch_speedup_vs_compact_json_median']:.2f}"]
            latex.append(" & ".join(cells) + r" \\")
        latex.append(r"\midrule")
    latex[-1] = r"\bottomrule"
    latex.extend([r"\end{tabular}", "}", r"\par\smallskip\footnotesize Time is the median of three sums of synchronized warm model-call times, including tokenization and parsing, excluding loading, warmup, garbage collection, queue waiting and file I/O. Ratios are median paired batch speedups, after taking the median across repeats within each batch. Peak memory includes weights. Quantile boundaries use equally weighted original batches and retain ties together; intervals are upper-inclusive. Repeats do not increase the sample count.", r"\end{table*}"])
    args.output.with_suffix(".tex").write_text("\n".join(latex) + "\n")
    print(json.dumps({"batches": len(batches), "roots": len(identities), "strata": strata}, indent=2))


if __name__ == "__main__":
    main()
