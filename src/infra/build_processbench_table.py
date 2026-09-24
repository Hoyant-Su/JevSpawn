import argparse
import csv
import json
from pathlib import Path

from summarize_processbench_timing import predictions


FORMULATIONS = {"per_step": "Per-step decisions", "first_error": "First-error selection"}
METHODS = {"independent": "Independent", "shared": "Shared", "streamed": "Streamed", "compact_json": "Compact JSON"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    args = parser.parse_args()
    uncertainty = json.loads(args.uncertainty.read_text())
    rows, agreement, sources = [], {}, {}
    for formulation in FORMULATIONS:
        directory = args.runs / formulation
        quality = json.loads((directory / "processbench_metrics.json").read_text())
        timing = json.loads((directory / "processbench_timing.json").read_text())
        metadata = json.loads((directory / "run.json").read_text())
        trials = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
        shared = predictions(trials, "shared", 0)
        streamed = predictions(trials, "streamed", 0)
        agreement[formulation] = sum(shared[key] != streamed[key] for key in shared)
        settings = {key: metadata["config"][key] for key in ["model_path", "dtype", "attention", "kernel",
                    "max_input_tokens", "seed", "cpu_threads", "batch_size", "branch_batch_size", "controller_max_new_tokens"]}
        sources[formulation] = {"directory": str(directory), "config": settings,
                                "backend": metadata["backend"], "repeats": timing["repeats"]}
        for method in METHODS:
            q, t = quality["methods"][method], timing["methods"][method]
            rows.append({
                "formulation": formulation, "method": method,
                "solutions": q["solutions"], "field_decisions": q["decisions"],
                "error_solutions": q["error_solutions"], "correct_solutions": q["correct_solutions"],
                "labelled_steps": q["labelled_steps"],
                "first_error_accuracy_pct": q["first_error_accuracy"] * 100,
                "error_accuracy_pct": q["error_accuracy"] * 100,
                "correct_accuracy_pct": q["correct_accuracy"] * 100,
                "processbench_f1_pct": q["processbench_f1"] * 100,
                "local_step_accuracy_pct": q["local_step_accuracy"] * 100 if q["local_step_accuracy"] is not None else None,
                "warm_total_seconds_median": t["total_seconds"]["median"],
                "peak_allocated_gib": t["peak_allocated_gib"],
                "inter_token_p95_ms": t["inter_token_ms"]["p95"] if t["inter_token_ms"] is not None else None,
                "inter_token_max_ms": t["inter_token_ms"]["maximum"] if t["inter_token_ms"] is not None else None,
                "inter_token_at_least_100ms": t["inter_token_at_least_100ms"],
                "field_disagreements_vs_independent": t["field_disagreements_vs_independent_repeat0"],
                "repeat_field_disagreements": sum(t["repeat_field_disagreements_vs_repeat0"]),
            })
            for metric in ["first_error_accuracy", "processbench_f1"]:
                bounds = uncertainty["formulations"][formulation]["methods"][method][metric]
                assert abs(bounds["estimate"] - rows[-1][metric + "_pct"]) < 1e-10
                rows[-1].update({metric + "_ci_lower_pct": bounds["lower"], metric + "_ci_upper_pct": bounds["upper"]})
    indexed = {(row["formulation"], row["method"]): row for row in rows}
    findings = {}
    for formulation in FORMULATIONS:
        independent, streamed, generated = [indexed[formulation, method] for method in ["independent", "streamed", "compact_json"]]
        findings[formulation] = {
            "streamed_speedup_vs_independent": independent["warm_total_seconds_median"] / streamed["warm_total_seconds_median"],
            "streamed_speedup_vs_compact_json": generated["warm_total_seconds_median"] / streamed["warm_total_seconds_median"],
            "streamed_allocated_memory_reduction_vs_independent_pct": 100 * (1 - streamed["peak_allocated_gib"] / independent["peak_allocated_gib"]),
            "streamed_minus_compact_json_f1_points": streamed["processbench_f1_pct"] - generated["processbench_f1_pct"],
            "shared_streamed_field_disagreements": agreement[formulation],
        }
    result = {
        "dataset": "ProcessBench GSM8K", "quality_metric_scale": "percent",
        "definitions": {
            "processbench_f1": "Harmonic mean of first-error accuracy on erroneous solutions and all-correct accuracy on correct solutions.",
            "first_error_accuracy": "Exact first-error index accuracy over all solutions, including the no-error label.",
            "local_step_accuracy": "Per-step formulation only; excludes steps after the first annotated error, whose local labels are unknown.",
            "time": "Median of three complete warm model-call totals; includes tokenization and parsing, excludes loading, warmup, garbage collection, queue waiting and file I/O.",
            "memory": "Maximum PyTorch allocated GPU memory across all measured trials; includes model weights.",
            "null": "Metric does not apply to this formulation or zero-token categorical readout.",
        },
        "sources": sources, "rows": rows, "findings": findings,
        "uncertainty": {"source": str(args.uncertainty), **{key: uncertainty[key] for key in
                         ["method", "resamples", "seed", "confidence", "problem_clusters", "solutions", "scope"]}},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    local, direct = indexed["per_step", "independent"], indexed["first_error", "independent"]
    source_config = sources["per_step"]["config"]
    caption = (f"ProcessBench GSM8K: quality and inference cost on {local['solutions']} solutions. "
               f"Accuracy and F1 are percentages. All methods use frozen {Path(source_config['model_path']).name} "
               f"with a per-GPU root batch cap of {source_config['batch_size']} on one H100.")
    notes = (f"Error acc. measures exact error location on {local['error_solutions']} erroneous solutions; "
             f"correct acc. measures the no-error decision on {local['correct_solutions']} correct solutions. "
             f"F1 is their harmonic mean. Per-step inference returns {local['field_decisions']:,} decisions; "
             f"local accuracy uses only {local['labelled_steps']:,} annotated steps. "
             f"First-error selection returns {direct['field_decisions']:,} decisions. "
             f"Time is the median of {sources['per_step']['repeats']} warm complete model-call totals. "
             "Peak memory is allocated GPU memory including weights. Compact JSON has higher F1 point estimates in both "
             "formulations; streaming adds overhead in the single-field formulation. "
             f"Brackets give {uncertainty['confidence'] * 100:g}\\% percentile intervals from "
             f"{uncertainty['resamples']:,} paired bootstrap samples of {uncertainty['problem_clusters']} "
             "original-problem clusters, keeping associated solutions together.")
    latex = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\caption{" + caption + "}",
             r"\label{tab:processbench}", r"\resizebox{\linewidth}{!}{", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r"Method & First-error acc. & Error acc. & Correct acc. & F1 [" + f"{uncertainty['confidence'] * 100:g}" + r"\% CI] & Local step acc. & Time (s) & Peak (GiB) \\",
             r"\midrule"]
    for formulation, title in FORMULATIONS.items():
        latex.append(r"\multicolumn{8}{l}{\textit{" + title + r"}} \\")
        for method, name in METHODS.items():
            row = indexed[formulation, method]
            columns = [name] + [f"{row[key]:.2f}" for key in ["first_error_accuracy_pct", "error_accuracy_pct", "correct_accuracy_pct", "processbench_f1_pct"]]
            columns[-1] += f" [{row['processbench_f1_ci_lower_pct']:.2f}, {row['processbench_f1_ci_upper_pct']:.2f}]"
            columns.append("---" if row["local_step_accuracy_pct"] is None else f"{row['local_step_accuracy_pct']:.2f}")
            columns.extend(f"{row[key]:.2f}" for key in ["warm_total_seconds_median", "peak_allocated_gib"])
            latex.append(" & ".join(columns) + r" \\")
        latex.append(r"\midrule")
    latex[-1] = r"\bottomrule"
    latex.extend([r"\end{tabular}", "}", r"\par\smallskip", r"\begin{minipage}{\textwidth}\footnotesize",
                  notes,
                  r"\end{minipage}", r"\end{table*}"])
    args.output.with_suffix(".tex").write_text("\n".join(latex) + "\n")
    intervals = [r"\begin{table*}[t]", r"\centering\small",
                 r"\caption{Paired problem-cluster bootstrap uncertainty for ProcessBench GSM8K. Values are estimates [" + f"{uncertainty['confidence'] * 100:g}" + r"\% percentile intervals]. Individual scores are percentages; paired differences are percentage points.}",
                 r"\label{tab:processbench_uncertainty}", r"\begin{tabular}{lrr}", r"\toprule",
                 r"Method or paired comparison & First-error accuracy & ProcessBench F1 \\", r"\midrule"]
    for formulation, title in FORMULATIONS.items():
        intervals.append(r"\multicolumn{3}{l}{\textit{" + title + r"}} \\")
        block = uncertainty["formulations"][formulation]
        entries = [(METHODS[method], metrics) for method, metrics in block["methods"].items()]
        entries += [("Streamed minus " + METHODS[name.removeprefix("streamed_minus_")], metrics)
                    for name, metrics in block["paired_differences"].items()]
        for name, metrics in entries:
            values = [name] + [f"{value['estimate']:.2f} [{value['lower']:.2f}, {value['upper']:.2f}]"
                               for value in [metrics['first_error_accuracy'], metrics['processbench_f1']]]
            intervals.append(" & ".join(values) + r" \\")
        intervals.append(r"\midrule")
    intervals[-1] = r"\bottomrule"
    intervals.extend([r"\end{tabular}", r"\end{table*}"])
    args.uncertainty.with_suffix(".tex").write_text("\n".join(intervals) + "\n")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
