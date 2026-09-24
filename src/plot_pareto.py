import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


METHODS = {
    "independent": ("Independent", "#0072B2", "s"),
    "shared": ("Shared cache", "#E69F00", "o"),
    "streamed": ("Streamed", "#009E73", "D"),
    "compact_json": ("Compact JSON", "#CC79A7", "^"),
}


def read_run(root, name):
    analysis = json.loads((root / "runs" / name / "analysis.json").read_text())
    metadata = json.loads((root / "runs" / name / "run.json").read_text())
    return analysis, metadata


def point(method, row):
    return {"method": method, "seconds": row["median_warm_seconds"],
            "seconds_range": [min(row["warm_seconds"]), max(row["warm_seconds"])],
            "repeat_seconds": row["warm_seconds"], "accuracy_pct": 100 * row["accuracy"],
            "accuracy_ci95_pct": [100 * value for value in row["accuracy_ci95"]],
            "peak_allocated_gib": row["peak_allocated_gib"], "roots": row["tasks"], "decisions": row["decisions"]}


def frontier(points, ykey, maximize):
    direction = -1 if maximize else 1
    return sorted([index for index, p in enumerate(points) if not any(
        q["seconds"] <= p["seconds"] and direction * q[ykey] <= direction * p[ykey]
        and (q["seconds"] < p["seconds"] or direction * q[ykey] < direction * p[ykey])
        for q in points)], key=lambda index: points[index]["seconds"])


def scatter(ax, p, ykey, size, uncertainty=False):
    _, color, marker = METHODS[p["method"]]
    x, y = p["seconds"], p[ykey]
    lo, hi = p["seconds_range"]
    yerr = np.array([[y - p["accuracy_ci95_pct"][0]], [p["accuracy_ci95_pct"][1] - y]]) if uncertainty else None
    ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], yerr=yerr, fmt="none", color=color,
                linewidth=1.0, capsize=2.5, alpha=0.72, zorder=2)
    ax.scatter(x, y, s=size, marker=marker, facecolors="none" if p["method"] == "shared" else color,
               edgecolors=color, linewidths=1.6, zorder=4 if p["method"] == "streamed" else 3)


def draw_frontier(ax, points, indices, ykey):
    chosen = [points[index] for index in indices]
    ax.plot([p["seconds"] for p in chosen], [p[ykey] for p in chosen], "--", color="#30343b",
            linewidth=1.0, zorder=1)
    for p in chosen:
        ax.scatter(p["seconds"], p[ykey], s=190, facecolors="none", edgecolors="#30343b", linewidths=0.6, zorder=1)


def decorate(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color="#e5e7eb", linewidth=0.6, zorder=0)
    ax.tick_params(length=3)


def legend(fig):
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=5,
                      markerfacecolor="none" if method == "shared" else color, label=name)
               for method, (name, color, marker) in METHODS.items()]
    handles.append(Line2D([], [], color="#30343b", linestyle="--", linewidth=1, label="Empirical frontier"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.52, 1.0), ncol=5,
               frameon=False, columnspacing=1.05, handletextpad=0.45)


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=300)
    fig.savefig(output / f"{name}_pagewidth.png", dpi=110)
    Image.open(output / f"{name}_pagewidth.png").convert("L").save(output / f"{name}_grayscale.png")
    plt.close(fig)


def quality_figure(root, output):
    fig = plt.figure(figsize=(7.2, 4.75))
    grid = fig.add_gridspec(2, 2, height_ratios=[2.8, 1.1], left=0.085, right=0.985,
                           bottom=0.12, top=0.865, wspace=0.29, hspace=0.33)
    artifacts = []
    for column, (name, title) in enumerate([("squad-main-001", "A  SQuAD2 answerability"),
                                           ("native-main-001/race_middle", "B  RACE-middle")]):
        analysis, metadata = read_run(root, name)
        points = [point(method, analysis["methods"][method]) for method in METHODS]
        selected = frontier(points, "accuracy_pct", True)
        ax = fig.add_subplot(grid[0, column])
        for p in points:
            scatter(ax, p, "accuracy_pct", 7 * p["peak_allocated_gib"], uncertainty=True)
        draw_frontier(ax, points, selected, "accuracy_pct")
        decorate(ax, title, r"Warm workload time, s  $\downarrow$", r"Accuracy, percent  $\uparrow$")
        ax.set_xlim(0, max(p["seconds_range"][1] for p in points) * 1.15)
        lower = min(p["accuracy_ci95_pct"][0] for p in points)
        upper = max(p["accuracy_ci95_pct"][1] for p in points)
        ax.set_ylim(lower - 0.14 * (upper - lower), upper + 0.28 * (upper - lower))
        ax.text(0.025, 0.955, f"{points[0]['roots']:,} paragraphs, {points[0]['decisions']:,} decisions",
                transform=ax.transAxes, va="top", fontsize=7, color="#535b65")
        near = "Shared and streamed overlap" if column == 0 else "Shared and streamed nearly overlap"
        ax.annotate(near, xy=(points[2]["seconds"], points[2]["accuracy_pct"]), xycoords="data",
                    xytext=(0.20, 0.08), textcoords="axes fraction", fontsize=6.7, color="#535b65",
                    arrowprops={"arrowstyle": "-", "color": "#7b8490", "linewidth": 0.6},
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        table_ax = fig.add_subplot(grid[1, column])
        table_ax.axis("off")
        table = table_ax.table(cellText=[[METHODS[p["method"]][0], f"{p['seconds']:.2f}",
                                          f"{p['accuracy_pct']:.2f}", f"{p['peak_allocated_gib']:.2f}"] for p in points],
                              colLabels=["Method", "Time, s", "Acc., %", "GiB"],
                              colWidths=[0.43, 0.20, 0.22, 0.15], cellLoc="right", loc="upper center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.10)
        for (row, col), cell in table.get_celld().items():
            cell.set_linewidth(0)
            if row == 0:
                cell.set_facecolor("#edf0f3")
                cell.set_text_props(weight="bold")
            if col == 0:
                cell.set_text_props(ha="left")
                if row:
                    cell.get_text().set_color(METHODS[points[row - 1]["method"]][1])
        artifacts.append({"source": f"runs/{name}", "points": points, "frontier_indices": selected,
                          "uncertainty": analysis["uncertainty"], "timing_scope": metadata["timing_scope"]})
    legend(fig)
    fig.text(0.085, 0.027, "Bars show 95 percent paragraph cluster intervals and time ranges across three repeats.\n"
             "Marker area scales with peak GiB. The frontier compares point estimates, not statistical superiority.",
             fontsize=7, color="#535b65", linespacing=1.5)
    save(fig, output, "quality_latency_pareto")
    return artifacts


def cost_figure(root, output):
    specs = [(batch, 128, f"squad-batch{batch}-001") for batch in [1, 4, 16]]
    specs += [(8, tile, f"squad-tile{tile}-001") for tile in [16, 64, 128]]
    all_points, reference_ids = [], None
    for batch, tile, name in specs:
        analysis, metadata = read_run(root, name)
        ids = {identity for group in metadata["batch_task_ids"] for identity in group}
        if reference_ids is None:
            reference_ids = ids
        assert ids == reference_ids
        assert metadata["config"]["batch_size"] == batch and metadata["config"]["branch_batch_size"] == tile
        for method in METHODS:
            p = point(method, analysis["methods"][method])
            p.update(batch=batch, tile=tile, source=f"runs/{name}")
            all_points.append(p)
    panels = [[p for p in all_points if p["tile"] == 128],
              [p for p in all_points if p["batch"] == 8 and (p["method"] == "streamed" or p["tile"] == 128)]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.27, top=0.82, wspace=0.27)
    offsets = {"independent": (5, 5), "shared": (5, 5), "streamed": (5, -12), "compact_json": (5, 5)}
    for panel, (ax, points) in enumerate(zip(axes, panels, strict=True)):
        selected = frontier(points, "peak_allocated_gib", False)
        for p in points:
            scatter(ax, p, "peak_allocated_gib", 38)
            offset = offsets[p["method"]]
            label = f"B{p['batch']}" if panel == 0 else f"T{p['tile']}" if p["method"] == "streamed" else METHODS[p["method"]][0]
            if panel == 0 and p["method"] == "streamed":
                offset = {1: (5, -13), 4: (2, -14), 8: (32, 1), 16: (14, 18)}[p["batch"]]
            if panel == 0 and p["method"] == "independent" and p["batch"] == 1:
                offset = (3, -12)
            if panel == 0 and p["method"] == "shared" and p["batch"] == 16:
                offset = (15, -5)
            if panel == 1:
                offset = {"independent": (8, -2), "shared": (8, -3), "compact_json": (-52, 9)}.get(p["method"],
                         {16: (35, -6), 64: (10, -18), 128: (15, 12)}[p["tile"]])
            ax.annotate(label, (p["seconds"], p["peak_allocated_gib"]), xytext=offset,
                        textcoords="offset points", color=METHODS[p["method"]][1], fontsize=7,
                        arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": METHODS[p["method"]][1]})
        draw_frontier(ax, points, selected, "peak_allocated_gib")
        if panel == 0:
            for method in METHODS:
                path = sorted([p for p in points if p["method"] == method], key=lambda p: p["batch"])
                ax.plot([p["seconds"] for p in path], [p["peak_allocated_gib"] for p in path],
                        color=METHODS[method][1], linewidth=0.7, alpha=0.45, zorder=0)
            ax.set_xscale("log")
            ax.set_xticks([7, 10, 20, 50, 100], ["7", "10", "20", "50", "100"])
            ax.set_xlim(6, 125)
        else:
            ax.set_xlim(5.5, 24)
            streamed = {p["tile"]: p for p in points if p["method"] == "streamed"}
            memory_change = 100 * (streamed[16]["peak_allocated_gib"] / streamed[128]["peak_allocated_gib"] - 1)
            time_change = 100 * (streamed[16]["seconds"] / streamed[128]["seconds"] - 1)
            ax.text(0.54, 0.74, f"T128 to T16\n{abs(memory_change):.1f} percent less memory\n{time_change:.1f} percent more time",
                    transform=ax.transAxes, fontsize=7, linespacing=1.5, color="#535b65")
        ax.set_ylim(min(p["peak_allocated_gib"] for p in points) - 1.3, max(p["peak_allocated_gib"] for p in points) + 2)
        title = "C  Root batch capacity, tile 128" if panel == 0 else "D  Streamed branch tile, batch 8"
        decorate(ax, title, r"Warm workload time, s  $\downarrow$", r"Peak allocated memory, GiB  $\downarrow$")
        ax.text(0.03, 0.96, "128 paragraphs, 1,274 decisions", transform=ax.transAxes, va="top", fontsize=7, color="#535b65")
    legend(fig)
    fig.text(0.085, 0.035, "B denotes root batch capacity. T denotes streamed branch tile. Each point processes the same 128 paragraphs.\n"
             "Panel C connects batch settings. Panel D uses T128 controls for other methods. Bars span three repeats.\n"
             "Dashed frontiers minimize time and memory over plotted estimates. Accuracy is not a frontier objective.",
             fontsize=7, color="#535b65", linespacing=1.5)
    save(fig, output, "batch_tile_cost_frontier")
    return {"all_measurements": all_points,
            "panels": [{"points": points, "frontier_indices": frontier(points, "peak_allocated_gib", False)} for points in panels]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    output = args.root / "manuscript/figures"
    output.mkdir(parents=True, exist_ok=True)
    quality = quality_figure(args.root, output)
    costs = cost_figure(args.root, output)
    quality_caption = ("Quality and inference cost on SQuAD2 answerability and RACE middle. Each point represents one method on the complete workload shown. "
                       "Horizontal bars span the minimum and maximum summed warm inference time across three repetitions. Vertical bars give 95 percent "
                       "bootstrap intervals for individual method accuracy, resampling original paragraphs and retaining their questions together. Marker area "
                       "is proportional to peak allocated GPU memory including weights. Rings and dashed lines identify the empirical nondominated set "
                       "among plotted accuracy and time estimates. They do not indicate statistical superiority. Coincident estimates retain their original coordinates.")
    cost_caption = ("Inference time and peak allocated memory on the same 128 SQuAD2 paragraphs and 1274 questions. Panel C varies root batch capacity "
                    "at branch tile 128. Thin colored paths connect settings within each method. Panel D varies the streamed branch tile at root batch eight "
                    "and displays the tile 128 controls for the other methods. The tile setting changes only streamed execution. Horizontal bars span three "
                    "complete workload repetitions. Rings and dashed lines identify the empirical nondominated set for time and memory over the plotted "
                    "points. Accuracy varies slightly across batch settings and is not an objective of these cost frontiers.")
    (output / "quality_latency_pareto.json").write_text(json.dumps({"caption": quality_caption, "panels": quality}, indent=2) + "\n")
    (output / "batch_tile_cost_frontier.json").write_text(json.dumps({"caption": cost_caption, **costs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
