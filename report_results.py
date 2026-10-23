"""Generate figures, descriptive tables, and derived quantities from raw results."""

from __future__ import annotations
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER / "results"
FIGURES = PAPER / "figures"
NAMES = ("jvrust", "jvhybrid", "jvcpp", "jvx", "jvs", "jvc", "scipy")
LABELS = ("JVRUST", "JVHYBRID", "JVCPP", "JVX", "JVS", "JVC", "SciPy")
COLORS = ("#176ca4", "#78519a", "#2a9d8f", "#d17c26", "#32905d", "#878787", "#bd4141")
COLOR = dict(zip(NAMES, COLORS))
LABEL = dict(zip(NAMES, LABELS))


def read(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name):
    fig.savefig(FIGURES / (name + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def batch_order(solvers):
    """Loop baseline, then each batch API serial before threaded, then SciPy."""
    threaded = sorted(name for name in solvers if name.startswith("jvrust_batch_t") and name != "jvrust_batch_t1")
    suffix = threaded[0].rsplit("_t", 1)[1]
    return (["jvrust_loop"] + [f"{family}_batch_t{t}" for family in ("jvrust", "jvhybrid", "jvcpp", "jvx")
                               for t in ("1", suffix)] + ["scipy_loop"])


def batch_label(solver):
    family, _, rest = solver.partition("_")
    if rest == "loop":
        return f"{LABEL[family]}, loop"
    workers = rest.removeprefix("batch_t")
    return f"{LABEL[family]}, {workers} worker" + ("s" if workers != "1" else "")


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "svg.fonttype": "none",
    })
    single = read("single_summary.csv")
    batch = read("batch_summary.csv")
    memory = json.loads((RESULTS / "memory.json").read_text())
    environment = json.loads((RESULTS / "environment.json").read_text())
    spec = importlib.util.spec_from_file_location("paper_experiments", PAPER / "run_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.initialize()
    cases = module.case_specs()
    by_case = {
        item["case"]: {row["solver"]: row for row in single if row["case"] == item["case"]}
        for item in cases
    }
    rng = np.random.default_rng(environment["seed"] + 1)
    dataset = []
    for item in cases:
        arrays = [module.make_cost(item, rng) for _ in range(environment["instances_per_single_case"])]
        counts = [len(module.solve("scipy", a, item["limit"])) for a in arrays]
        dataset.append({
            "case": item["case"], "n": item["n"], "m": item["m"],
            "limit": str(item["limit"]), "mean_matches": float(np.mean(counts)),
            "mean_unmatched_smaller_side": float(min(item["n"], item["m"]) - np.mean(counts)),
            "minimum_matches": min(counts), "maximum_matches": max(counts),
        })
    module.csv_write(RESULTS / "dataset_summary.csv", dataset)
    derived = {
        item["case"]: {
            f"{other}_over_jvrust": float(by_case[item["case"]][other]["median_ns"])
                                     / float(by_case[item["case"]]["jvrust"]["median_ns"])
            for other in NAMES if other != "jvrust"
        } for item in cases
    }
    (RESULTS / "derived.json").write_text(json.dumps(derived, indent=2) + "\n")
    table = [
        "# Generated result tables", "",
        "Single-call values are median microseconds per matrix.", "",
        "| Case | " + " | ".join(LABELS) + " |",
        "| --- |" + " ---: |" * len(LABELS),
    ]
    for item in cases:
        name = item["case"]
        values = [float(by_case[name][solver]["median_ns"]) / 1000 for solver in NAMES]
        table.append("| " + name + " | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    table += ["", "Batch values are median milliseconds per whole batch.", "",
              "| Case | Method | Median ms |", "| --- | --- | ---: |"]
    for row in batch:
        table.append(f"| {row['case']} | {row['solver']} | {float(row['median_ns'])/1e6:.3f} |")
    table += ["", "Memory values are MiB; each row uses one fresh subprocess.", "",
              "| Case | Method | Peak before | Peak after | Additional peak |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in memory:
        table.append(
            f"| {row['case']} | {row['solver']} | {row['peak_before']/2**20:.3f} | "
            f"{row['peak_after']/2**20:.3f} | {row['additional_peak']/2**20:.3f} |"
        )
    (RESULTS / "generated_tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    # Figure 1: retain the unfavorable block case and the tied case.
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 4.8), sharey=True)
    groups = [
        [item["case"] for item in cases if not math.isfinite(item["limit"])],
        [item["case"] for item in cases if math.isfinite(item["limit"])],
    ]
    for ax, group, title in zip(axes, groups, ("Required assignment", "Finite rejection limit")):
        x = np.arange(len(group))
        for k, (solver, label, color) in enumerate(zip(NAMES, LABELS, COLORS)):
            values = np.array([float(by_case[name][solver]["median_ns"]) / 1e6 for name in group])
            lo = np.array([float(by_case[name][solver]["q25_ns"]) / 1e6 for name in group])
            hi = np.array([float(by_case[name][solver]["q75_ns"]) / 1e6 for name in group])
            positions = x + (k - 3) * 0.115
            ax.bar(positions, values, width=0.11, color=color, label=label)
            ax.errorbar(positions, values, yerr=[values-lo, hi-values],
                        fmt="none", ecolor="#333333", linewidth=0.6, capsize=1)
        ax.set_yscale("log")
        ax.set_xticks(x, group, rotation=48, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Median time per matrix (ms, logarithmic)")
    axes[0].legend(ncol=7, loc="upper left", bbox_to_anchor=(0, 1.24), frameon=False)
    fig.tight_layout()
    save(fig, "single_runtime")

    # Figure 2: compare whole batches, preserving the slower threaded baseline.
    batch_cases = [spec["case"] for spec in module.batch_specs()]
    fig, axes = plt.subplots(1, len(batch_cases), figsize=(15.5, 4.3))
    for ax, case in zip(axes, batch_cases):
        rows = {row["solver"]: row for row in batch if row["case"] == case}
        group = [rows[name] for name in batch_order(rows)]
        values = [float(row["median_ns"]) / 1e6 for row in group]
        names = [batch_label(row["solver"]) for row in group]
        colors = [COLOR[row["solver"].split("_")[0]] for row in group]
        lo = [float(row["median_ns"])-float(row["q25_ns"]) for row in group]
        hi = [float(row["q75_ns"])-float(row["median_ns"]) for row in group]
        ax.bar(range(len(group)), values, color=colors)
        ax.errorbar(range(len(group)), values, yerr=np.array([lo, hi])/1e6,
                    fmt="none", color="#333333", linewidth=0.8, capsize=2)
        ax.set_xticks(range(len(group)), names, rotation=50, ha="right", fontsize=8)
        ax.set_title(case)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Median time per whole batch (ms, logarithmic)")
    fig.tight_layout()
    save(fig, "batch_runtime")

    # Figure 3: high-water-mark increases, not total memory or allocation counts.
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    methods = module.MEMORY_SOLVERS
    for k, case in enumerate(("required", "optional")):
        group = {row["solver"]: row for row in memory if row["case"] == case}
        values = [group[name]["additional_peak"] / 2**20 for name in methods]
        positions = np.arange(len(methods)) + (k - 0.5) * 0.34
        bars = ax.bar(positions, values, width=0.32,
                      color=("#176ca4", "#d17c26")[k], label=case)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value*1.12, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.07, 500)
    ax.set_xticks(range(len(methods)), [LABEL[name] for name in methods])
    ax.set_ylabel("Additional peak working set (MiB, logarithmic)")
    ax.set_title("256 × 4096 float64 costs; 8 MiB input already resident")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "memory_peak")
    print("Generated tables, dataset summaries, and three PNG/SVG figures.")


if __name__ == "__main__":
    main()
