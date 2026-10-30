"""Publication scaling plots; adapted from the preserved lap-paper-full generator.

Reads saved summaries only; no solver or benchmark is executed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
STYLES = {
    "jvrust": ("JVRUST", "#176ca4", "o", "-"),
    "jvhybrid": ("JVHYBRID", "#78519a", "s", "--"),
    "jvcpp": ("JVCPP", "#2a9d8f", "X", "-."),
    "jvx": ("JVX", "#d17c26", "^", "-"),
    "jvs": ("JVS", "#32905d", "v", "--"),
    "jvc": ("JVC", "#878787", "P", "-."),
    "scipy": ("SciPy", "#bd4141", "D", ":"),
    "scipy_square": ("SciPy square", "#bd4141", "D", ":"),
    "scipy_dummies": ("SciPy dummy cols", "#32905d", "v", "-."),
    "scipy_capped": ("SciPy capped", "#35616f", "P", "--"),
}
FAMILIES = {
    "square": r"Both dimensions grow: $N=M$",
    "wide": r"Both dimensions grow: $M=4N$",
    "fixed": r"One dimension grows: $N=64$",
    "blocks": r"Congested blocks: $M=5N/4$, $L=0.5$",
}


def generate(results=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
    import numpy as np

    results = Path(results or PAPER / "results" / "scaling")
    metadata = json.loads((results / "environment.json").read_text())
    assert metadata["status"] == "completed", "Do not plot incomplete scaling runs."
    datasets = json.loads((results / "datasets.json").read_text())
    with (results / "summary.csv").open(newline="") as stream:
        summary = {(row["case"], row["solver"]): row for row in csv.DictReader(stream)}
    images = PAPER / "latex" / "images"
    figures = PAPER / "figures"
    images.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    for regime, families, stem in (
        ("required", ("square", "wide", "fixed"), "scaling_required"),
        ("optional", ("square", "wide", "fixed"), "scaling_optional"),
        ("optional", ("blocks",), "scaling_blocks"),
    ):
        methods = metadata[regime + "_methods"]
        height = 7.25 if len(families) == 3 else 3.0
        fig, axes = plt.subplots(len(families), 2, figsize=(6.5, height), squeeze=False)
        for row_index, family in enumerate(families):
            cases = sorted((d for d in datasets if d["regime"] == regime and d["family"] == family),
                           key=lambda d: d["n"] + d["m"])
            x = np.array([d["n"] + d["m"] for d in cases])
            minima, maxima = [], []
            for method in methods:
                label, color, marker, linestyle = STYLES[method]
                data = [summary[d["case"], method] for d in cases]
                values, low, high = [np.array([float(s[key]) / 1e6 for s in data])
                                     for key in ("median_ns", "q25_ns", "q75_ns")]
                minima.append(low.min())
                maxima.append(high.max())
                for ax in axes[row_index]:
                    ax.fill_between(x, low, high, color=color, alpha=.1, linewidth=0)
                    ax.errorbar(x, values, yerr=(values-low, high-values), color=color,
                                marker=marker, markersize=3.2, markerfacecolor="white",
                                markeredgewidth=.8, linestyle=linestyle, linewidth=1.1,
                                elinewidth=.65, capsize=1.2, label=label)
            for column_index, ax in enumerate(axes[row_index]):
                letter = chr(ord("a") + 2 * row_index + column_index)
                title = FAMILIES[family] if column_index == 0 else "Same measurements: log-log axes"
                ax.set_title(f"({letter}) {title}", loc="left", pad=7)
                ax.set_xlabel(r"Sum of matrix dimensions, $N+M$")
                ax.set_ylabel("Runtime per matrix (ms)")
                ax.grid(axis="both", alpha=.18, linewidth=.6)
                ax.set_axisbelow(True)
                if column_index == 0:
                    ax.set_ylim(0, max(maxima) * 1.1)
                    ax.set_xlim(0, max(x) * 1.04)
                    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                else:
                    ax.set_xscale("log", base=2)
                    ax.set_yscale("log")
                    ax.set_ylim(min(minima) * .75, max(maxima) * 1.4)
                    tick_indices = sorted(set(range(0, len(x), 2)) | {len(x)-1})
                    ax.xaxis.set_major_locator(FixedLocator(x[tick_indices]))
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:g}"))
                    ax.xaxis.set_minor_locator(NullLocator())
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, 1.005),
                   ncol=4, frameon=False, columnspacing=1.6, handlelength=2.6, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, .935 if len(families) == 3 else .82), h_pad=1.8, w_pad=1.3)
        for directory, suffix in ((images, "pdf"), (figures, "png"), (figures, "svg")):
            target = directory / f"{stem}.{suffix}"
            fig.savefig(target, dpi=220, bbox_inches="tight", metadata={"Creator": "Scaling artifact"})
            print(target)
        plt.close(fig)

