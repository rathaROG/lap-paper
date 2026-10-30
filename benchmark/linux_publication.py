"""Regenerate current four-environment comparisons from the two controlled ZIPs.

Table projection uses only the standard library. Figure rendering additionally
needs requirements-publication.txt. No solver or measurement is executed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import sys

from update_publication import digest, indexed, normalize_svg, read, results_source, rows

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DEFAULT_LINUX = HERE / "controlled-linux-20261029.zip"
DEFAULT_WINDOWS = HERE / "controlled-windows-20261029.zip"
ENVIRONMENTS = (("windows", "py314-results", "Windows 3.14.10"),
                ("windows", "py311-results", "Windows 3.11.18"),
                ("linux", "py314-results", "Linux 3.14.10"),
                ("linux", "py311-results", "Linux 3.11.18"))
SUMMARY_FILES = {"single": "base/single_summary.csv", "batch": "base/batch_summary.csv",
                 "formulations": "formulations/summary.csv", "scaling": "scaling/summary.csv",
                 "extended": "extended/summary.csv"}
MEMORY_FILES = ("base/memory.json", "extended/memory_samples.json")


def project_tables(linux_root, windows_root):
    """Pure saved-summary projection; return the seven Appendix B table bodies."""
    from verify_results import controlled_paper_tables, table_data
    controlled = [tuple(table_data(Path(root) / name) for name in ("py314-results", "py311-results"))
                  for root in (windows_root, linux_root)]
    projected = controlled_paper_tables(*controlled)
    return {label: {"values": values, "places": places}
            for label, (values, places) in projected.items() if label.startswith("B")}


def ratio_records(tables):
    result = []
    for suite in ("single", "formulations", "scaling", "extended"):
        table = tables[suite]
        for (case, method), record in table.items():
            if method not in ("jvrust", "jvcpp", "jvhybrid"):
                continue
            reference = "scipy" if (case, "scipy") in table else "scipy_capped"
            ref = table[case, reference]
            result.append({"suite": suite, "case": case, "solver": method, "reference": reference,
                           "native_ns": float(record["median_ns"]),
                           "reference_ns": float(ref["median_ns"]),
                           "native_over_reference": float(record["median_ns"]) / float(ref["median_ns"]),
                           "native_iqr_ns": [float(record[k]) for k in ("q25_ns", "q75_ns")],
                           "reference_iqr_ns": [float(ref[k]) for k in ("q25_ns", "q75_ns")]})
    return result


def prepare(linux=None, windows=None, tables_only=False):
    linux, windows = Path(linux or DEFAULT_LINUX), Path(windows or DEFAULT_WINDOWS)
    with ExitStack() as stack:
        roots = {"linux": stack.enter_context(results_source(linux)),
                 "windows": stack.enter_context(results_source(windows))}
        projection = project_tables(roots["linux"], roots["windows"])
        (HERE / "publication-linux-20261029-projections.json").write_text(
            json.dumps(projection, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        folders = {label: roots[system] / tag for system, tag, label in ENVIRONMENTS}
        tables = {label: {suite: indexed(folder / name) for suite, name in SUMMARY_FILES.items()}
                  for label, folder in folders.items()}
        memories = {label: [record for name in MEMORY_FILES for record in read(folder / name)]
                    for label, folder in folders.items()}
        facts = {"measurement_date": "2026-10-29", "environments": {},
                 "scope": "Four measured environments, each relative to its matching SciPy baseline; no causal operating-system/compiler conclusion.",
                 "source_bundles": {"linux": {"path": linux.name, "sha256": digest(linux)},
                                    "windows": {"path": windows.name, "sha256": digest(windows)}}}
        for label, folder in folders.items():
            comparisons = ratio_records(tables[label])
            memory = memories[label]
            if len(memory) != 415:
                raise ValueError("Memory observation coverage differs")
            facts["environments"][label] = {
                "suite_environment": read(folder / "base/environment.json"),
                "run_all": read(folder / "run_all.json"),
                "native_comparisons": comparisons,
                "wins": {method: sum(r["native_over_reference"] < 1 for r in comparisons if r["solver"] == method)
                         for method in ("jvrust", "jvcpp", "jvhybrid")},
                "maximum_native_over_reference": max(r["native_over_reference"] for r in comparisons),
                "zero_peak_increments": sum(r["additional_peak"] == 0 for r in memory),
                "memory_observations": len(memory),
                "memory_counter": sorted({r.get("counter", r.get("memory_counter")) for r in memory}),
                "memory_baseline_headroom_mib": [max(0, r["peak_before"] - r["rss_before"]) / 2**20 for r in memory]}
        (HERE / "publication-linux-20261029-comparison.json").write_text(
            json.dumps(facts, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        if tables_only:
            return {"tables": len(projection), "figures": 0}

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import NullLocator
        from verify_results import quartiles

        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "lap-paper-linux-20261029"})
        destination, images = PAPER / "figures/paper", PAPER / "latex/images"
        destination.mkdir(parents=True, exist_ok=True)
        images.mkdir(parents=True, exist_ok=True)
        palette = dict(zip(folders, ("#a34e3c", "#d39a70", "#176ca4", "#46a6bd")))
        manifest = {"measurement_date": "2026-10-29", "source_bundles": facts["source_bundles"],
                    "rendering_environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                    "scripts": {name: digest(HERE / name) for name in ("linux_publication.py", "update_publication.py", "verify_results.py")},
                    "inputs": {f"{system}/{tag}/{name}": digest(roots[system] / tag / name)
                               for system, tag, _ in ENVIRONMENTS
                               for name in (*SUMMARY_FILES.values(), *MEMORY_FILES)}, "figures": {}}

        def finish(fig, stem, note):
            for suffix in ("pdf", "png", "svg"):
                metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else None
                fig.savefig(destination / f"{stem}.{suffix}", dpi=200, bbox_inches="tight", metadata=metadata)
            normalize_svg(destination / f"{stem}.svg")
            shutil.copyfile(destination / f"{stem}.pdf", images / f"{stem}.pdf")
            plt.close(fig)
            manifest["figures"][stem] = {
                "note": note,
                "outputs": {f"{directory}/{stem}.{suffix}": digest(PAPER / directory / f"{stem}.{suffix}")
                            for directory, suffix in (("figures/paper", "pdf"), ("figures/paper", "png"),
                                                      ("figures/paper", "svg"), ("latex/images", "pdf"))}}

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.1))
        all_values = []
        for version, color, marker in (("3.14.10", "#176ca4", "o"), ("3.11.18", "#c17c26", "^")):
            win = [r for r in facts["environments"]["Windows " + version]["native_comparisons"] if r["solver"] == "jvrust"]
            lin = [r for r in facts["environments"]["Linux " + version]["native_comparisons"] if r["solver"] == "jvrust"]
            if [(r["suite"], r["case"]) for r in win] != [(r["suite"], r["case"]) for r in lin] or len(win) != 201:
                raise ValueError("Cross-platform comparison coverage differs")
            x, y = [[1 / r["native_over_reference"] for r in records] for records in (win, lin)]
            all_values.extend(x + y)
            agree = sum((a - 1) * (b - 1) >= 0 for a, b in zip(x, y))
            axes[0].scatter(x, y, s=13, alpha=.6, marker=marker, color=color, label=f"{version}: {agree}/201 agree")
        low, high = min(all_values) * .8, max(all_values) * 1.25
        axes[0].plot([low, high], [low, high], color="#777777", lw=.8, ls=":")
        axes[0].axhline(1, color="#555555", lw=.7)
        axes[0].axvline(1, color="#555555", lw=.7)
        axes[0].set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
                    xlabel="Matching SciPy / Rust, Windows", ylabel="Matching SciPy / Rust, Linux",
                    title="(a) Ratios of matched median times")
        axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for label in folders:
            table = tables[label]["scaling"]
            for method, style in (("jvrust", "-"), ("scipy_capped", "--")):
                records = [table[f"optional_blocks_{n}x{5*n//4}", method] for n in sizes]
                med, lo, hi = [np.array([float(r[k]) / 1e6 for r in records]) for k in ("median_ns", "q25_ns", "q75_ns")]
                axes[1].plot(sizes, med, color=palette[label], ls=style, lw=1, marker="o", ms=2,
                             label=label if method == "jvrust" else None)
                axes[1].fill_between(sizes, lo, hi, color=palette[label], alpha=.12)
        axes[1].set(xscale="log", yscale="log", xlabel="Smaller N; M = 5N/4", ylabel="Median time per matrix (ms)",
                    title="(b) Congestion, L = 0.5")
        axes[1].set_xticks(sizes, [str(n) for n in sizes], fontsize=6.5)
        axes[1].xaxis.set_minor_locator(NullLocator())
        axes[1].legend(fontsize=6.2, frameon=False, loc="upper left")
        axes[1].text(.03, .65, "Solid: Rust; dashed: capped SciPy\nShading: timing-round IQR", transform=axes[1].transAxes, fontsize=6.3)
        for ax in axes:
            ax.title.set_fontsize(8.5)
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
        fig.tight_layout(pad=.8, w_pad=1.6)
        finish(fig, "linux_platform_overview", "All 201 Rust/reference median-time ratios per version; congestion curves show medians and timing-round IQRs. Above one in panel a favors Rust; each environment uses its matching SciPy baseline.")

        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.5), gridspec_kw={"height_ratios": [1.25, 1]})
        cases = ["batch64x64x80", "batch16x128x160_optional", "batch1024x8x10"]
        case_labels = ["64 x (64 x 80)", "16 x (128 x 160), L = 0.5", "1024 x (8 x 10)"]
        families = [("jvrust", "Rust"), ("jvhybrid", "Hybrid"), ("jvcpp", "C++")]
        positions = np.array([4*i+j for i in range(3) for j in range(3)])
        for i, label in enumerate(folders):
            table = tables[label]["batch"]
            records = [table[case, family + "_batch_t4"] for case in cases for family, _ in families]
            med, lo, hi = [np.array([float(r[k]) / 1e6 for r in records]) for k in ("median_ns", "q25_ns", "q75_ns")]
            x = positions + (i - 1.5) * .18
            axes[0].bar(x, med, .17, color=palette[label], label=label)
            axes[0].errorbar(x, med, yerr=(med-lo, hi-med), fmt="none", color="#333333", capsize=1, lw=.6)
            samples = {(r["case"], r["solver"], int(r["round"])): float(r["nanoseconds"])
                       for r in rows(folders[label] / "base/batch_samples.csv")}
            ratios = [quartiles([samples[case, "jvcpp_batch_t4", round_] / samples[case, "jvrust_batch_t4", round_]
                                 for round_ in range(9)]) for case in cases]
            med, lo, hi = [np.array([r[k] for r in ratios]) for k in range(3)]
            x = np.arange(3) + (i - 1.5) * .18
            bars = axes[1].bar(x, med, .17, color=palette[label])
            axes[1].errorbar(x, med, yerr=(med-lo, hi-med), fmt="none", color="#333333", capsize=1, lw=.6)
            for position, value, upper in zip(x, med, hi):
                axes[1].text(position, upper + .025, f"{value:.3f}", ha="center", va="bottom", fontsize=6)
        axes[0].set_xticks(positions, [family for _ in cases for _, family in families], fontsize=7)
        for center, label in zip((1, 5, 9), case_labels):
            axes[0].text(center, -.11, label, transform=axes[0].get_xaxis_transform(), ha="center", va="top", fontsize=7, clip_on=False)
        axes[0].set(ylabel="Median whole-batch time (ms)", title="(a) Four requested workers: medians and IQRs")
        axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.12)
        for boundary in (3, 7):
            axes[0].axvline(boundary, color="#dddddd", lw=.7)
        axes[1].set_xticks(range(3), case_labels, fontsize=7)
        axes[1].set(ylabel="C++ / Rust round-time ratio", title="(b) Paired round ratios: medians and IQRs")
        axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.12)
        axes[1].axhline(1, color="#555555", ls=":", lw=.8)
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1), frameon=False, fontsize=7)
        for ax in axes:
            ax.title.set_fontsize(8.5)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
        fig.tight_layout(rect=(0, 0, 1, .92), pad=.8, h_pad=3)
        finish(fig, "linux_batch_comparison", "Four-worker whole-batch medians and nine-round IQRs; bottom panel uses the median and IQR of nine paired within-round C++/Rust ratios. Below one favors C++. Logical CPU indices do not establish matching physical core groups across platforms.")
        for system, tag, _ in ENVIRONMENTS:
            manifest["inputs"][f"{system}/{tag}/base/batch_samples.csv"] = digest(roots[system] / tag / "base/batch_samples.csv")

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0))
        zero_counts = []
        for label in folders:
            memory = memories[label]
            zero_counts.append(sum(r["additional_peak"] == 0 for r in memory))
            headroom = [max(0, r["peak_before"] - r["rss_before"]) / 2**20 for r in memory]
            growth = [r["additional_peak"] / 2**20 for r in memory]
            axes[1].scatter(headroom, growth, s=13, alpha=.4, color=palette[label], marker="o" if "3.14" in label else "^", label=label)
        bars = axes[0].bar(range(4), np.array(zero_counts)/415, color=[palette[label] for label in folders], width=.65)
        axes[0].bar_label(bars, labels=[f"{n}/415" for n in zero_counts], padding=3, fontsize=7)
        axes[0].set_xticks(range(4), [label.replace(" ", "\n", 1) for label in folders], fontsize=7)
        axes[0].set(ylim=(0, 1), ylabel="Fraction of memory observations", title="(a) Zero recorded peak increments")
        axes[1].set_xscale("symlog", linthresh=.25)
        axes[1].set_yscale("symlog", linthresh=.25)
        axes[1].set(xlabel="Baseline peak above current RSS (MiB)", ylabel="Additional OS-specific peak (MiB)",
                    title="(b) All 415 observations per environment")
        axes[1].legend(fontsize=6.1, frameon=False, loc="upper left")
        fig.text(.71, .015, "Windows: peak working set; Linux: ru_maxrss high-water RSS", ha="center", fontsize=6.1)
        for ax in axes:
            ax.title.set_fontsize(8.5)
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
        fig.tight_layout(rect=(0, .05, 1, 1), pad=.8, w_pad=1.5)
        finish(fig, "linux_memory_context", "All 415 observations per environment retained, including zeros. Windows additional PeakWorkingSetSize and Linux additional getrusage(RUSAGE_SELF).ru_maxrss are separate OS counters after resident inputs; zero high-water increments do not imply zero allocation.")
        (HERE / "publication-linux-20261029-figures.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        return {"tables": len(projection), "figures": len(manifest["figures"]),
                "rendering_environment": manifest["rendering_environment"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linux", type=Path, default=DEFAULT_LINUX)
    parser.add_argument("--windows", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.linux, args.windows, args.tables_only), indent=2))
