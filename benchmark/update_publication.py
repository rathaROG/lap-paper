"""Regenerate current Windows publication tables and figures from saved data.

No solver is imported or executed. The audited source ZIP remains immutable.
Linux comparisons, application evidence and LSAPE use their separate pipelines.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import zipfile

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
DEFAULT_BUNDLE = HERE / "controlled-windows-20261029.zip"
BASE_STEMS = {"single_runtime", "batch_runtime", "memory_peak", "scaling_required", "scaling_optional", "scaling_blocks"}
SOURCE_FILES = (
    "base/single_summary.csv", "base/batch_summary.csv", "base/memory.json",
    "formulations/summary.csv", "formulations/datasets.json", "scaling/summary.csv",
    "scaling/datasets.json", "scaling/environment.json", "extended/summary.csv",
    "extended/plan.json", "extended/checks.json", "extended/memory_summary.csv",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def indexed(path):
    return {(r["case"], r["solver"]): r for r in rows(path)}


def normalize_svg(path):
    """Keep generated text exports free of trailing spaces and CRLF changes."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in source.splitlines()) + "\n",
                    encoding="utf-8", newline="\n")


@contextmanager
def results_source(source=None):
    source = Path(source or DEFAULT_BUNDLE)
    if source.is_dir():
        yield source
        return
    with TemporaryDirectory(prefix="lap-paper-windows-") as temporary:
        destination = Path(temporary)
        # Use the existing, independently tested archive path validation.
        from verify_results import extract_results
        extract_results(source, destination)
        yield destination


def load_table_data(folder):
    from verify_results import table_data
    return table_data(Path(folder))


def latest_table_values(root):
    from verify_results import environment_tables
    return environment_tables(load_table_data(Path(root) / "py314-results"),
                              load_table_data(Path(root) / "py311-results"))


def prepare(source=None, tables_only=False):
    source = Path(source or DEFAULT_BUNDLE)
    with results_source(source) as root:
        primary, replication = root / "py314-results", root / "py311-results"
        data = latest_table_values(root)
        fragments = {label: {"values": values, "places": places}
                     for label, (values, places) in data.items()}
        (HERE / "publication-windows-20261029-tables.json").write_text(
            json.dumps(fragments, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if tables_only:
            return {"tables": len(fragments), "figures": 0}

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import MaxNLocator, NullLocator
        import publication_extended as extended
        import publication_scaling as scaling
        from verify_results import SINGLE_CASES, SINGLE_METHODS, BATCH_CASES, BATCH_METHODS, MEMORY_METHODS

        figures, images = PAPER / "figures", PAPER / "latex/images"
        figures.mkdir(exist_ok=True)
        (figures / "paper").mkdir(exist_ok=True)
        images.mkdir(exist_ok=True)
        manifest = {
            "measurement_date": "2026-10-29",
            "primary": "Windows Python 3.14.10",
            "replication": "Windows Python 3.11.18",
            "packages": {"lapx": "0.13.0", "laprust": "1.2.0", "lapcpp": "1.2.0"},
            "rendering_environment": {"python": sys.version, "numpy": np.__version__, "matplotlib": matplotlib.__version__},
            "source_bundle": {"path": source.name, "sha256": digest(source)} if source.is_file() else None,
            "scripts": {p.name: digest(p) for p in (Path(__file__), HERE / "publication_extended.py", HERE / "publication_scaling.py")},
            "inputs": {f"{tag}/{name}": digest(root / tag / name)
                       for tag in ("py314-results", "py311-results") for name in SOURCE_FILES},
            "figures": {},
        }
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                             "axes.spines.top": False, "axes.spines.right": False,
                             "pdf.fonttype": 42, "svg.fonttype": "none"})

        def record(stem, note):
            directory = "figures" if stem in BASE_STEMS else "figures/paper"
            normalize_svg(PAPER / directory / f"{stem}.svg")
            manifest["figures"][stem] = {
                "note": note,
                "outputs": {f"{directory}/{stem}.{suffix}": digest(PAPER / directory / f"{stem}.{suffix}")
                            for directory, suffix in ((directory, "pdf"), (directory, "png"),
                                                      (directory, "svg"), ("latex/images", "pdf"))},
            }

        def finish(fig, stem, note):
            destination = figures if stem in BASE_STEMS else figures / "paper"
            for suffix in ("pdf", "png", "svg"):
                fig.savefig(destination / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
            shutil.copyfile(destination / f"{stem}.pdf", images / f"{stem}.pdf")
            plt.close(fig)
            record(stem, note)

        names = ("Square\n128 x 128", "Square\n512 x 512", "Wide\n64 x 512", "Tall\n512 x 64",
                 "Uniform\n256 x 320", "Blocks\n256 x 320", "Ties\n128 x 160", "128 x 160\nL = 0.5",
                 "256 x 320\nL = 0.5", "128 x 160\nL = 0.05", "Signed 128 x 160\nL = -0.25", "Square 128\nL = 0.5")
        colors = ("#176ca4", "#78519a", "#2a9d8f", "#d17c26", "#32905d", "#878787", "#bd4141")
        palette = dict(zip(SINGLE_METHODS, colors))
        single = indexed(primary / "base/single_summary.csv")
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.9))
        for ax, selection, title in zip(axes, (range(7), range(7, 12)), ("Required assignment", "Finite rejection limit")):
            selection = list(selection)
            x = np.arange(len(selection))
            for i, (method, color) in enumerate(zip(SINGLE_METHODS, colors)):
                series = [single[SINGLE_CASES[j], method] for j in selection]
                med, low, high = [np.array([float(r[key]) / 1e6 for r in series])
                                  for key in ("median_ns", "q25_ns", "q75_ns")]
                positions = x + (i - 3) * .115
                ax.bar(positions, med, width=.11, color=color, label=method.upper())
                ax.errorbar(positions, med, yerr=(med-low, high-med), fmt="none", color="#333333", lw=.6, capsize=1)
            ax.set_xticks(x, [names[j] for j in selection], fontsize=7.2)
            ax.set(title=title, ylabel="Median time per matrix (ms)", yscale="log")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.36), frameon=False, fontsize=8)
        fig.tight_layout(h_pad=2.5)
        finish(fig, "single_runtime", "Windows Python 3.14.10; medians and timing-round IQRs; finite-limit JVS/JVC/SciPy use square expansion.")

        batch = indexed(primary / "base/batch_summary.csv")
        titles = ("Required: 64 problems, 64 x 80", "Finite limit: 16 problems, 128 x 160, L = 0.5", "Required: 1024 problems, 8 x 10")
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5))
        for ax, case, title in zip(axes, BATCH_CASES, titles):
            series = [batch[case, method] for method in BATCH_METHODS]
            med, low, high = [np.array([float(r[key]) / 1e6 for r in series])
                              for key in ("median_ns", "q25_ns", "q75_ns")]
            labels = [m.replace("_batch_t", "\nbatch, ").replace("_loop", "\nloop").upper() for m in BATCH_METHODS]
            ax.bar(range(len(series)), med, color=[palette[m.split("_")[0]] for m in BATCH_METHODS])
            ax.errorbar(range(len(series)), med, yerr=(med-low, high-med), fmt="none", color="#333333", lw=.7, capsize=2)
            ax.set_xticks(range(len(series)), labels, fontsize=6.1)
            ax.set(title=title, ylabel="Median whole-batch time (ms)", yscale="log")
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.tight_layout(h_pad=2.2)
        finish(fig, "batch_runtime", "Windows Python 3.14.10; whole-batch medians and timing-round IQRs; 1/4 requested workers.")

        memory = {(r["case"], r["solver"]): r["additional_peak"] / 2**20 for r in read(primary / "base/memory.json")}
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        positive = []
        for i, case in enumerate(("required", "optional")):
            values = [memory[case, method] for method in MEMORY_METHODS]
            positive += [v for v in values if v > 0]
            x = np.arange(len(values)) + (i-.5)*.34
            bars = ax.bar(x, values, width=.32, color=("#176ca4", "#d17c26")[i], label=("Required", "Finite limit L = 0.5")[i])
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value*1.12 if value else .03, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(MEMORY_METHODS)), [m.upper() for m in MEMORY_METHODS])
        ax.set(yscale="log", ylim=(min(positive)*.5, 500), ylabel="Additional peak working set (MiB)", title="256 x 4096 float64; 8 MiB input already resident")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.2)
        fig.tight_layout()
        finish(fig, "memory_peak", "Windows Python 3.14.10; one fresh process per condition; increments are not allocation counts.")

        scaling.PAPER = PAPER
        scaling.generate(primary / "scaling")
        for stem in ("scaling_required", "scaling_optional", "scaling_blocks"):
            shutil.copyfile(images / f"{stem}.pdf", figures / f"{stem}.pdf")
            record(stem, "Windows Python 3.14.10; medians and timing-round IQRs; bounded initializer B=min(c,4).")

        heights = {"large_square": 3.4, "large_rectangles": 5.4, "extreme_rectangles": 5.4,
                   "structure_required": 5.2, "structure_optional": 5.2, "rejection_sweep": 8.2, "memory_sweeps": 5.4}

        def publish_extended(fig, axes, destination, stem, title, note, ignored):
            if stem not in heights:
                plt.close(fig)
                return
            fig.set_size_inches(7.0, heights[stem])
            if stem == "rejection_sweep":
                grid = fig.add_gridspec(4, 2)
                for col in range(4):
                    axes[0, col].set_subplotspec(grid[col, 0])
                    axes[1, col].set_subplotspec(grid[col, 1])
            handles, labels = [], []
            for i, ax in enumerate(np.asarray(axes).ravel()):
                ax.title.set_fontsize(9)
                ax.xaxis.label.set_fontsize(8)
                ax.yaxis.label.set_fontsize(8)
                ax.tick_params(axis="both", labelsize=7.5)
                if stem.startswith("structure_"):
                    ax.set_xlabel(("Competing row fraction", "Forbidden probability", "Quantization levels")[i % 3])
                if stem == "memory_sweeps":
                    ax.set_xlabel("Larger dimension (smaller = 256)")
                    ax.set_ylabel("Additional peak (MiB)")
                if stem == "rejection_sweep" and i >= 4:
                    ax.set_ylabel("Unmatched (%)")
                for handle, label in zip(*ax.get_legend_handles_labels()):
                    if label not in labels:
                        handles.append(handle)
                        labels.append(label)
            fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1), fontsize=8, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, .90 if heights[stem] > 7 else .86), pad=.8, w_pad=1.2, h_pad=1.4)
            finish(fig, stem, "Windows Python 3.14.10. " + note)

        extended.legend_and_save = publish_extended
        extended.new_timing_charts(primary / "extended", [])
        extended.rejection_chart(primary / "extended", [])
        extended.memory_chart(primary / "extended", [])

        files = {"single": "base/single_summary.csv", "formulations": "formulations/summary.csv",
                 "scaling": "scaling/summary.csv", "extended": "extended/summary.csv"}
        tables = {version: {suite: indexed(folder / relative) for suite, relative in files.items()}
                  for version, folder in (("3.14.10", primary), ("3.11.18", replication))}
        cross, exceptions = [], []
        for suite, table in tables["3.14.10"].items():
            for (case, method), value in table.items():
                if method not in ("jvrust", "jvcpp"):
                    continue
                reference = "scipy" if (case, "scipy") in table else "scipy_capped"
                other = tables["3.11.18"][suite]
                a = float(value["median_ns"]) / float(table[case, reference]["median_ns"])
                b = float(other[case, method]["median_ns"]) / float(other[case, reference]["median_ns"])
                cross.append({"suite": suite, "case": case, "solver": method, "python314_method_over_scipy": a,
                              "python311_method_over_scipy": b, "strict_ordering_reversal": (a-1)*(b-1) < 0})
                if (a-1)*(b-1) < 0:
                    exceptions.append((suite, case, method, reference))
        (HERE / "publication-windows-20261029-comparison.json").write_text(json.dumps(cross, indent=2) + "\n", encoding="utf-8")
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        suite_colors = dict(zip(files, ("#78519a", "#d17c26", "#32905d", "#176ca4")))
        points = [r for r in cross if r["solver"] == "jvrust"]
        all_values = []
        for suite, color in suite_colors.items():
            selected = [r for r in points if r["suite"] == suite]
            x = [1/r["python314_method_over_scipy"] for r in selected]
            y = [1/r["python311_method_over_scipy"] for r in selected]
            all_values += x+y
            axes[0].scatter(x, y, label=f"{suite} ({len(selected)})", color=color, s=13, alpha=.7)
        low, high = min(all_values)*.75, max(all_values)*1.4
        changed = sum(r["strict_ordering_reversal"] for r in points)
        axes[0].set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
                    xlabel="SciPy / JVRUST, Python 3.14.10", ylabel="SciPy / JVRUST, Python 3.11.18",
                    title=f"{len(points)-changed} / {len(points)} median orderings agree")
        axes[0].plot([low, high], [low, high], color="#777777", ls=":", lw=.8)
        axes[0].axhline(1, color="#555555", lw=.6)
        axes[0].axvline(1, color="#555555", lw=.6)
        axes[0].legend(fontsize=6.4, frameon=False, loc="upper left")
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for version, color, style in (("3.14.10", "#176ca4", "-"), ("3.11.18", "#d17c26", "--")):
            table = tables[version]["scaling"]
            values = [float(table[f"optional_blocks_{n}x{5*n//4}", "scipy_capped"]["median_ns"]) /
                      float(table[f"optional_blocks_{n}x{5*n//4}", "jvrust"]["median_ns"]) for n in sizes]
            axes[1].plot(sizes, values, color=color, ls=style, marker="o", ms=3, label="Python "+version)
        axes[1].set(xscale="log", yscale="log", xlabel="Smaller N; M = 5N/4", ylabel="Capped SciPy / JVRUST", title="Sampled congestion sweep")
        axes[1].set_xticks(sizes, [str(n) for n in sizes], fontsize=6.5)
        axes[1].xaxis.set_minor_locator(NullLocator())
        axes[1].axhline(1, color="#555555", ls=":", lw=.8)
        axes[1].legend(fontsize=7, frameon=False)
        for ax in axes:
            ax.grid(alpha=.15)
            ax.set_axisbelow(True)
        fig.tight_layout(pad=1, w_pad=2)
        finish(fig, "environment_consistency", f"Latest Windows runs; {changed} of 201 JVRUST median orderings reverse. Ratios use within-environment medians.")

        if len(exceptions) != 4:
            raise ValueError(f"Expected four updated ranking-exception panels, got {len(exceptions)}")
        fig, axes = plt.subplots(2, 2, figsize=(7, 5.8))
        for ax, (suite, case, method, reference) in zip(axes.ravel(), exceptions):
            for i, (version, color) in enumerate((("3.14.10", "#176ca4"), ("3.11.18", "#d17c26"))):
                series = [tables[version][suite][case, m] for m in (method, reference)]
                med, low, high = [np.array([float(r[key])/1e6 for r in series]) for key in ("median_ns", "q25_ns", "q75_ns")]
                x = np.arange(2)+(i-.5)*.34
                bars = ax.bar(x, med, width=.32, label="Python "+version, color=color)
                ax.errorbar(x, med, yerr=(med-low, high-med), fmt="none", color="#333333", capsize=2, lw=.7)
                ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=6.3)
            title = case.removeprefix("optional_").replace("congestion_optional_", "congestion ").replace("_", " ")
            ax.set_xticks(range(2), (method.upper(), "SciPy capped"), fontsize=8)
            ax.set(ylabel="Time per matrix (ms)", title=suite+": "+title)
            ax.title.set_fontsize(8)
            ax.set_ylim(0, ax.get_ylim()[1]*1.20)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(.5, 1), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .94), pad=.8, w_pad=1.4, h_pad=2)
        finish(fig, "ranking_exceptions", "All four latest native/SciPy ordering reversals: JVRUST one, JVCPP three. Medians and timing-round IQRs; axes start at zero.")
        if len(manifest["figures"]) != 15:
            raise ValueError("Updated figure coverage differs")
        (HERE / "publication-windows-20261029-figures.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"tables": len(fragments), "figures": len(manifest["figures"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="Audited bundle ZIP or expanded campaign directory")
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.results, args.tables_only), indent=2))
