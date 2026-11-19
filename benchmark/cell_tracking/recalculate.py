"""Recalculate the ByoTrack comparison from saved observations using only Python.

After a complete measurement run, put observations.json and the corresponding
benchmark.py beside this script, then run it with --write-manifest to bind them.
Use --table PATH for the full LaTeX fragment, --compact-table PATH for
the single-column version, and --profile-table PATH for instrumented timings.
The audit checks saved records without executing the tracker or solvers.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from statistics import median


ROOT = Path(__file__).resolve().parent
BOUND_FILES = ("benchmark.py", "observations.json")
DATASETS = {
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC",
    "Hydra",
}
ARMS = ("scipy_stock", "scipy_capped", "lapx_rust")
STOCK, CAPPED, RUST = ARMS
SHA256 = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def manifest():
    return {"sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                       for name in BOUND_FILES}}


def check_track_record(record, label):
    require(type(record["count"]) is int and record["count"] >= 0, f"{label}: track count")
    require(SHA256.fullmatch(record["sha256"]), f"{label}: track hash")


def derive(report):
    datasets = report["datasets"]
    require(set(datasets) == DATASETS, "Expected all eight CTC excerpts and Hydra")
    require(type(report["seed"]) is int, "Recorded order seed")
    orders = list(itertools.permutations(ARMS))
    random.Random(report["seed"]).shuffle(orders)
    if "orders" in report:
        require([tuple(order) for order in report["orders"]] == orders, "Recorded order schedule")

    rows = []
    for name, dataset in datasets.items():
        frames = dataset["frames"]
        identity = dataset["input"]
        require(type(frames) is int and 0 < frames <= 100, f"{name}: excerpt size")
        require(identity["stop"] - identity["start"] == frames, f"{name}: input frame bounds")
        if name == "Hydra":
            require((identity["start"], identity["stop"]) == (130, 230), "Hydra frame bounds")
        require(SHA256.fullmatch(identity["sha256"]), f"{name}: input hash")
        detections = dataset["mean_detections"]
        require(math.isfinite(detections) and detections > 0, f"{name}: detections per frame")

        rounds = dataset["rounds"]
        require([tuple(round_) for round_ in rounds] == orders, f"{name}: six balanced rounds")
        profiles, validation = dataset["profiles"], dataset["validation"]
        require(set(profiles) == set(validation) == set(ARMS), f"{name}: profile and validation coverage")
        stock = profiles[STOCK]
        times = {}
        comparisons = {}
        for arm in ARMS:
            label = f"{name}/{arm}"
            profile = profiles[arm]
            require(0 < profile["nonempty"] <= profile["calls"], f"{label}: assignment call counts")
            require(profile["calls"] == frames, f"{label}: association profile frame coverage")
            require(len(profile["max_shape"]) == 2 and
                    all(type(size) is int and size > 0 for size in profile["max_shape"]),
                    f"{label}: maximum matrix shape")
            require(math.isfinite(profile["association_ns_per_frame"]) and
                    profile["association_ns_per_frame"] > 0, f"{label}: profiled association timing")
            require(SHA256.fullmatch(profile["cost_sha256"]), f"{label}: cost hash")
            check_track_record(profile["tracks"], label)

            observations = [round_[arm] for round_ in rounds]
            require(all(item["tracks"] == profile["tracks"] for item in observations),
                    f"{label}: recorded track hashes across timed and profiled runs")
            ns_per_frame = [item["ns_per_frame"] for item in observations]
            require(all(math.isfinite(value) and value > 0 for value in ns_per_frame),
                    f"{label}: positive, finite timings")
            times[arm] = median(ns_per_frame) / 1e6

            matches = profile["cost_sha256"] == stock["cost_sha256"]
            check = validation[arm]
            require(check["same_costs_as_stock"] is matches, f"{label}: recorded cost comparison")
            gap = check["max_abs_objective_gap"]
            if matches:
                require(gap is not None and math.isfinite(gap) and gap >= 0,
                        f"{label}: recorded objective gap")
                for field in ("calls", "nonempty", "max_shape"):
                    require(profile[field] == stock[field], f"{label}: shared-cost {field}")
            else:
                require(gap is None, f"{label}: incomparable objective gap")
            if arm != STOCK:
                comparisons[arm] = (matches, profile["tracks"] == stock["tracks"], gap)

        rows.append({
            "name": name, "frames": frames, "detections": detections, "shape": stock["max_shape"],
            "start": identity["start"], "stop": identity["stop"],
            "times": times, "comparisons": comparisons,
            "association_times": {arm: profiles[arm]["association_ns_per_frame"] / 1e6 for arm in ARMS},
            "stock_over_rust": times[STOCK] / times[RUST],
            "capped_over_rust": times[CAPPED] / times[RUST],
        })
    return sorted(rows, key=lambda row: (row["detections"], row["name"]))


def table(rows):
    lines = [
        "% Generated by benchmark/cell_tracking/recalculate.py from the saved observations.",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{Linker time (ms/frame)} & \multicolumn{2}{c}{Time ratio} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"Sequence & Frames & Det./frame & Max. matrix & Stock & Capped & Rust & Stock/Rust & Capped/Rust \\",
        r"\midrule",
    ]
    for row in rows:
        times = row["times"]
        n, m = row["shape"]
        cells = [row["name"], str(row["frames"]), f"{row['detections']:.1f}", rf"${n}\times{m}$",
                 *(f"{times[arm]:.3f}" for arm in ARMS),
                 f"{row['stock_over_rust']:.3f}", f"{row['capped_over_rust']:.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\endgroup"))
    return "\n".join(lines) + "\n"


def association_totals(rows):
    """Sum one recorded instrumented association run per excerpt, in seconds.

    Profile means use all frames, including empty association calls. Convert
    each unrounded mean back to its clip total before summing; primary timing
    rounds and rounded table cells are not inputs to this aggregate.
    """
    return {arm: math.fsum(row["frames"] * row["association_times"][arm] / 1000
                           for row in rows) for arm in ARMS}


def compact_table(rows):
    lines = [
        "% Generated by benchmark/cell_tracking/recalculate.py from the saved observations.",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r"\multicolumn{6}{@{}r@{}}{Linker time (ms/frame)} \\",
        r"\cmidrule(lr){4-6}",
        r"Sequence & Frames & Mean det./frame & Stock & Capped & Rust \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [row["name"], str(row["frames"]), f"{row['detections']:.1f}",
                 *(f"{row['times'][arm]:.3f}" for arm in ARMS)]
        lines.append(" & ".join(cells) + r" \\")
    totals = association_totals(rows)
    frames = sum(row["frames"] for row in rows)
    lines.extend((
        r"\midrule",
        rf"\multicolumn{{3}}{{@{{}}l}}{{Association time, all {frames} frames (s)}} & "
        + " & ".join(f"{totals[arm]:.3f}" for arm in ARMS) + r" \\",
        r"\multicolumn{3}{@{}l}{Relative to Rust ($\times$)} & "
        + " & ".join(f"{totals[arm] / totals[RUST]:.3f}" for arm in ARMS) + r" \\",
    ))
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\endgroup"))
    return "\n".join(lines) + "\n"


def profile_table(rows):
    lines = [
        "% Generated by benchmark/cell_tracking/recalculate.py from the saved observations.",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\begin{tabular}{@{}lrrrrr@{}}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Profiled association time (ms/frame)} \\",
        r"\cmidrule(lr){4-6}",
        r"Sequence & Source frame range & Max. stock input matrix & Stock & Capped & Rust \\",
        r"\midrule",
    ]
    for row in rows:
        n, m = row["shape"]
        cells = [row["name"], f"{row['start']}--{row['stop'] - 1}", rf"${n}\times{m}$",
                 *(f"{row['association_times'][arm]:.3f}" for arm in ARMS)]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\endgroup"))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true", help="Bind the completed observation and driver files")
    parser.add_argument("--table", type=Path, help="Write the full recalculated LaTeX table fragment")
    parser.add_argument("--compact-table", type=Path, help="Write the single-column LaTeX table fragment")
    parser.add_argument("--profile-table", type=Path, help="Write the instrumented-association LaTeX table fragment")
    args = parser.parse_args()

    # Validate every export before writing anything, including a new manifest.
    # Resolve aliases such as '..' and symlinks into the frozen study directory.
    for option in ("table", "compact_table", "profile_table"):
        destination = getattr(args, option)
        if destination is not None:
            destination = destination.resolve()
            if destination.is_relative_to(ROOT.resolve()):
                parser.error("Table exports must be outside benchmark/cell_tracking; frozen evidence cannot be replaced")
            setattr(args, option, destination)

    identity = manifest()
    manifest_path = ROOT / "manifest.json"
    if not args.write_manifest:
        require(identity == read(manifest_path), "Saved observation or benchmark driver checksum mismatch")
    rows = derive(read(ROOT / "observations.json"))
    require(len(rows) == 9 and sum(row["frames"] for row in rows) == 816,
            "Aggregate association workload: nine excerpts and 816 frames")
    if args.write_manifest:
        manifest_path.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.table:
        args.table.write_text(table(rows), encoding="utf-8", newline="\n")
    if args.compact_table:
        args.compact_table.write_text(compact_table(rows), encoding="utf-8", newline="\n")
    if args.profile_table:
        args.profile_table.write_text(profile_table(rows), encoding="utf-8", newline="\n")

    print(f"{len(rows)} excerpts, {sum(row['frames'] for row in rows)} frames, six balanced rounds per arm.")
    totals = association_totals(rows)
    print("Separate profiled association totals (s): "
          + ", ".join(f"{arm}={totals[arm]:.6f}" for arm in ARMS))
    for label, key in (("Stock/Rust", "stock_over_rust"), ("Capped/Rust", "capped_over_rust")):
        values = [row[key] for row in rows]
        print(f"{label}: {min(values):.3f} to {max(values):.3f}; "
              f"{sum(value < 1 for value in values)} ratios below 1.")
    comparisons = [comparison for row in rows for comparison in row["comparisons"].values()]
    gaps = [gap for _, _, gap in comparisons if gap is not None]
    print(f"Recorded comparisons with stock: {sum(matches for matches, _, _ in comparisons)}/{len(comparisons)} "
          f"matching cost streams, {sum(matches for _, matches, _ in comparisons)}/{len(comparisons)} matching track records.")
    if gaps:
        print(f"Maximum recorded absolute objective gap on {len(gaps)} comparable streams: {max(gaps):.6g}.")


if __name__ == "__main__":
    main()
