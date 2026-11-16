"""Project the separate, checksum-bound microscopy session without running it."""
from pathlib import Path
import csv
import hashlib
import json

if __package__:
    from .cell_tracking import recalculate as study
else:
    from cell_tracking import recalculate as study


def load_saved(repo):
    folder = Path(repo) / "benchmark/cell_tracking"
    recorded = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    actual = {"sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in study.BOUND_FILES}}
    study.require(actual == recorded, "Saved microscopy driver or observations checksum mismatch")
    observations = json.loads((folder / "observations.json").read_text(encoding="utf-8"))
    rows = study.derive(observations)
    study.require(len(rows) == 9 and sum(row["frames"] for row in rows) == 816,
                  "Microscopy session must retain nine excerpts and 816 frames")
    return observations, rows


def generate(repo):
    repo = Path(repo)
    observations, rows = load_saved(repo)
    tables = repo / "latex/generated"
    exports = repo / "benchmark/paper-artifacts"
    for directory in (tables, exports):
        directory.mkdir(parents=True, exist_ok=True)
    for name, render in (("T_microscopy.tex", study.compact_table),
                         ("A_microscopy_profiles.tex", study.profile_table)):
        text = render(rows)
        (tables / name).write_text(text, encoding="utf-8", newline="\n")
    records = []
    rounds = []
    paired_ratios = []
    for row in rows:
        data = observations["datasets"][row["name"]]
        records.append({
            "sequence": row["name"], "frames": row["frames"],
            "start_frame": row["start"], "stop_frame_exclusive": row["stop"],
            "detections_per_frame": row["detections"],
            "max_rows": row["shape"][0], "max_columns": row["shape"][1],
            **{f"{arm}_linker_median_ms": row["times"][arm] for arm in study.ARMS},
            **{f"{arm}_profile_ms": row["association_times"][arm] for arm in study.ARMS},
            "stock_over_rust": row["stock_over_rust"],
            "capped_over_rust": row["capped_over_rust"],
        })
        for index, timing in enumerate(data["rounds"], 1):
            paired_ratios.append(timing[study.CAPPED]["ns_per_frame"] /
                                 timing[study.RUST]["ns_per_frame"])
            for position, arm in enumerate(timing, 1):
                rounds.append({"sequence": row["name"], "round": index,
                               "position": position, "arm": arm,
                               "whole_linker_ns_per_frame": timing[arm]["ns_per_frame"]})
    for name, records_ in (("S7_microscopy.csv", records), ("S7_microscopy_rounds.csv", rounds)):
        with (exports / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records_[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records_)
    totals = study.association_totals(rows)
    comparisons = [check for row in rows for check in row["comparisons"].values()]
    summary = {
        "source_file": "benchmark/cell_tracking/observations.json",
        "source_sha256": hashlib.sha256((repo / "benchmark/cell_tracking/observations.json").read_bytes()).hexdigest(),
        "measured_utc": observations["measured_utc"], "versions": observations["versions"],
        "python": observations["python"], "platform": observations["platform"],
        "excerpts": len(rows), "frames": sum(row["frames"] for row in rows),
        "nonempty_calls_per_arm": {arm: sum(d["profiles"][arm]["nonempty"] for d in observations["datasets"].values())
                                  for arm in study.ARMS},
        "whole_linker_rounds_per_arm": 6,
        "whole_linker_ratio_estimator": "Ratio of separate medians of six whole-clip means",
        "association_profile_runs_per_arm_and_excerpt": 1,
        "association_boundary": "Pylapy adapter, solver and pair conversion; excludes cost construction and profiling checks",
        "association_total_seconds": totals,
        "association_time_over_rust": {arm: value / totals[study.RUST] for arm, value in totals.items()},
        "whole_linker_ratio_ranges": {label: [min(row[key] for row in rows), max(row[key] for row in rows)]
                                      for label, key in (("stock_over_rust", "stock_over_rust"), ("capped_over_rust", "capped_over_rust"))},
        "capped_median_ratios_inside_3_percent_band": sum(.97 <= row["capped_over_rust"] <= 1.03 for row in rows),
        "capped_over_rust_round_ratio_range": [min(paired_ratios), max(paired_ratios)],
        "excerpts_with_round_favoring_capped": sum(any(t[study.CAPPED]["ns_per_frame"] < t[study.RUST]["ns_per_frame"]
                                                       for t in d["rounds"]) for d in observations["datasets"].values()),
        "recorded_cross_arm_comparisons": len(comparisons),
        "recorded_matching_cost_streams": sum(check[0] for check in comparisons),
        "recorded_matching_tracks": sum(check[1] for check in comparisons),
        "maximum_recorded_objective_gap": max(check[2] for check in comparisons if check[2] is not None),
        "qualification": "Saved-record arithmetic and digest comparisons only; original costs and tracks are not included.",
        "benchmarks_rerun": False, "solvers_executed": False, "trackers_executed": False,
    }
    (exports / "microscopy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(generate(Path(__file__).resolve().parents[1]), indent=2))
