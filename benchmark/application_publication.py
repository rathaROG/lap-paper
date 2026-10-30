"""Project fresh real-world campaign results into Tables 12 and C1-C4.

Standard library only. No solver, tracker, detector, or measurement is executed.
Projection formulas are retained from the original paper evidence generator;
source identities and measurement scope come from the explicitly selected run.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
COLLECTION = "real-world-windows-20261029-lapx-0.13.0"
SOLVERS = ("lapjv", "lapjvx", "lapjvxa", "lapjvh", "laprust", "lapcpp", "scipy", "lapjvs", "lapjvc")
SELECTED = ("scipy", "lapjvh", "laprust", "lapcpp")
SEQUENCES = ("mot16_04", "mot20_03")
SOURCES = ("gt", "rfdetr_nano")
TRACKERS = ("centroid", "sort")
CASES = tuple(f"{seq}_{source}_{tracker}" for seq in SEQUENCES for source in SOURCES for tracker in TRACKERS)
LABELS = ("12", "C1", "C2", "C3", "C4")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def indexed(rows, keys, label):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise ValueError("Duplicate " + label + ": " + str(key))
        result[key] = row
    return result


def load_campaign(folder, inputs=None):
    folder = Path(folder)
    inputs = Path(inputs or HERE / "real_world/data/inputs")
    summary = read(folder / "summary.json")
    protocol = summary["protocol"]
    if tuple(protocol["solvers"]) != SOLVERS or protocol["rounds"] != 9 or protocol["frames_per_clip"] != 999:
        raise ValueError("Publication scope requires the complete nine-method, nine-round, 999-frame campaign")
    if Counter((r["case"], r["solver"]) for r in summary["tracking"]) != Counter((case, solver) for case in CASES for solver in SOLVERS):
        raise ValueError("Tracking summary coverage differs")
    if Counter((r["case"], r["precision"], r["solver"]) for r in summary["replay"]) != Counter((case, precision, solver) for case in CASES for precision in ("float32", "float64") for solver in SOLVERS):
        raise ValueError("Replay summary coverage differs")
    caches, identities = {}, {}
    for sequence in SEQUENCES:
        for source in SOURCES:
            stem = f"{sequence}_{source}"
            plain, compressed = inputs / (stem + ".json"), inputs / (stem + ".json.gz")
            path = plain if plain.is_file() else compressed
            payload = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
            caches[stem] = json.loads(payload)
            if len(caches[stem]["frames"]) != 999:
                raise ValueError("Publication input frame count differs: " + stem)
            identities[stem] = {"sha256": hashlib.sha256(payload).hexdigest(), "compressed_sha256": digest(path) if path.suffix == ".gz" else None}
    captures = {case: read(folder / "profiles" / (case + "__scipy.json")) for case in CASES}
    if any(not captures[case]["matrix_metadata"] for case in CASES):
        raise ValueError("Missing common-matrix capture metadata")
    return {"folder": folder, "summary": summary, "inputs": caches, "captures": captures, "input_identities": identities}


def case_label(case):
    seq = "MOT16-04" if case.startswith("mot16") else "MOT20-03"
    source = "GT" if "_gt_" in case else "RF768"
    tracker = "Centroid" if case.endswith("_centroid") else "SORT"
    return f"{seq} / {source} / {tracker}"


def _table(label, caption, headers, labels, cells, values, widths, places):
    return {"label": label, "caption": caption, "headers": headers, "row_labels": labels,
            "cells": cells, "numeric_values": values, "column_numeric_widths": widths,
            "places": places}


def _range(lo, hi, places):
    return f"{lo:.{places}f}--{hi:.{places}f}"


def project_evidence(data):
    """Deterministic Table 12/C1--C4 cells plus independent scalar projections.

    Accept a loaded ``load_evidence`` result to avoid rereading the packet.
    Numeric values are unrounded and flattened per row. ``column_numeric_widths``
    states how many scalar values each rendered cell contains.
    """
    summary = data["summary"]
    tracking = indexed(summary["tracking"], ("case", "solver"), "tracking summary")
    replay = indexed(summary["replay"], ("case", "precision", "solver"), "replay summary")
    tables = {}
    cells, numbers = [], []
    for method in SOLVERS:
        ranges = []
        for precision in ("float32", "float64"):
            ratios = [replay[case, precision, "scipy"]["median_ns_per_assignment"] / replay[case, precision, method]["median_ns_per_assignment"] for case in CASES]
            ranges.extend((min(ratios), max(ratios)))
        for tracker in TRACKERS:
            ratios = [tracking[case, "scipy"]["median_tracker_ns_per_frame"] / tracking[case, method]["median_tracker_ns_per_frame"] for case in CASES if case.endswith("_" + tracker)]
            ranges.extend((min(ratios), max(ratios)))
        numbers.append(ranges)
        cells.append([_range(ranges[i], ranges[i + 1], 3) for i in range(0, 8, 2)])
    tables["12"] = _table("12", "Real-sequence speedup ranges relative to SciPy, computed within each pipeline and originating session. Values are minima--maxima across pipelines of SciPy's nine-round median time divided by the method's nine-round median time; replay covers eight pipelines per precision, and tracker updates cover four per tracker. Ranges are descriptive extrema, not interquartile ranges.",
        ["Method", "Replay float32", "Replay float64", "Centroid update", "SORT update"], list(SOLVERS), cells, numbers, [2] * 4, [3] * 8)
    labels = [case_label(case) for case in CASES]
    cells, numbers = [], []
    for case in CASES:
        tracker = case.rsplit("_", 1)[1]
        cache = data["inputs"][case.removesuffix("_" + tracker)]
        gt = sum(len(frame["gt"]) for frame in cache["frames"])
        detected = sum(len(frame["detections"]) for frame in cache["frames"])
        records = data["captures"][case]["matrix_metadata"]
        rr, cc = ([record["shape"][index] for record in records] for index in (0, 1))
        square = 100 * sum(r == c for r, c in zip(rr, cc)) / len(records)
        values = [gt, detected, detected / 999, len(records), min(rr), max(rr), min(cc), max(cc), square]
        numbers.append(values)
        cells.append([str(gt), str(detected), f"{detected / 999:.2f}", str(len(records)), _range(min(rr), max(rr), 0), _range(min(cc), max(cc), 0), f"{square:.1f}"])
    tables["C1"] = _table("C1", "Real-sequence workload over 999 decoded frames per clip. GT and detector box totals count observations, not unique people. LAP dimensions and square percentages describe actual SciPy-captured assignment calls; SORT fast paths are retained. RF768 denotes Nano at 768 pixels and confidence 0.2, with person class 1, max_det 300 and minimum box width 15 pixels.",
        ["Pipeline", "GT boxes", "Det. boxes", "Det./frame", "LAP calls", "Rows min--max", "Cols min--max", "Square %"], labels, cells, numbers, [1, 1, 1, 1, 2, 2, 1], [0, 0, 2, 0, 0, 0, 0, 0, 1])
    cells, numbers = [], []
    for case in CASES:
        rowcells, values = [], []
        for method in SELECTED:
            row = tracking[case, method]
            triplet = [row[key] / 1e6 for key in ("median_tracker_ns_per_frame", "q25_tracker_ns_per_frame", "q75_tracker_ns_per_frame")]
            values.extend(triplet)
            rowcells.append(f"{triplet[0]:.3f} [{triplet[1]:.3f}, {triplet[2]:.3f}]")
        cells.append(rowcells)
        numbers.append(values)
    tables["C2"] = _table("C2", "Tracker update time in milliseconds per frame: median [Q1, Q3] across nine whole-clip means. Fresh Persons, decoding, detector inference, result copying and evaluation are excluded. The frozen artifact includes all nine methods; these absolute times show SciPy and the three native implementations.",
        ["Pipeline", *SELECTED], labels, cells, numbers, [3] * 4, [3] * 12)
    cells, numbers, replay_labels = [], [], []
    for case in CASES:
        for precision in ("float32", "float64"):
            rowcells, values = [], []
            for method in SELECTED:
                row = replay[case, precision, method]
                triplet = [row[key] / 1000 for key in ("median_ns_per_assignment", "q25_ns_per_assignment", "q75_ns_per_assignment")]
                values.extend(triplet)
                rowcells.append(f"{triplet[0]:.2f} [{triplet[1]:.2f}, {triplet[2]:.2f}]")
            replay_labels.append(case_label(case) + " / " + precision)
            cells.append(rowcells)
            numbers.append(values)
    tables["C3"] = _table("C3", "Common-cost replay time in microseconds per assignment: median [Q1, Q3] over nine corpus means. Every method receives the same captured pipeline corpus. The float64 view widens the captured float32 costs; cost construction is unchanged. Adapter conversion, finite checks and pair normalization remain inside replay timing.",
        ["Pipeline / precision", *SELECTED], replay_labels, cells, numbers, [3] * 4, [2] * 12)
    cells, numbers = [], []
    for case in CASES:
        ranges = []
        for key, factor in (("idf1", 100), ("mota", 100), ("num_switches", 1), ("num_false_positives", 1), ("num_misses", 1)):
            values = [tracking[case, method][key] * factor for method in SOLVERS]
            ranges.extend((min(values), max(values)))
        numbers.append(ranges)
        cells.append([_range(ranges[i], ranges[i + 1], 2 if i < 4 else 0) for i in range(0, 10, 2)])
    tables["C4"] = _table("C4", "Supplied-box accuracy ranges across all nine assignment methods. IDF1 and MOTA are percentages; IDSW, FP and FN are counts. Evaluation uses motmetrics 1.4.0 and IoU at least 0.5 outside all timing. Converted annotations lack official MOTChallenge class/ignore metadata, so these are not leaderboard scores. Equal-cost assignment choices can propagate to different identities; no method-independent tracking-accuracy gain is claimed.",
        ["Pipeline", "IDF1 % min--max", "MOTA % min--max", "IDSW min--max", "FP min--max", "FN min--max"], labels, cells, numbers, [2] * 5, [2] * 4 + [0] * 6)
    return {"schema_version": 1, "collection": COLLECTION,
            "source_summary_sha256": digest(data["folder"] / "summary.json"),
            "range_definition": "minimum and maximum across pipelines of SciPy's nine-round median time divided by the method's nine-round median time, within the same pipeline and originating session; each component is a ratio of separate medians, not a median of paired round ratios; ranges are not IQRs or inferential intervals",
            "tables": tables}


def tables_markdown(projections):
    blocks = []
    for label in LABELS:
        table = projections["tables"][label]
        lines = ["| " + " | ".join(table["headers"]) + " |", "| " + " | ".join("---" for _ in table["headers"]) + " |"]
        lines += ["| " + " | ".join([name, *cells]) + " |" for name, cells in zip(table["row_labels"], table["cells"])]
        lines += ["", f"**Table {label}.** {table['caption']}"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"



def project_campaign(folder, inputs=None):
    """Return all numeric table cells from a completed campaign; never write."""
    data = load_campaign(folder, inputs)
    projections = project_evidence(data)
    summary = data["summary"]
    projections.update(
        generator_sha256=digest(__file__),
        environment=summary["environment"],
        measurement_started_utc=summary.get("started_utc"),
        measurement_completed_utc=summary.get("completed_utc"),
        input_identities=data["input_identities"],
        capture_identities={case: digest(Path(folder) / "profiles" / (case + "__scipy.json")) for case in CASES},
        session_scope="All 72 method/pipeline settings are interleaved in each of nine seeded-shuffled rounds within one campaign; measurements never run simultaneously; ratios use separate medians within each pipeline.",
    )
    validation = []
    for precision in ("float32", "float64"):
        for solver in (*SOLVERS, "lapjvr"):
            checks = [summary["validation"][case][precision][solver] for case in CASES]
            validation.append({"precision": precision, "solver": solver,
                "matrices": sum(check["matrices"] for check in checks),
                "objective_nonzero_gap_count": sum(check["objective_nonzero_gap_count"] for check in checks),
                "objective_mismatch_count": sum(check["objective_mismatch_count"] for check in checks),
                "objective_max_abs_gap": max(check["objective_max_abs_gap"] for check in checks),
                "alias_objective_difference_count": sum(check.get("laprust_alias_objective_differences") or 0 for check in checks)})
    projections["numerical_validation"] = validation
    return projections


def prepare(folder, output=None, inputs=None):
    projections = project_campaign(folder, inputs)
    validation = projections["numerical_validation"]
    output = Path(output or HERE / "publication-tracking-20261029-projections.json")
    output.write_text(json.dumps(projections, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(tables_markdown(projections), encoding="utf-8")
    return {"tables": len(projections["tables"]), "captured_matrices": sum(check["matrices"] for check in validation if check["solver"] == "scipy" and check["precision"] == "float32"),
            "backend_matrix_checks": sum(check["matrices"] for check in validation),
            "nonzero_objective_gap_checks": sum(check["objective_nonzero_gap_count"] for check in validation),
            "tolerance_mismatches": sum(check["objective_mismatch_count"] for check in validation), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    print(json.dumps(prepare(options.results, options.output, options.inputs), indent=2))
