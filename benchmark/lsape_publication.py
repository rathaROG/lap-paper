"""Project a completed three-session LSAPE rerun into Tables 10 and 11.

Saved-data arithmetic only: no bridge, solver, timing, or bootstrap is run.
Table 9 is conceptual and is deliberately outside these numeric projections.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
PAIRS = {"O_U": "original_over_one_sided", "U_P": "one_sided_over_padded",
         "O_P": "original_over_padded", "P_T": "padded_over_compact"}
BLOCKS = {"A": "A: squares", "B": "B: uniform 5:4 rectangles",
          "C": "C: congested 5:4 rectangles", "D": "D: uniform 4:1 rectangles",
          "E": "E: extreme rectangles", "F": "F: threshold ties"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=1e-14),
            f"Saved LSAPE arithmetic differs: {label}")


def band_class(value, band):
    return "denominator_faster" if value > band[1] else (
        "denominator_slower" if value < band[0] else "point_within_practical_band")


def project_summary(folder, verification=None):
    """Return raw numeric cells and descriptive findings without writing files."""
    folder = Path(folder)
    summary = read(folder / "summary.json")
    require(summary["status"] == "complete", "LSAPE summary is incomplete")
    manifest, cases = summary["manifest"], summary["cases"]
    require(len(cases) == len(manifest["cases"]) == 67, "LSAPE case coverage differs")
    require(len(summary["sessions"]) == 3 and len({r["pid"] for r in summary["sessions"]}) == 3,
            "Three fresh process identities are required")
    require([c["case"] for c in cases] == [c["case"] for c in manifest["cases"]],
            "LSAPE summary and manifest case order differ")
    require(manifest["rounds"] == 9 and manifest["inputs_per_case"] == 4,
            "LSAPE timing scope differs from the paper")
    band = manifest["practical_ratio_band"]
    require(band == [.97, 1.03], "The declared descriptive band changed")

    exported = {(r["case"], r["comparison"]): r for r in rows(folder / "all_cases.csv")}
    require(len(exported) == len(cases) * len(manifest["comparisons"]), "Ratio export coverage differs")
    findings = {}
    for name in manifest["comparisons"]:
        values = []
        for case in cases:
            record = case["comparisons"][name]
            require(len(record["per_session"]) == 3, "LSAPE comparison needs three session points")
            points = []
            for session in record["per_session"]:
                ratios = session["paired_round_ratios"]
                require(len(ratios) == 9 and all(math.isfinite(v) and v > 0 for v in ratios),
                        "Invalid paired timing-round ratios")
                point = math.exp(statistics.mean(math.log(value) for value in ratios))
                close(point, session["paired_log_central_ratio"], name)
                points.append(point)
            central = statistics.median(points)
            for field, value in (("median_estimate", central), ("min_session", min(points)), ("max_session", max(points))):
                close(record[field], value, case["case"] + "/" + name)
                close(exported[case["case"], name][field], value, "CSV " + case["case"] + "/" + name)
            require(record["median_point_band_class"] == band_class(central, band), "Band classification differs")
            values.append((case["case"], central))
        counts = Counter(band_class(value, band) for _, value in values)
        low, high = min(values, key=lambda item: item[1]), max(values, key=lambda item: item[1])
        findings[name] = {
            **{key: counts[key] for key in ("denominator_faster", "denominator_slower", "point_within_practical_band")},
            "above_one": sum(value > 1 for _, value in values),
            "below_one": sum(value < 1 for _, value in values),
            "minimum": low[1], "minimum_case": low[0], "maximum": high[1], "maximum_case": high[0],
        }

    selected = rows(folder / "paper_table.csv")
    require(len(selected) == 12 and [row["case"] for row in selected] == [case["case"] for case in cases if case["paper_subset"]],
            "Preselected Table 10 cases differ")
    table10 = []
    by_case = {case["case"]: case for case in cases}
    for row in selected:
        case = by_case[row["case"]]
        padded = case["methods"]["specialized_padded"]
        original = case["methods"]["original"]
        values = [original["median_of_session_medians_us"] / 1000,
                  100 * padded["median_session_smaller_side_unmatched_fraction"]]
        close(row["O_ms"], values[0], "Table 10 original time")
        close(row["smaller_side_unmatched_pct"], values[1], "Table 10 rejection")
        for prefix, name in PAIRS.items():
            for field in ("median_estimate", "min_session", "max_session"):
                value = case["comparisons"][name][field]
                close(row[prefix + "_" + field], value, "Table 10 ratio")
                values.append(value)
        table10.append(values)

    block_rows = rows(folder / "block_counts.csv")
    table11 = []
    for block in BLOCKS:
        selected_cases = [case for case in cases if case["block"] == block]
        counts = Counter(case["comparisons"]["original_over_padded"]["median_point_band_class"] for case in selected_cases)
        values = [len(selected_cases)] + [counts[key] for key in ("denominator_faster", "denominator_slower", "point_within_practical_band")]
        matches = [row for row in block_rows if row["block"] == block and row["comparison"] == "original_over_padded"]
        require(len(matches) == 1, "Table 11 block export is missing or duplicated")
        require([int(matches[0][key]) for key in ("cases", "denominator_faster", "denominator_slower", "point_within_practical_band")] == values,
                "Table 11 block export differs from case counts")
        table11.append(values)
    table11.append([sum(row[column] for row in table11) for column in range(4)])

    fact_sheet = {
        "comparisons": findings,
        "cases": len(cases), "sessions": len(summary["sessions"]),
        "generated_inputs": len(cases) * manifest["inputs_per_case"] * len(summary["sessions"]),
        "timing_round_averages": len(cases) * len(manifest["methods"]) * manifest["rounds"] * len(summary["sessions"]),
        "session_records": summary["sessions"], "same_utc_day": summary["same_utc_day"],
        "session_scope": summary["replication_scope"], "statistics": summary["statistics"],
        "preselected_rows": selected,
        "source_review": {
            "arithmetic": "The common bridge converts costs to NPY_DOUBLE; cost buffers, potentials, slacks, and both upstream/specialized instantiations already use double arithmetic.",
            "initialization": "This Hungarian experiment uses row-minimum preprocessing, upstream column-order greedy matching, and FIFO tree completion; it has no JV displacement-retry initialization stage to receive the JV scan/stall policy.",
            "rejection_cache": "The cache recomputes penalty minus the current maximal reached-row potential, uses its fast path only for a finite potential with p/2 < u <= 2p, retains the ordered fallback elsewhere, and refreshes the maximum after dual updates.",
            "half_limit": "The common bridge uses half=L/2 in double and penalty=2*half, preserving the represented endpoint price before restoring the full objective.",
            "conclusion": "Source inspection found no necessary mechanical port of the latest JV fixes; no new measured arms or solver-source edits are introduced.",
            "scope": "Algorithm-specific source review and bounded validation, not a proof of arbitrary-range floating-point correctness.",
            "anchors": {"lsape_bridge.cpp": [145, 159, 162, 204, 218, 234, 238, 272],
                        "src/specialized/hungarian-lsape-uniform.hh": [94, 99, 133, 150, 164, 243, 253]},
        },
    }
    verification = Path(verification) if verification else folder.parent / "verification/report.json"
    if verification.is_file():
        verified = read(verification)
        require(verified["status"] == "passed", "LSAPE strict verification did not pass")
        fact_sheet["verification"] = verified
    build = folder.parent / "experiment/build/provenance.json"
    if build.is_file():
        fact_sheet["build_provenance"] = read(build)
        log = build.with_name("build.log")
        if log.is_file():
            versions = re.findall(r"cl\.exe:\s+Version\s+([0-9.]+)", log.read_text(encoding="utf-8"))
            require(len(set(versions)) == 1, "Compiler version is missing or ambiguous")
            fact_sheet["compiler_version"] = versions[0]
    return {
        "schema_version": 1, "collection": "lsape-windows-20261029",
        "source_summary_sha256": digest(folder / "summary.json"),
        "source_csv_sha256": {name: digest(folder / name) for name in ("all_cases.csv", "all_case_methods.csv", "paper_table.csv", "block_counts.csv")},
        "generator_sha256": digest(__file__), "fact_sheet": fact_sheet,
        "tables": {
            "10": {"numeric_values": table10, "column_numeric_widths": [1, 1, 3, 3, 3, 3], "places": [3, 2] + [2] * 12,
                   "case_ids": [row["case"] for row in selected], "triplet_definition": "Median and observed min/max of the three within-session paired-log central estimates; not IQRs or confidence intervals."},
            "11": {"numeric_values": table11, "column_numeric_widths": [1] * 4, "places": 0,
                   "row_labels": [*BLOCKS.values(), "Total"]},
        },
    }


def prepare(folder, output=None, verification=None):
    result = project_summary(folder, verification)
    output = Path(output or HERE / "publication-lsape-20261029-projections.json")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return {"output": str(output), "tables": len(result["tables"]), **{name: result["fact_sheet"][name] for name in ("cases", "sessions", "generated_inputs", "timing_round_averages")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Completed summary directory")
    parser.add_argument("--verification", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    print(json.dumps(prepare(options.results, options.output, options.verification), indent=2))
