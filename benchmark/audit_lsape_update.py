"""Independent standard-library audit of the fresh controlled LSAPE campaign.

No solver is imported or executed. Recorded certificates and strict-verification
counters are checked as saved evidence; this does not re-prove solver correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


METHODS = ("original", "one_sided_materialized", "implicit_scan", "specialized_cached", "specialized_padded")
CATEGORIES = ("denominator_faster", "denominator_slower", "point_within_practical_band")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, context):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), rel_tol=2e-12, abs_tol=1e-12),
            "LSAPE arithmetic differs: " + context)


def quartiles(values):
    ordered = sorted(values)
    def percentile(p):
        index = (len(ordered) - 1) * p
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return statistics.median(ordered), percentile(.25), percentile(.75)


def classification(value, band):
    return CATEGORIES[0] if value > band[1] else CATEGORIES[1] if value < band[0] else CATEGORIES[2]


def safe(root, name):
    name = str(name).replace("\\", "/")
    require(all(part and part not in (".", "..") and ":" not in part for part in name.split("/")),
            "Unsafe LSAPE source path")
    result = root.joinpath(*name.split("/"))
    require(result.resolve().is_relative_to(root.resolve()), "LSAPE path escapes root")
    return result


def audit(root):
    root = Path(root)
    experiment = root / "experiment"
    summary_folder = root / "summary"
    reports = [read(root / f"session-{index}/report.json") for index in (1, 2, 3)]
    manifest = reports[0]["manifest"]
    specifications = manifest["cases"]
    require(len(specifications) == 67 and sum(s["paper_subset"] for s in specifications) == 12,
            "LSAPE frozen grid differs")
    require(manifest["rounds"] == 9 and manifest["inputs_per_case"] == 4 and
            manifest["practical_ratio_band"] == [.97, 1.03] and set(manifest["methods"]) == set(METHODS),
            "LSAPE timing contract differs")
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                             allow_nan=False).encode()).hexdigest()
    require(len({r["pid"] for r in reports}) == 3, "LSAPE sessions do not use three fresh processes")
    build = read(experiment / "build/provenance.json")
    binary = experiment / "build/_lsape_native.pyd"
    binary_hash = sha(binary)
    require(build["returncode"] == 0 and build["output_sha256"] == binary_hash, "LSAPE build identity differs")
    require("/fp:precise" in build["compiler_flags"] and "/fp:fast" not in build["compiler_flags"],
            "LSAPE floating-point build contract differs")
    source_paths = {"lsape_bridge.cpp": "lsape_bridge.cpp", "build_bridge.py": "build_bridge.py",
                    "lsape-v4.zip": "src/lsape-v4.zip",
                    "hungarian-lsape.hh": "src/upstream-v4/lsape/include/hungarian-lsape.hh",
                    "src/specialized/hungarian-lsape-uniform.hh": "src/specialized/hungarian-lsape-uniform.hh"}
    for name, relative in source_paths.items():
        require(build["input_sha256"][name] == sha(experiment / relative), "LSAPE build source differs: " + name)
    methods = {}
    certificates = timing_count = input_count = 0
    for index, report in enumerate(reports, 1):
        require(report["session_index"] == index and report["status"] == "complete" and not report["validate_only"],
                "LSAPE session is incomplete")
        require(report["manifest"] == manifest and report["manifest_sha256"] == manifest_hash,
                "LSAPE session manifest differs")
        require(report["driver_sha256"] == sha(experiment / "rectangular_campaign.py") and
                report["shared_harness_sha256"] == sha(experiment / "controlled_comparison.py"),
                "LSAPE executed source identity differs")
        require(report["provenance"]["loaded_bridge_sha256"] == binary_hash and
                report["provenance"]["build"] == build, "LSAPE session build differs")
        require(report["python"].startswith("3.11.18 ") and report["versions"] == {"numpy": "2.4.6", "scipy": "1.17.1"},
                "LSAPE measured environment differs")
        folder = root / f"session-{index}"
        for item in read(folder / "provenance_manifest.json"):
            path = safe(folder, item["snapshot"])
            require(path.stat().st_size == item["bytes"] and sha(path) == item["sha256"],
                    "LSAPE session provenance snapshot differs")
        require([c["case"] for c in report["cases"]] == [s["case"] for s in specifications],
                "LSAPE case coverage/order differs")
        for case in report["cases"]:
            key = case["case"]
            require(case["status"] == "complete" and case["input_unchanged_after_timing"] is True,
                    "LSAPE input mutation or incomplete case")
            require(len(case["input_identities"]) == len(case["diagnostics"]) == 4,
                    "LSAPE input/certificate coverage differs")
            input_count += 4
            for ordinal, identity in enumerate(case["input_identities"]):
                seed_key = f"{manifest['grid_id']}|{manifest['base_seed']}|{index}|{case['input_group']}|{ordinal}"
                require(identity["input_index"] == ordinal and identity["seed"] ==
                        int(hashlib.sha256(seed_key.encode()).hexdigest()[:32], 16), "LSAPE input seed identity differs")
                require(all(len(identity[field]) == 64 for field in ("base_sha256", "input_sha256")),
                        "LSAPE input hash missing")
            require(len(case["round_order"]) == 9 and all(set(order) == set(METHODS) and len(order) == 5
                                                        for order in case["round_order"]),
                    "LSAPE shuffled round coverage differs")
            for method in METHODS:
                stats = case["methods"][method]
                values = stats["round_average_us"]
                require(len(values) == len(stats["round_elapsed_ns"]) == 9 and
                        all(math.isfinite(v) and v > 0 for v in values), "LSAPE invalid timing rounds")
                require(stats["cycles_per_round"] > 0, "LSAPE nonpositive calibrated cycles")
                for average, elapsed in zip(values, stats["round_elapsed_ns"]):
                    close(average, elapsed / (1000 * stats["cycles_per_round"] * 4), "calibrated round mean")
                for field, value in zip(("median_us", "q1_us", "q3_us"), quartiles(values)):
                    close(stats[field], value, key + "/" + method + "/" + field)
                methods[key, index, method] = stats
                timing_count += 9
            for item in case["diagnostics"]:
                require(set(item["methods"]) == set(METHODS), "LSAPE diagnostic method coverage differs")
                baseline = item["methods"]["one_sided_materialized"]["state_sha256"]
                for method, record in item["methods"].items():
                    certificate = record["certificate"]
                    require(certificate["passed"] is True and record["input_unchanged"] is True,
                            "LSAPE recorded certificate/input check failed")
                    require(certificate["maximum_residual"] <= certificate["tolerance"],
                            "LSAPE recorded certificate exceeds its tolerance")
                    require(math.isclose(record["objective"], item["expected"], rel_tol=1e-10, abs_tol=1e-10) and
                            math.isclose(record["dual_F"], item["expected"], rel_tol=1e-10, abs_tol=1e-10),
                            "LSAPE recorded objective/dual differs from reference")
                    if method != "original":
                        require(record["bitwise_state_equals_one_sided_upstream"] is True and
                                record["state_sha256"] == baseline, "LSAPE one-sided state identity differs")
                    certificates += 1
    require(timing_count == 9045 and input_count == 804 and certificates == 4020, "LSAPE raw coverage differs")
    for position, spec in enumerate(specifications):
        require(len({tuple(x["input_sha256"] for x in r["cases"][position]["input_identities"])
                     for r in reports}) == 3, "LSAPE input sets repeat across sessions")
    for report in reports:
        groups = {}
        for case in report["cases"]:
            groups.setdefault(case["input_group"], []).append(case)
        for group in groups.values():
            if len(group) == 2:
                require(sorted(c["transpose"] for c in group) == [False, True] and
                        [x["base_sha256"] for x in group[0]["input_identities"]] ==
                        [x["base_sha256"] for x in group[1]["input_identities"]], "LSAPE transpose pairing differs")
            else:
                require(len(group) == 1, "LSAPE unexpected input-group membership")
    method_rows = rows(summary_folder / "all_case_methods.csv")
    require(len(method_rows) == 1005 and
            {(r["case"], int(r["session_index"]), r["method"]) for r in method_rows} == set(methods),
            "LSAPE method summary coverage differs")
    for row in method_rows:
        stats = methods[row["case"], int(row["session_index"]), row["method"]]
        for field in ("median_us", "q1_us", "q3_us"):
            close(row[field], stats[field], "method CSV " + field)
        case = next(c for c in reports[int(row["session_index"]) - 1]["cases"] if c["case"] == row["case"])
        for field, diagnostic in (("mean_row_rejection_fraction", "row_rejection_fraction"),
                                  ("mean_smaller_side_unmatched_fraction", "smaller_side_unmatched_fraction")):
            close(row[field], statistics.mean(item["methods"][row["method"]][diagnostic]
                                             for item in case["diagnostics"]), "method rejection fraction")
    comparisons = {}
    all_rows = rows(summary_folder / "all_cases.csv")
    require(len(all_rows) == 67 * len(manifest["comparisons"]), "LSAPE ratio summary count differs")
    specs = {s["case"]: s for s in specifications}
    for row in all_rows:
        definition = manifest["comparisons"][row["comparison"]]
        require((row["numerator"], row["denominator"]) == (definition["numerator"], definition["denominator"]),
                "LSAPE comparison definition differs")
        points = []
        for index in (1, 2, 3):
            a, b = (methods[row["case"], index, row[name]]["round_average_us"] for name in ("numerator", "denominator"))
            point = math.exp(statistics.mean(math.log(x / y) for x, y in zip(a, b)))
            close(row[f"s{index}_ratio"], point, "paired-log session ratio")
            points.append(point)
        for field, value in (("median_estimate", statistics.median(points)), ("min_session", min(points)), ("max_session", max(points))):
            close(row[field], value, "three-session " + field)
        label = classification(statistics.median(points), manifest["practical_ratio_band"])
        require(row["median_point_band_class"] == label, "LSAPE descriptive ratio-band class differs")
        key = row["case"], row["comparison"]
        require(key not in comparisons, "Duplicate LSAPE comparison row")
        comparisons[key] = row
    counts = rows(summary_folder / "block_counts.csv")
    require(len(counts) == 6 * len(manifest["comparisons"]) and
            len({(r["block"], r["comparison"]) for r in counts}) == len(counts), "LSAPE block count coverage differs")
    for row in counts:
        selected = [s for s in specifications if s["block"] == row["block"]]
        tally = Counter(comparisons[s["case"], row["comparison"]]["median_point_band_class"] for s in selected)
        require(int(row["cases"]) == len(selected) and all(int(row[name]) == tally[name] for name in CATEGORIES),
                "LSAPE descriptive block counts differ")
    paper = rows(summary_folder / "paper_table.csv")
    require([r["case"] for r in paper] == [s["case"] for s in specifications if s["paper_subset"]],
            "LSAPE preselected paper rows differ")
    for row in paper:
        for letter, method in (("O", "original"), ("U", "one_sided_materialized"),
                               ("P", "specialized_padded"), ("T", "specialized_cached")):
            value = statistics.median(methods[row["case"], index, method]["median_us"] for index in (1, 2, 3)) / 1000
            close(row[letter + "_ms"], value, "paper latency")
        for prefix, comparison in (("O_U", "original_over_one_sided"), ("U_P", "one_sided_over_padded"),
                                   ("O_P", "original_over_padded"), ("P_T", "padded_over_compact")):
            for field in ("median_estimate", "min_session", "max_session"):
                close(row[prefix + "_" + field], comparisons[row["case"], comparison][field], "paper ratio")
        for field, diagnostic in (("row_rejected_pct", "row_rejection_fraction"),
                                  ("smaller_side_unmatched_pct", "smaller_side_unmatched_fraction")):
            value = 100 * statistics.median(statistics.mean(item["methods"]["specialized_padded"][diagnostic]
                       for item in report["cases"][next(i for i, s in enumerate(specifications) if s["case"] == row["case"])]["diagnostics"])
                       for report in reports)
            close(row[field], value, "paper rejection fraction")
    verification = read(root / "verification/report.json")
    require(verification["status"] == "passed" and verification["binary_sha256"] == binary_hash and
            verification["provenance_verified_before_loading"] is True, "LSAPE strict verification/build identity differs")
    require(verification["cases"] == 10190 and verification["stress_cases"] == 10000 and
            verification["certificates"] == 50950 and verification["state_identity_comparisons"] == 122280 and
            verification["timed_api_identity_checks"] == 50950 and verification["domain_rejections"] == 50,
            "LSAPE strict verification coverage differs")
    provenance_check = read(root / "verification/provenance-check.json")
    require(provenance_check["passed"] is True and
            all(c["expected"] == c["actual"] for c in provenance_check["checks"].values()),
            "LSAPE strict-verification source identity differs")
    for name, expected in verification["source_sha256"].items():
        require(sha(safe(root / "verification/sources", name)) == expected,
                "LSAPE strict-verification source snapshot differs")
    with (root / "verification/cases.jsonl").open(encoding="utf-8") as stream:
        strict = [json.loads(line) for line in stream if line.strip()]
    require(len(strict) == verification["cases"] and sum(c["category"] == "stress" for c in strict) == 10000,
            "LSAPE strict per-case records differ")
    for case in strict:
        values = list(case["objectives"].values())
        require(len(values) == 5 and all(math.isfinite(v) and math.isclose(v, values[0], rel_tol=1e-10, abs_tol=0)
                                        for v in values), "LSAPE strict recorded objectives disagree")
    summary = read(summary_folder / "summary.json")
    require(summary["status"] == "complete" and summary["manifest"] == manifest,
            "LSAPE combined summary is incomplete")
    for index, session in enumerate(summary["sessions"], 1):
        require(session["session_index"] == index and session["report_sha256"] ==
                sha(root / f"session-{index}/report.json"), "LSAPE summary session source differs")
    return {"status": "passed", "sessions": 3, "cases": 67, "timing_round_averages": timing_count,
            "timed_input_identities": input_count, "recorded_timing_certificates": certificates,
            "paired_log_summaries_recomputed": len(all_rows), "block_counts_recomputed": len(counts),
            "strict_cases": verification["cases"], "strict_certificates": verification["certificates"],
            "binary_sha256": binary_hash, "manifest_sha256": manifest_hash,
            "scope": "Saved-data/provenance arithmetic audit; recorded certificates are not an independent solver re-execution. Exploratory bootstrap intervals are retained, not resampled by this auditor."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = audit(options.results)
    if options.output:
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
