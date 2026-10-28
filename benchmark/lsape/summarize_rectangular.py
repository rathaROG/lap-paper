"""Summarize three complete, fresh rectangular LSAPE sessions without solving.

Keep paired timing rounds within their own session. Across-session ranges are
descriptive ranges over three processes, not confidence intervals or evidence
of independent days, reboots, machines, or application populations.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

from redaction import redact

ROOT = Path(__file__).resolve().parent
BOOTSTRAP_REPLICATES = 10000
np = None


def write_json(path, value):
    path.write_text(json.dumps(redact(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def seed_for(text):
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:32], 16)


def paired_statistics(numerator, denominator, seed, band):
    a, b = np.asarray(numerator, dtype=np.float64), np.asarray(denominator, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Invalid paired timing arrays")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Paired timings must be positive")
    logs = np.log(a / b)
    central = float(np.exp(logs.mean()))
    indices = np.random.default_rng(seed).integers(0, len(logs), size=(BOOTSTRAP_REPLICATES, len(logs)))
    resampled = np.exp(logs[indices].mean(axis=1))
    lower, upper = (float(x) for x in np.quantile(resampled, [0.025, 0.975]))
    point_class = "denominator_faster" if central > band[1] else (
        "denominator_slower" if central < band[0] else "point_within_practical_band")
    return {
        "paired_log_central_ratio": central,
        "paired_round_ratios": (a / b).tolist(),
        "paired_log_ratios": logs.tolist(),
        "ratio_of_session_medians": float(np.median(a) / np.median(b)),
        "rounds": len(logs),
        "exploratory_interval": {
            "method": "Percentile bootstrap of paired round log-ratios within this session",
            "confidence_level_nominal": 0.95, "replicates": BOOTSTRAP_REPLICATES,
            "seed": seed, "lower": lower, "upper": upper,
            "assumptions": "Pairs are resampled as if IID; thermal drift and serial correlation can invalidate nominal coverage. Conditional on the four fixed inputs in this session.",
        },
        "point_band_class": point_class,
        "interval_within_practical_band": lower >= band[0] and upper <= band[1],
        "interval_excludes_one": upper < 1.0 or lower > 1.0,
    }


def check_reports(paths, reports):
    if len(reports) != 3 or {r.get("session_index") for r in reports} != {1, 2, 3}:
        raise ValueError("Supply exactly one report for each session index 1, 2, and 3")
    manifest = reports[0]["manifest"]
    expected_cases = [c["case"] for c in manifest["cases"]]
    expected_methods = set(manifest["methods"])
    if len(expected_cases) != 67:
        raise ValueError("This publication campaign must contain all 67 predeclared cases")
    signatures = {
        "manifest_sha256": {r["manifest_sha256"] for r in reports},
        "driver_sha256": {r["driver_sha256"] for r in reports},
        "shared_harness_sha256": {r["shared_harness_sha256"] for r in reports},
        "bridge_sha256": {r["provenance"]["loaded_bridge_sha256"] for r in reports},
    }
    if any(len(values) != 1 for values in signatures.values()):
        raise ValueError(f"Campaign manifest, executed code, or bridge differs across sessions: {signatures}")
    if len({r["pid"] for r in reports}) != 3:
        raise ValueError("Reports do not identify three different process IDs")
    for path, report in zip(paths, reports):
        if report.get("status") != "complete" or report.get("validate_only"):
            raise ValueError(f"Session is not a completed timing run: {path}")
        if report["manifest"] != manifest:
            raise ValueError("Session manifests differ")
        if [c["case"] for c in report["cases"]] != expected_cases:
            raise ValueError(f"Missing, reordered, or extra cases in {path}")
        for case in report["cases"]:
            if case["status"] != "complete" or not case.get("input_unchanged_after_timing"):
                raise ValueError(f"Incomplete or mutated input in {case['case']}")
            if set(case["methods"]) != expected_methods or len(case["diagnostics"]) != manifest["inputs_per_case"]:
                raise ValueError(f"Method or validation coverage differs in {case['case']}")
            if len(case["round_order"]) != manifest["rounds"]:
                raise ValueError("Round scheduling coverage is incomplete")
            if any(set(order) != expected_methods or len(order) != len(expected_methods) for order in case["round_order"]):
                raise ValueError("A round does not contain each method exactly once")
            for method in expected_methods:
                if len(case["methods"][method]["round_average_us"]) != manifest["rounds"]:
                    raise ValueError("Timing round count differs from the frozen manifest")
            for item in case["diagnostics"]:
                if set(item["methods"]) != expected_methods:
                    raise ValueError("Diagnostic method coverage is incomplete")
                for method, diagnostic in item["methods"].items():
                    if not diagnostic["certificate"]["passed"] or not diagnostic["input_unchanged"]:
                        raise ValueError("A primal-dual certificate or input integrity check failed")
                    if method != "original" and not diagnostic.get("bitwise_state_equals_one_sided_upstream"):
                        raise ValueError("One-sided state-equivalence evidence is missing")
    return manifest, {key: next(iter(values)) for key, values in signatures.items()}


def verify_input_pairing(manifest, reports):
    pair_checks = []
    for report in reports:
        groups = {}
        for case in report["cases"]:
            groups.setdefault(case["input_group"], []).append(case)
        for name, group in groups.items():
            if len(group) == 1:
                continue
            if len(group) != 2 or sorted(c["transpose"] for c in group) != [False, True]:
                raise ValueError("A predeclared transpose pair has unexpected membership")
            a, b = group
            if [i["base_sha256"] for i in a["input_identities"]] != [i["base_sha256"] for i in b["input_identities"]]:
                raise ValueError(f"Wide/tall cases do not share base costs: {name}")
            pair_checks.append({"session_index": report["session_index"], "input_group": name,
                                "shared_base_costs": True})
    by_case = {c["case"]: [] for c in manifest["cases"]}
    for report in reports:
        for case in report["cases"]:
            by_case[case["case"]].append(tuple(i["input_sha256"] for i in case["input_identities"]))
    if any(len(set(values)) != 3 for values in by_case.values()):
        raise ValueError("At least one case reused its complete input set across sessions")
    return {"transpose_pair_checks": pair_checks, "fresh_input_sets_across_sessions": True}


def method_summary(cases, method):
    per_session = []
    for session, case in enumerate(cases, 1):
        diagnostics = [item["methods"][method] for item in case["diagnostics"]]
        row_rejection = [d["row_rejection_fraction"] for d in diagnostics]
        smaller_rejection = [d["smaller_side_unmatched_fraction"] for d in diagnostics]
        per_session.append({
            "session_index": session, "median_us": case["methods"][method]["median_us"],
            "q1_us": case["methods"][method]["q1_us"], "q3_us": case["methods"][method]["q3_us"],
            "row_rejection_fractions": row_rejection,
            "smaller_side_unmatched_fractions": smaller_rejection,
            "mean_row_rejection_fraction": statistics.mean(row_rejection),
            "mean_smaller_side_unmatched_fraction": statistics.mean(smaller_rejection),
            "match_counts": [d["pairs"] for d in diagnostics],
            "objectives": [d["objective"] for d in diagnostics],
            "diagnostic_coverage": case["diagnostic_coverage"][method],
        })
    times = [s["median_us"] for s in per_session]
    row_means = [s["mean_row_rejection_fraction"] for s in per_session]
    smaller_means = [s["mean_smaller_side_unmatched_fraction"] for s in per_session]
    return {
        "per_session": per_session,
        "median_of_session_medians_us": statistics.median(times),
        "min_session_median_us": min(times), "max_session_median_us": max(times),
        "median_session_row_rejection_fraction": statistics.median(row_means),
        "min_session_row_rejection_fraction": min(row_means), "max_session_row_rejection_fraction": max(row_means),
        "median_session_smaller_side_unmatched_fraction": statistics.median(smaller_means),
        "min_session_smaller_side_unmatched_fraction": min(smaller_means),
        "max_session_smaller_side_unmatched_fraction": max(smaller_means),
    }


def main(argv=None):
    global np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    import numpy as numpy
    np = numpy
    loaded = [(path.resolve(), json.loads(path.read_text(encoding="utf-8"))) for path in args.reports]
    loaded.sort(key=lambda item: item[1]["session_index"])
    paths, reports = [item[0] for item in loaded], [item[1] for item in loaded]
    manifest, signatures = check_reports(paths, reports)
    pairing = verify_input_pairing(manifest, reports)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = (args.output or ROOT / "results" / ("rectangular-summary-" + stamp)).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    band = manifest["practical_ratio_band"]
    summary = {
        "schema_version": 1, "status": "started", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": manifest, "signatures": signatures, "input_pairing": pairing,
        "summarizer_sha256": digest(Path(__file__)), "numpy_version": np.__version__,
        "primary_comparison": manifest["primary_comparison"],
        "sessions": [{"session_index": r["session_index"], "report": str(p), "report_sha256": digest(p),
                      "pid": r["pid"], "started_utc": r["started_utc"], "completed_utc": r["completed_utc"],
                      "affinity": r["affinity"], "versions": r["versions"]} for p, r in loaded],
        "same_utc_day": len({r["started_utc"].split("T")[0] for r in reports}) == 1,
        "replication_scope": "Three fresh processes on the same host in the current day/window; no designed or recorded independent-day/reboot replication",
        "statistics": {
            "per_session_point": "exp(mean(log(numerator_round/denominator_round))) using matched round indices within one session",
            "per_session_interval": "Exploratory 95% percentile bootstrap of paired log-ratios; 10000 IID pair resamples; serial correlation and thermal drift can invalidate nominal coverage",
            "across_sessions": "Median and observed minimum/maximum of the three separate session point estimates, not an interval on an underlying population parameter",
            "no_pooling": "The 27 rounds are never pooled as independent observations across sessions",
            "practical_ratio_band": band,
            "equivalence_caution": "Point estimates inside the 3% band are descriptive. Interval containment is reported separately and is conditional on the exploratory bootstrap assumptions, not proof of population-level equivalence.",
        },
        "cases": [],
    }
    write_json(dest / "summary.json", summary)
    all_rows, method_rows, table_rows = [], [], []
    try:
        for index, spec in enumerate(manifest["cases"]):
            cases = [report["cases"][index] for report in reports]
            entry = {**spec, "methods": {}, "comparisons": {}}
            for method in manifest["methods"]:
                entry["methods"][method] = method_summary(cases, method)
                for session in entry["methods"][method]["per_session"]:
                    coverage = session["diagnostic_coverage"]
                    method_rows.append({
                        "case": spec["case"], "block": spec["block"], "regime": spec["regime"],
                        "rows": spec["shape"][0], "columns": spec["shape"][1], "limit": spec["limit"],
                        "paper_subset": spec["paper_subset"], "session_index": session["session_index"],
                        "method": method, "label": manifest["methods"][method]["label"],
                        "median_us": session["median_us"], "q1_us": session["q1_us"], "q3_us": session["q3_us"],
                        "mean_row_rejection_fraction": session["mean_row_rejection_fraction"],
                        "mean_smaller_side_unmatched_fraction": session["mean_smaller_side_unmatched_fraction"],
                        "searches": coverage.get("searches"), "dual_updates": coverage.get("dual_updates"),
                        "cache_fast_path": coverage.get("cache_fast_path"), "cache_fallback": coverage.get("cache_fallback"),
                        "cache_fast_path_share": coverage.get("cache_fast_path_share"),
                    })
            for name, comparison in manifest["comparisons"].items():
                sessions = []
                for session, case in enumerate(cases, 1):
                    values = paired_statistics(
                        case["methods"][comparison["numerator"]]["round_average_us"],
                        case["methods"][comparison["denominator"]]["round_average_us"],
                        seed_for(f"{summary['signatures']['manifest_sha256']}|{spec['case']}|{session}|{name}"), band)
                    sessions.append({"session_index": session, **values})
                points = [s["paired_log_central_ratio"] for s in sessions]
                central = statistics.median(points)
                ratio = {**comparison, "per_session": sessions, "median_estimate": central,
                         "min_session": min(points), "max_session": max(points),
                         "median_point_band_class": "denominator_faster" if central > band[1] else (
                             "denominator_slower" if central < band[0] else "point_within_practical_band"),
                         "all_session_intervals_within_band": all(s["interval_within_practical_band"] for s in sessions),
                         "all_session_points_above_one": all(p > 1 for p in points),
                         "all_session_points_below_one": all(p < 1 for p in points)}
                entry["comparisons"][name] = ratio
                row = {"case": spec["case"], "block": spec["block"], "family": spec["family"],
                       "regime": spec["regime"], "rows": spec["shape"][0], "columns": spec["shape"][1],
                       "limit": spec["limit"], "paper_subset": spec["paper_subset"], "comparison": name,
                       "numerator": comparison["numerator"], "denominator": comparison["denominator"],
                       "median_estimate": central, "min_session": min(points), "max_session": max(points),
                       "median_point_band_class": ratio["median_point_band_class"],
                       "all_session_intervals_within_band": ratio["all_session_intervals_within_band"]}
                for session in sessions:
                    prefix = f"s{session['session_index']}"
                    row[prefix + "_ratio"] = session["paired_log_central_ratio"]
                    row[prefix + "_exploratory_low"] = session["exploratory_interval"]["lower"]
                    row[prefix + "_exploratory_high"] = session["exploratory_interval"]["upper"]
                    row[prefix + "_point_band_class"] = session["point_band_class"]
                all_rows.append(row)
            if spec["paper_subset"]:
                padded = entry["methods"]["specialized_padded"]
                table = {"case": spec["case"], "block": spec["block"], "regime": spec["regime"],
                         "rows": spec["shape"][0], "columns": spec["shape"][1], "limit": spec["limit"],
                         "row_rejected_pct": 100 * padded["median_session_row_rejection_fraction"],
                         "smaller_side_unmatched_pct": 100 * padded["median_session_smaller_side_unmatched_fraction"]}
                for letter, method in (("O", "original"), ("U", "one_sided_materialized"),
                                       ("P", "specialized_padded"), ("T", "specialized_cached")):
                    table[letter + "_ms"] = entry["methods"][method]["median_of_session_medians_us"] / 1000
                for label, name in (("O_U", "original_over_one_sided"), ("U_P", "one_sided_over_padded"),
                                    ("O_P", "original_over_padded"), ("P_T", "padded_over_compact")):
                    for field in ("median_estimate", "min_session", "max_session"):
                        table[label + "_" + field] = entry["comparisons"][name][field]
                table_rows.append(table)
            summary["cases"].append(entry)
        counts = []
        for block in sorted({case["block"] for case in summary["cases"]}):
            cases = [case for case in summary["cases"] if case["block"] == block]
            for comparison in manifest["comparisons"]:
                counts.append({"block": block, "comparison": comparison, "cases": len(cases),
                    **{category: sum(c["comparisons"][comparison]["median_point_band_class"] == category for c in cases)
                       for category in ("denominator_faster", "denominator_slower", "point_within_practical_band")},
                    "all_session_intervals_within_band": sum(c["comparisons"][comparison]["all_session_intervals_within_band"] for c in cases)})
        summary["block_descriptive_counts"] = counts
        summary["status"] = "complete"
        summary["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(dest / "summary.json", summary)
        write_csv(dest / "all_cases.csv", all_rows)
        write_csv(dest / "all_case_methods.csv", method_rows)
        write_csv(dest / "paper_table.csv", table_rows)
        write_csv(dest / "block_counts.csv", counts)
        print(redact(f"Summarized all {len(summary['cases'])} cases and {len(table_rows)} predeclared paper rows: {dest}"))
        return 0
    except Exception as error:
        summary["status"] = "failed"
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        write_json(dest / "summary.json", summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
