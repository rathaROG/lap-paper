"""Independent, standard-library audit of new four-suite paper measurements.

No solver, benchmark driver, or third-party package is imported or executed.
Usage: python audit_controlled_update.py py311-results [py314-results] --output audit.json

This is the unchanged audit logic preserved with the 29 October 2026 campaign;
the public entry point verify_results.py also checks the packaged manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


SUITES = ("base", "formulations", "scaling", "extended")
TABLES = (
    ("base/single", 12, 84), ("base/batch", 3, 30),
    ("formulations", 10, 50), ("scaling", 47, 302), ("extended", 132, 528),
)
DIGESTS = {
    "single": "605a4fddc780f4f7be52619395830e7906184bffaabbb8b0a137eae132605b95",
    "batch": "2d153f269137aaa93c655092c5f97f3274b36a75f683d263b58172125805a292",
    "formulations": "db48a7d60d31cf6f6a8151cc49c211fb4ecb009e68c94cfb6ebc301913f2c095",
    "scaling": "016640124330696fd46ac68c08040e4fbb16790a85c7194697e0eca4bc0a7c72",
    "extended_timing": "33a83919da40b5503bcfe61baa9def3859690ceb7f0db7c0f1d9d32f179402c8",
    "extended_memory": "5ab6c267f46e4627130aa97bf51ece7c63db72c96a424b1bd257883b36ea6842",
}
PACKAGES = {"lapx": "0.13.0", "laprust": "1.2.0", "lapcpp": "1.2.0"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def quartiles(values):
    values = sorted(values)
    require(bool(values), "Empty summary group")
    def percentile(p):
        i = (len(values) - 1) * p
        lo, hi = math.floor(i), math.ceil(i)
        return values[lo] + (values[hi] - values[lo]) * (i - lo)
    return statistics.median(values), percentile(.25), percentile(.75)


def close(actual, expected, context, *, rtol=2e-12, atol=1e-12):
    actual, expected = float(actual), float(expected)
    require(math.isfinite(actual) and math.isfinite(expected), context + ": nonfinite value")
    require(math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol),
            f"{context}: {actual!r} differs from {expected!r}")


def input_digest(pairs):
    return hashlib.sha256("\n".join(f"{case}:{digest}" for case, digest in pairs).encode()).hexdigest()


def objective_records(records, prefix):
    result = {}
    for row in records:
        key = (prefix, row["case"], row["solver"], row.get("instance", row.get("trial", 0)))
        require(key not in result, f"Duplicate objective record {key}")
        close(row["objective"], row["reference_objective"], str(key), rtol=1e-10, atol=1e-10)
        result[key] = (float(row["objective"]), row.get("pairs"))
    return result


def audit(folder):
    folder = folder.resolve()
    launch = read(folder / "run_all.json")
    require(launch["status"] == "completed" and launch["smoke"] is False,
            "Expected a completed full campaign, not a smoke run")
    require([r["suite"] for r in launch["suites"]] == list(SUITES), "Suite order/coverage differs")
    require(all(r["exit_code"] == 0 for r in launch["suites"]), "Suite failed")
    totals = {"cases": 0, "settings": 0, "timing_samples": 0}
    table_reports, schedules = {}, {}
    for table, cases_expected, groups_expected in TABLES:
        base = table.startswith("base/")
        sample_file = folder / (table + "_samples.csv" if base else table + "/samples.csv")
        summary_file = folder / (table + "_summary.csv" if base else table + "/summary.csv")
        samples, summaries = rows(sample_file), rows(summary_file)
        groups, order = defaultdict(list), defaultdict(list)
        seen = set()
        for row in samples:
            key = (row["case"], row["solver"])
            identity = (*key, int(row["round"]))
            require(identity not in seen, f"Duplicate timing round {table}/{identity}")
            seen.add(identity)
            value = float(row["nanoseconds"])
            require(math.isfinite(value) and value > 0, f"Invalid timing {table}/{identity}")
            require(int(row["cycles"]) > 0, "Nonpositive calibrated cycle count")
            groups[key].append(value)
            order[(row["case"], int(row["round"]))].append((int(row["position"]), row["solver"]))
        cases = {case for case, _ in groups}
        require(len(cases) == cases_expected and len(groups) == groups_expected, f"Coverage differs: {table}")
        require(len(samples) == groups_expected * 9, f"Sample count differs: {table}")
        summary_keys = [(r["case"], r["solver"]) for r in summaries]
        require(len(set(summary_keys)) == len(summary_keys) == len(groups), f"Summary keys differ: {table}")
        for row in summaries:
            values = groups[(row["case"], row["solver"])]
            require(len(values) == int(row["rounds"]) == 9, f"Incomplete rounds: {table}")
            for field, value in zip(("median_ns", "q25_ns", "q75_ns"), quartiles(values)):
                close(row[field], value, f"{table}/{row['case']}/{row['solver']}/{field}")
        for key in groups:
            require({n for c, s, n in seen if (c, s) == key} == set(range(9)), "Round indices differ")
        for key, schedule in order.items():
            require(sorted(pos for pos, _ in schedule) == list(range(len(schedule))), "Method-order positions differ")
            schedules[(table, *key)] = sorted(schedule)
        totals["cases"] += len(cases)
        totals["settings"] += len(groups)
        totals["timing_samples"] += len(samples)
        table_reports[table] = {"cases": len(cases), "settings": len(groups), "samples": len(samples)}
    require(totals == {"cases": 204, "settings": 994, "timing_samples": 8946}, "Overall coverage differs")

    validation = read(folder / "base/validation.json")
    require(not validation["failures"] and validation["solver_calls"] == 16800
            and validation["infeasible_calls"] == 80, "Base enumeration differs")
    require(validation["maximum_absolute_objective_error"] == 0, "Base enumeration objective error")
    form_validation = read(folder / "formulations/validation.json")
    require(form_validation["status"] == "passed" and form_validation["grid_solver_calls"] == 10500
            and form_validation["targeted_solver_calls"] == 20, "Formulation validation differs")
    require(form_validation["maximum_absolute_grid_objective_error"] == 0, "Formulation objective error")
    base_checks = read(folder / "base/benchmark_checks.json")
    require(len(base_checks["single"]) == 12 and len(base_checks["batch"]) == 3, "Base check coverage differs")
    require(all(r["objective_checks_passed"] is True for section in base_checks.values() for r in section),
            "Base objective check did not pass")
    require(sum(r["instances"] * r["solvers"] for r in base_checks["single"]) == 336,
            "Base single check count differs")
    require(sum(sum(r["checked_outputs"].values()) for r in base_checks["batch"]) == 11040,
            "Batch output check count differs")
    edges = read(folder / "extended/edge_validation.json")
    require(edges["status"] == "passed" and edges["instances"] == 23 and len(edges["checks"]) == 129,
            "Edge-validation coverage differs")
    require(Counter(r["status"] for r in edges["checks"]) == {"passed": 114, "expected_infeasible": 15},
            "Unexpected edge-check status")
    for r in edges["checks"]:
        if r["status"] == "passed":
            close(r["objective"], r["reference_objective"], "edge objective", rtol=1e-10, atol=1e-10)

    extended_run = read(folder / "extended/run.json")
    require(extended_run["status"] == "completed" and not extended_run["failures"]
            and extended_run["excluded_method_settings"] == 0, "Extended run incomplete or excluded methods")
    statuses = read(folder / "extended/method_status.json")
    require(len(statuses) == 528 and all(r["status"] == "completed" for r in statuses), "Extended method status differs")
    objectives = {}
    for suite, expected in (("formulations", 200), ("scaling", 1208), ("extended", 2112)):
        checks = read(folder / suite / "checks.json")
        require(len(checks) == expected, f"Objective coverage differs: {suite}")
        objectives.update(objective_records(checks, suite))
    base_memory = read(folder / "base/memory.json")
    memory = read(folder / "extended/memory_samples.json")
    require(len(base_memory) == 10 and len(memory) == 405, "Memory coverage differs")
    require(all(r["status"] == "passed" for r in memory), "Memory worker did not pass")
    for r in base_memory + memory:
        require(r["additional_peak"] == max(0, r["peak_after"] - r["peak_before"]), "Memory peak arithmetic differs")
    objectives.update(objective_records(memory, "memory"))
    memory_groups = defaultdict(list)
    for r in memory:
        memory_groups[(r["case"], r["solver"])].append(r)
    memory_summaries = rows(folder / "extended/memory_summary.csv")
    require(len(memory_groups) == len(memory_summaries) == 81, "Memory-summary coverage differs")
    for r in memory_summaries:
        group = memory_groups[(r["case"], r["solver"])]
        require(len(group) == int(r["trials"]) == 5 and {g["trial"] for g in group} == set(range(5)),
                "Memory trial coverage differs")
        values = [g["additional_peak"] for g in group]
        for field, value in zip(("median_bytes", "q25_bytes", "q75_bytes"), quartiles(values)):
            close(r[field], value, "memory " + field)
        require(int(r["min_bytes"]) == min(values) and int(r["max_bytes"]) == max(values), "Memory extrema differ")

    datasets = {s: read(folder / s / "datasets.json") for s in SUITES[1:]}
    pairs = {
        "single": [(r["case"], h) for r in base_checks["single"] for h in r["input_sha256"]],
        "batch": [(r["case"], r["input_sha256"]) for r in base_checks["batch"]],
        "formulations": [(r["case"], h) for r in datasets["formulations"] for h in r["input_sha256"]],
        "scaling": [(r["case"], h) for r in datasets["scaling"] for h in r["input_sha256"]],
        "extended_timing": [(r["case"], h["sha256"]) for r in datasets["extended"] for h in r["inputs"]],
        "extended_memory": sorted((r["job"], r["input_sha256"]) for r in memory),
    }
    digests = {name: input_digest(p) for name, p in pairs.items()}
    require(digests == DIGESTS, "Recorded input identities differ from frozen paper workload")
    require(sum(map(len, pairs.values())) == 1212, "Input identity count differs")
    environments = {s: read(folder / s / "environment.json") for s in SUITES}
    first = environments["base"]
    for suite, env in environments.items():
        package_versions = {name: value.get("version") if isinstance(value, dict) else value
                            for name, value in env["packages"].items()}
        require(package_versions == PACKAGES, f"Solver versions differ: {suite}")
        require(env["python"] == first["python"] and env["numpy"] == first["numpy"]
                and env["scipy"] == first["scipy"], "Environment changed within campaign")
        native = env["native_modules"]
        require(len(native) == 7 and all("sha256" in m and "error" not in m for m in native.values()),
                "Native binary identity missing")
        require(native == first["native_modules"], "Native binaries changed within campaign")
        require(all(p["num_threads"] == 1 for p in (env.get("threadpools") or [])), "Library thread count differs")
    report = {"status": "passed", "result_directory": str(folder), **totals,
              "memory_observations": 415, "memory_summaries_recomputed": 81,
              "numeric_objective_records": len(objectives), "base_enumeration_calls": 16800,
              "base_single_objective_checks": 336, "batch_output_checks": 11040,
              "formulation_enumeration_and_targeted_calls": 10520, "edge_checks": 129,
              "frozen_input_identities": 1212, "input_digests": digests, "tables": table_reports,
              "python": first["python"], "numpy": first["numpy"], "scipy": first["scipy"],
              "packages": first["packages"], "native_modules": first["native_modules"],
              "affinity": {k: first.get(k) for k in ("single_affinity", "batch_affinity")},
              "scope": "Saved-data consistency and arithmetic audit, not an independent re-solve or universal correctness proof."}
    return report, objectives, schedules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="one or two new run_all output folders")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    require(1 <= len(options.results) <= 2, "Supply one or two environments")
    audited = [audit(p) for p in options.results]
    result = {"status": "passed", "environments": [r[0] for r in audited]}
    if len(audited) == 2:
        a, b = audited[0][1], audited[1][1]
        require(a.keys() == b.keys() and len(a) == 3925, "Cross-environment objective coverage differs")
        differences, cardinality_differences = [], 0
        for key, (v, cardinality) in a.items():
            other, other_cardinality = b[key]
            close(v, other, "cross-environment objective " + str(key), rtol=1e-10, atol=1e-10)
            if v != other:
                differences.append({"record": list(key), "absolute_difference": abs(v - other)})
            cardinality_differences += cardinality != other_cardinality
        require(audited[0][2] == audited[1][2], "Cross-environment method schedules differ")
        result["cross_environment"] = {"numeric_objectives_compared": len(a),
            "exact_objective_agreements": len(a) - len(differences), "nonzero_differences": differences,
            "maximum_absolute_objective_difference": max((r["absolute_difference"] for r in differences), default=0),
            "recorded_cardinality_differences": cardinality_differences,
            "cardinality_policy": "Differences are descriptive, not failures when objectives agree under ties.",
            "same_schedules": True, "same_frozen_input_digests": True}
    if options.output:
        options.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1)
