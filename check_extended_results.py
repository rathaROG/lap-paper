"""Audit the extension independently of the frozen paper's tables and claims."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

PAPER = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(output):
    plan, run = read(output / "plan.json"), read(output / "run.json")
    cfg = plan["configuration"]
    assert run["status"] in ("completed", "completed_with_exclusions"), run["status"]
    assert cfg["rounds"] == 9 and cfg["instances"] == 4
    environment = read(output / "environment.json")
    assert environment["python"].startswith("3.14.9 ")
    assert Path(environment["executable"]).resolve() == Path(r"C:\dev\exc\python\p314\python.exe").resolve()
    baseline = read(PAPER / "results" / "runtime.json")
    for name, item in environment["native_modules"].items():
        assert item["sha256"] == baseline["native_modules"][name]["sha256"] == digest(Path(item["path"]))
    for name, expected in run["source_sha256"].items():
        assert digest(PAPER / name) == digest(output / "sources" / name) == expected
    preserved = read(output / "protected_files.json")
    for name, expected in preserved.items():
        assert digest(PAPER / name) == expected, name
    raw = csv_rows(output / "samples.csv")
    summaries = csv_rows(output / "summary.csv")
    groups = defaultdict(list)
    for row in raw:
        assert math.isfinite(float(row["nanoseconds"])) and float(row["nanoseconds"]) > 0
        assert int(row["instances"]) == 4 and int(row["cycles"]) >= 1
        groups[row["case"], row["solver"]].append(row)
    reported = {(r["case"], r["solver"]):r for r in summaries}
    assert len(reported) == len(summaries)
    valid_keys, excluded = set(), []
    pairing = {}
    all_checks = []
    case_files = sorted((output / "cases").glob("*.json"))
    assert len(case_files) == len(plan["timing"])
    for spec in plan["timing"]:
        data = read(output / "cases" / (spec["case"] + ".json"))
        assert data["spec"] == spec
        assert len(data["inputs"]) == 4
        if data["status"] == "timeout":
            excluded.append(dict(case=spec["case"], reason="whole-case timeout"))
            continue
        assert data["status"] == "completed"
        expected_methods = plan["protocol"]["required_methods" if spec["limit"] is None else "optional_methods"]
        assert set(data["methods"]) == set(expected_methods)
        counts = Counter(row["solver"] for row in data["checks"])
        assert counts == {method:4 for method in expected_methods}
        for row in data["checks"]:
            assert math.isclose(row["objective"], row["reference_objective"], rel_tol=1e-10, abs_tol=1e-10)
        all_checks.extend(data["checks"])
        # Required/finite regimes, thresholds and transpose pairs reuse the intended inputs.
        key = tuple((name, spec.get(name)) for name in
                    ("input_key", "distribution", "signed", "levels", "row_fraction", "column_fraction", "forbidden", "planted_feasible"))
        canonical = [row["canonical_sha256"] for row in data["inputs"]]
        if key in pairing:
            assert pairing[key] == canonical, spec["case"]
        pairing[key] = canonical
        for method, status in data["methods"].items():
            group_key = spec["case"], method
            if status["status"] != "completed":
                assert status["status"] in ("budget_excluded", "budget_exceeded")
                assert group_key not in reported
                excluded.append(dict(case=spec["case"], solver=method, reason=status["status"]))
                continue
            valid_keys.add(group_key)
            values = groups[group_key]
            assert len(values) == 9 and {int(r["round"]) for r in values} == set(range(9))
            assert len({int(r["cycles"]) for r in values}) == 1
            times = sorted(float(r["nanoseconds"]) for r in values)
            summary = reported[group_key]
            assert int(summary["rounds"]) == 9
            for name, value in (("median_ns", times[4]), ("q25_ns", times[2]), ("q75_ns", times[6])):
                assert math.isclose(float(summary[name]), value, rel_tol=1e-12)
    assert valid_keys == set(reported)
    assert len(all_checks) == run["objective_checks"]
    assert len(raw) == run["timing_samples"]
    memory = read(output / "memory_samples.json")
    assert len(memory) == len(plan["memory"]) == run["memory_trials"]
    assert len({row["job"] for row in memory}) == len(memory)
    assert {row["job"] for row in memory} == {row["job"] for row in plan["memory"]}
    memory_groups = defaultdict(list)
    memory_pairing = {}
    for row in memory:
        assert row["status"] == "passed"
        assert Path(row["python_executable"]).resolve() == Path(environment["executable"]).resolve()
        assert row["additional_peak"] == max(0, row["peak_after"]-row["peak_before"])
        assert row["input_bytes"] == row["n"]*row["m"]*8
        assert math.isclose(row["objective"], row["reference_objective"], rel_tol=1e-10, abs_tol=1e-10)
        key = row["case"], row["trial"]
        if key in memory_pairing:
            assert memory_pairing[key] == row["input_sha256"]
        memory_pairing[key] = row["input_sha256"]
        memory_groups[row["case"], row["solver"]].append(row)
    memory_summaries = csv_rows(output / "memory_summary.csv")
    assert len(memory_summaries) == len(memory_groups)
    for summary in memory_summaries:
        group = memory_groups[summary["case"], summary["solver"]]
        trials = 2 if cfg["smoke"] else 5
        assert len(group) == int(summary["trials"]) == trials
        assert {row["trial"] for row in group} == set(range(trials))
        values = sorted(row["additional_peak"] for row in group)
        assert float(summary["median_bytes"]) == statistics.median(values)
        if trials == 5:
            assert float(summary["q25_bytes"]) == values[1] and float(summary["q75_bytes"]) == values[3]
    edges = read(output / "edge_validation.json")
    assert edges["status"] == "passed"
    for row in edges["checks"]:
        assert row["status"] in ("passed", "expected_infeasible")
        if row["status"] == "passed":
            assert math.isclose(row["objective"], row["reference_objective"], rel_tol=1e-10, abs_tol=1e-10)
    cpu = csv_rows(output / "cpu_load.csv")
    outside = [float(row["mean_outside_cpu0_3_percent"]) for row in cpu]
    assert outside
    paired_ratios = 0
    for filename, source, numerator in (
        ("paired_formulation_speedups.json", PAPER / "results/formulations/samples.csv", "scipy_capped"),
        ("paired_implementation_speedups.json", output / "samples.csv", "jvrust"),
    ):
        if not (output / filename).exists():
            continue
        values = {(r["case"], r["solver"], int(r["round"])): float(r["nanoseconds"])
                  for r in csv_rows(source)}
        for row in read(output / filename):
            ratios = sorted(values[row["case"], numerator, i] / values[row["case"], row["solver"], i]
                            for i in range(9))
            for name, index in (("q25", 2), ("median_paired_speedup", 4), ("q75", 6)):
                assert math.isclose(row[name], ratios[index], rel_tol=1e-12)
            paired_ratios += 1
    figure_count = 0
    if (output / "figure_manifest.json").exists():
        figures = read(output / "figure_manifest.json")
        assert figures["plot_script_sha256"] == digest(PAPER / "plot_extended_results.py")
        assert figures["plot_script_sha256"] == digest(output / "sources/plot_extended_results.py")
        figure_count = len(figures["figures"])
        assert figure_count == figures["total_figures"] == 13
        for item in figures["figures"]:
            assert len(item["files"]) == 3
            for name in item["files"]:
                assert (output / "figures" / name).stat().st_size > 0
        if (output / "figure_review.json").exists():
            review = read(output / "figure_review.json")
            assert review["status"] == "passed" and len(review["figures"]) == figure_count
            for item in review["figures"]:
                assert item["pdf_pages"] == 1 and item["svg_parse"] == "passed"
                for name, expected in item["sha256"].items():
                    assert digest(output / "figures" / name) == expected
    report = dict(status="passed" if not excluded else "passed_with_explicit_exclusions", audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        timing_cases=len(case_files), timing_samples=len(raw), complete_method_settings=len(valid_keys),
        timed_input_objective_checks=len(all_checks), memory_trials=len(memory), memory_conditions=len(memory_groups),
        edge_instances=edges["instances"], edge_solver_checks=len(edges["checks"]),
        expected_infeasible_checks=sum(r["status"]=="expected_infeasible" for r in edges["checks"]),
        existing_files_unchanged=len(preserved), python_3_11_run=False, paper_updated=False,
        figures=figure_count, paired_ratio_summaries_checked=paired_ratios,
        exclusions=excluded, cpu_monitor_samples=len(outside), median_outside_cpu0_3_percent=statistics.median(outside),
        scope="Data, paired inputs, summaries, provenance and preservation. The unchanged manuscript is not checked against the new measurements.")
    (output / "audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    inventory = {p.relative_to(output).as_posix():dict(bytes=p.stat().st_size, sha256=digest(p))
                 for p in sorted(output.rglob("*")) if p.is_file() and p.name != "file_manifest.json"}
    (output / "file_manifest.json").write_text(json.dumps(inventory, indent=2)+"\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(audit(parser.parse_args().output.resolve()), indent=2))
