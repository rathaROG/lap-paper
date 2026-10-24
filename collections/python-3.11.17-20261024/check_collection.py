"""Audit the 3.11 replication without changing or checking manuscript claims."""
from collections import Counter
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import check_artifact as base_checks
import check_extended_results as extension_checks
from collect import preserved

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[1]
RESULTS = ROOT / "results"
EXTENDED = ROOT / "extended"
OLD_EXTENDED = ORIGINAL / "extensions/python-3.14.9-20261024"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def audit(data_only=False):
    expected = {"lapx": "0.11.1", "lapcpp": "1.0.0", "laprust": "1.0.0"}
    runtime = read(RESULTS / "runtime.json")
    assert sys.version_info[:3] == (3, 11, 17) and sys.version == runtime["python"]
    assert Path(sys.executable).resolve() == Path(r"C:\dev\exc\python\p311\python.exe").resolve()
    assert {k: v["version"] for k, v in runtime["packages"].items()} == expected
    assert runtime["record_script_sha256"] == digest(ROOT / "record_runtime.py")
    for binary in runtime["native_modules"].values():
        assert binary["sha256"] == digest(Path(binary["path"]))
    adaptations = read(ROOT / "source_adaptations.json")
    for name, value in adaptations.items():
        assert value["original_sha256"] == digest(ORIGINAL / name)
        assert value["collection_sha256"] == digest(ROOT / name)
    for name in ("run_experiments.py", "compare_optional.py", "benchmark_scaling.py"):
        assert digest(ROOT / name) == digest(ORIGINAL / name)
    environment = read(RESULTS / "environment.json")
    old_environment = read(ORIGINAL / "results/environment.json")
    for relative in ("environment.json", "formulations/environment.json", "scaling/environment.json"):
        current = read(RESULTS / relative)
        old = read(ORIGINAL / "results" / relative)
        base_checks.check_binaries(current, runtime)
        assert current["python"] == runtime["python"]
        assert {k: v["version"] for k, v in current["packages"].items()} == expected
        assert current["numpy"] == environment["numpy"] and current["scipy"] == environment["scipy"]
        assert current["experiment_sha256"] == digest(ROOT / "run_experiments.py")
        for key in ("seed", "rounds", "instances_per_single_case", "block_seconds", "single_affinity", "batch_affinity"):
            if key in current:
                assert current[key] == old[key], (relative, key)
    assert read(RESULTS / "formulations/environment.json")["supplement_sha256"] == digest(ROOT / "compare_optional.py")
    scaling = read(RESULTS / "scaling/environment.json")
    assert scaling["scaling_script_sha256"] == digest(ROOT / "benchmark_scaling.py")
    assert scaling["optional_script_sha256"] == digest(ROOT / "compare_optional.py")
    assert scaling["status"] == "completed" and scaling["completed_cases"] == scaling["planned_cases"] == 47
    counts = {}
    for name, sample, summary, expected_count in (
        ("single", "single_samples.csv", "single_summary.csv", 756),
        ("batch", "batch_samples.csv", "batch_summary.csv", 270),
        ("formulations", "formulations/samples.csv", "formulations/summary.csv", 450),
        ("scaling", "scaling/samples.csv", "scaling/summary.csv", 2718),
    ):
        raw, summaries = rows(RESULTS / sample), rows(RESULTS / summary)
        base_checks.check_summary(raw, summaries, 9)
        assert len(raw) == expected_count
        assert all(math.isfinite(float(r["nanoseconds"])) and float(r["nanoseconds"]) > 0 for r in raw)
        rounds = Counter((r["case"], r["solver"], int(r["round"])) for r in raw)
        assert set(rounds.values()) == {1} and {r[2] for r in rounds} == set(range(9))
        schedule = lambda data: {(r["case"], r["solver"], int(r["round"])): int(r["position"]) for r in data}
        assert schedule(raw) == schedule(rows(ORIGINAL / "results" / sample))
        counts[name] = dict(samples=len(raw), settings=len(summaries))
    validation = read(RESULTS / "validation.json")
    assert not validation["failures"] and validation["maximum_absolute_objective_error"] == 0
    assert validation["solver_calls"] == 16800 and validation["infeasible_calls"] == 80
    optional = read(RESULTS / "formulations/validation.json")
    assert optional["status"] == "passed" and optional["maximum_absolute_grid_objective_error"] == 0
    assert optional["grid_solver_calls"] + optional["targeted_solver_calls"] == 10520
    batch = read(RESULTS / "batch_validation.json")
    base_checks.check_batch_validation(batch, environment, runtime)
    benchmark = read(RESULTS / "benchmark_checks.json")
    assert benchmark["batch"] == batch["checks"]
    assert all(r["objective_checks_passed"] for group in benchmark.values() for r in group)
    for suite, count in (("formulations", 200), ("scaling", 1208)):
        checks = read(RESULTS / suite / "checks.json")
        assert len(checks) == count
        assert all(math.isclose(r["objective"], r["reference_objective"], rel_tol=1e-10, abs_tol=1e-10) for r in checks)
    memory = read(RESULTS / "memory.json")
    assert len(memory) == 10
    assert all(r["additional_peak"] == r["peak_after"] - r["peak_before"] >= 0 for r in memory)
    release = read(RESULTS / "test_validation.json")
    assert release["packages"] == expected and len(release["suites"]) == 8
    assert all(r["exit_code"] == 0 for r in release["suites"].values())
    assert "100% tests passed" in (RESULTS / "validation_logs/lapcpp_native.log").read_text()
    assert re.search(r"\b17 passed\b", (ROOT / "logs/paper_harness.log").read_text())
    assert "No broken requirements found." in (ROOT / "logs/pip_check.log").read_text()
    run = read(ROOT / "collection_run.json")
    assert run["status"] == "completed"
    for index, stage in enumerate(run["stages"]):
        assert stage["status"] == "completed" and stage["exit_code"] == 0
        assert Path(stage["command"][0]).resolve() == Path(sys.executable).resolve()
        if index:
            assert stage["started_utc"] >= run["stages"][index-1]["completed_utc"]
    base_run = read(RESULTS / "rerun/rerun.json")
    assert base_run["status"] == "completed" and len(base_run["suites"]) == 3
    for stage in base_run["suites"]:
        assert stage["exit_code"] == 0 and stage["script_sha256"] == digest(Path(stage["command"][1]))
        assert Path(stage["command"][0]).resolve() == Path(sys.executable).resolve()
    cpu = rows(RESULTS / "rerun/cpu_load.csv")
    cpu_summary = {}
    for stage in ("baseline", "main", "optional", "scaling"):
        values = [float(r["mean_outside_cpu0_3_percent"]) for r in cpu if r["stage"] == stage]
        assert values
        cpu_summary[stage] = dict(samples=len(values), median_outside_cpu0_3_percent=statistics.median(values))
    extension = extension_checks.audit(EXTENDED)
    assert extension["status"] == "passed" and extension["timing_samples"] == 4752
    assert read(EXTENDED / "plan.json") == read(OLD_EXTENDED / "plan.json")
    assert schedule(rows(EXTENDED / "samples.csv")) == schedule(rows(OLD_EXTENDED / "samples.csv"))
    paired_inputs = {}
    for suite in ("formulations", "scaling"):
        current = {r["case"]: r["input_sha256"] for r in read(RESULTS / suite / "datasets.json")}
        old = {r["case"]: r["input_sha256"] for r in read(ORIGINAL / "results" / suite / "datasets.json")}
        assert current == old
        paired_inputs[suite] = sum(len(v) for v in current.values())
    current = {r["case"]: [i["sha256"] for i in r["inputs"]] for r in read(EXTENDED / "datasets.json")}
    old = {r["case"]: [i["sha256"] for i in r["inputs"]] for r in read(OLD_EXTENDED / "datasets.json")}
    assert current == old
    paired_inputs["extended_timing"] = sum(len(v) for v in current.values())
    current = {r["job"]: r["input_sha256"] for r in read(EXTENDED / "memory_samples.json")}
    old = {r["job"]: r["input_sha256"] for r in read(OLD_EXTENDED / "memory_samples.json")}
    assert current == old
    paired_inputs["extended_memory_trials"] = len(current)
    old_batch = read(ORIGINAL / "results/batch_validation.json")
    assert [r["input_sha256"] for r in batch["checks"]] == [r["input_sha256"] for r in old_batch["checks"]]
    paired_inputs["batches"] = len(batch["checks"])
    figures = []
    if not data_only:
        review = read(ROOT / "figure_review.json")
        assert review["status"] == "passed" and len(review["figures"]) == 19
        assert review["chart_packet"]["pages"] == 19 and review["chart_packet"]["rendered_pages_identical"]
        assert digest(ROOT / "charts.pdf") == review["chart_packet"]["sha256"]
        for figure in review["figures"]:
            assert figure["pdf_pages"] == 1 and figure["svg_parse"] == "passed"
            for name, expected_digest in figure["sha256"].items():
                assert digest(ROOT / name) == expected_digest
            figures.append(figure["stem"])
    report = dict(status="passed", audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Completed Python 3.11.17 measurements; manuscript claims and arXiv build remain deferred.",
        python=runtime["python"], executable=sys.executable, packages=expected,
        dependencies={"numpy":environment["numpy"], "scipy":environment["scipy"]},
        previous_dependencies={"numpy":old_environment["numpy"], "scipy":old_environment["scipy"]},
        base_suites=counts, extended=extension, total_timing_samples=4194+extension["timing_samples"],
        total_memory_trials=10+extension["memory_trials"], enumeration_solver_calls=27320,
        batch_outputs_checked=batch["checked_outputs"], package_suites=release["suites"], harness_tests_passed=17,
        same_input_hashes_as_python314=paired_inputs, base_cpu_monitor=cpu_summary,
        same_method_order_as_python314=["single", "batch", "formulations", "scaling", "extended"],
        previous_files_unchanged=preserved(), paper_updated=False, figures_reviewed=figures,
        caveat="Separate environment runs, with different NumPy/SciPy versions and native wheels. Differences cannot be attributed to Python version alone.")
    save(ROOT / "audit.json", report)
    inventory = {p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name != "file_manifest.json"
        and not any(x in p.relative_to(ROOT).parts for x in ("__pycache__", ".pytest_cache", "tmp"))}
    save(ROOT / "file_manifest.json", inventory)
    print(json.dumps({k: report[k] for k in ("status", "total_timing_samples", "total_memory_trials",
          "same_input_hashes_as_python314", "previous_files_unchanged", "figures_reviewed")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true")
    audit(parser.parse_args().data_only)
