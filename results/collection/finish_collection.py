"""Promote and audit this Python 3.14 run without changing the paper files."""
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

PAPER = Path(__file__).resolve().parents[2]
NOTES = Path(__file__).resolve().parent
RUN = PAPER / "tmp" / "python-3.14.9-20261024"
RESULTS = PAPER / "results"
sys.path.insert(0, str(PAPER))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def log_text(path):
    data = path.read_bytes()
    return data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")


def promote():
    run = read(RUN / "rerun.json")
    assert run["status"] == "completed"
    for source, destination in ((RUN / "main", RESULTS),
                                (RUN / "optional", RESULTS / "formulations"),
                                (RUN / "scaling", RESULTS / "scaling")):
        destination.mkdir(exist_ok=True)
        for path in source.iterdir():
            target = destination / path.name
            assert not target.exists(), target
            shutil.copy2(path, target)
    logs = RESULTS / "rerun"
    logs.mkdir()
    for path in RUN.iterdir():
        if path.is_file():
            shutil.copy2(path, logs / path.name)
    print("Promoted completed main, formulation and scaling measurements.", flush=True)


def figures():
    import report_results
    import build_pdf
    import plot_scaling
    report_results.main()
    # Existing plot functions write relative to these module-level output roots.
    # Stage vector exports away from the frozen LaTeX submission project.
    staging = NOTES / "figure-export"
    build_pdf.PROJECT = staging / "vector"
    build_pdf.figures()
    plot_scaling.PAPER = staging
    plot_scaling.generate(RESULTS / "scaling")
    for source in (staging / "vector" / "images", staging / "latex" / "images",
                   staging / "figures"):
        for path in source.iterdir():
            target = PAPER / "figures" / path.name
            assert not target.exists(), target
            shutil.copy2(path, target)
    print("Generated six figures in PNG, SVG and PDF; paper files unchanged.", flush=True)


def audit():
    import check_artifact as checks
    runtime = read(RESULTS / "runtime.json")
    expected = {"lapx": "0.11.1", "lapcpp": "1.0.0", "laprust": "1.0.0"}
    assert {k: v["version"] for k, v in runtime["packages"].items()} == expected
    assert sys.version == runtime["python"] and sys.version_info[:3] == (3, 14, 9)
    assert Path(sys.executable).resolve() == Path(r"C:\dev\exc\python\p314\python.exe").resolve()
    for binary in runtime["native_modules"].values():
        assert digest(Path(binary["path"])) == binary["sha256"]
    assert runtime["record_script_sha256"] == digest(PAPER / "record_runtime.py")
    environment = read(RESULTS / "environment.json")
    for path in (RESULTS / "environment.json", RESULTS / "formulations" / "environment.json",
                 RESULTS / "scaling" / "environment.json"):
        data = read(path)
        checks.check_binaries(data, runtime)
        assert data["python"] == sys.version
        assert {k: v["version"] for k, v in data["packages"].items()} == expected
        assert data["numpy"] == environment["numpy"] and data["scipy"] == environment["scipy"]
        assert data["experiment_sha256"] == digest(PAPER / "run_experiments.py")
        assert data["rounds"] == 9 and data["instances_per_single_case"] == 4
    checks.check_batch_validation(read(RESULTS / "batch_validation.json"), environment, runtime)
    assert read(RESULTS / "formulations" / "environment.json")["supplement_sha256"] == digest(PAPER / "compare_optional.py")
    scaling = read(RESULTS / "scaling" / "environment.json")
    assert scaling["scaling_script_sha256"] == digest(PAPER / "benchmark_scaling.py")
    assert scaling["optional_script_sha256"] == digest(PAPER / "compare_optional.py")
    assert scaling["status"] == "completed" and scaling["completed_cases"] == scaling["planned_cases"] == 47
    counts = {}
    for name, sample, summary in (("single", "single_samples.csv", "single_summary.csv"),
                                  ("batch", "batch_samples.csv", "batch_summary.csv"),
                                  ("formulations", "formulations/samples.csv", "formulations/summary.csv"),
                                  ("scaling", "scaling/samples.csv", "scaling/summary.csv")):
        samples, summaries = rows(RESULTS / sample), rows(RESULTS / summary)
        checks.check_summary(samples, summaries, 9)
        assert all(math.isfinite(float(r["nanoseconds"])) and float(r["nanoseconds"]) > 0 for r in samples)
        rounds = Counter((r["case"], r["solver"], int(r["round"])) for r in samples)
        assert set(rounds.values()) == {1}
        assert {key[2] for key in rounds} == set(range(9))
        counts[name] = {"timing_samples": len(samples), "settings": len(summaries)}
    assert [counts[k]["timing_samples"] for k in counts] == [756, 270, 450, 2718]
    validation = read(RESULTS / "validation.json")
    assert not validation["failures"] and validation["maximum_absolute_objective_error"] == 0
    assert validation["solver_calls"] == 16800 and validation["infeasible_calls"] == 80
    optional = read(RESULTS / "formulations" / "validation.json")
    assert optional["status"] == "passed" and optional["maximum_absolute_grid_objective_error"] == 0
    assert optional["grid_solver_calls"] + optional["targeted_solver_calls"] == 10520
    benchmark = read(RESULTS / "benchmark_checks.json")
    assert all(row["objective_checks_passed"] for group in benchmark.values() for row in group)
    batch = read(RESULTS / "batch_validation.json")
    assert benchmark["batch"] == batch["checks"]
    objective_counts = {}
    for suite in ("formulations", "scaling"):
        data = read(RESULTS / suite / "checks.json")
        assert all(math.isclose(r["objective"], r["reference_objective"], rel_tol=1e-10, abs_tol=1e-10) for r in data)
        objective_counts[suite] = len(data)
        datasets = read(RESULTS / suite / "datasets.json")
        assert all(len(r["input_sha256"]) == 4 for r in datasets)
    assert objective_counts == {"formulations": 200, "scaling": 1208}
    memory = read(RESULTS / "memory.json")
    assert len(memory) == 10
    assert all(row["additional_peak"] == row["peak_after"] - row["peak_before"] >= 0 for row in memory)
    release = read(RESULTS / "test_validation.json")
    assert release["packages"] == expected
    assert len(release["suites"]) == 8 and all(r["exit_code"] == 0 for r in release["suites"].values())
    harness = log_text(RESULTS / "validation_logs" / "paper_harness.log")
    assert re.search(r"\b11 passed\b", harness) and not re.search(r"\d+ failed", harness)
    assert "100% tests passed" in log_text(RESULTS / "validation_logs" / "lapcpp_native.log")
    run = read(RESULTS / "rerun" / "rerun.json")
    assert run["status"] == "completed" and len(run["suites"]) == 3
    for i, suite in enumerate(run["suites"]):
        assert suite["exit_code"] == 0
        assert Path(suite["command"][0]).resolve() == Path(sys.executable).resolve()
        assert suite["script_sha256"] == digest(Path(suite["command"][1]))
        if i:
            assert suite["started_utc"] >= run["suites"][i-1]["completed_utc"]
    cpu = rows(RESULTS / "rerun" / "cpu_load.csv")
    cpu_summary = {}
    for stage in ("baseline", "main", "optional", "scaling"):
        values = [float(r["mean_outside_cpu0_3_percent"]) for r in cpu if r["stage"] == stage]
        assert values, stage
        cpu_summary[stage] = {"samples": len(values), "median_outside_cpu0_3_percent": statistics.median(values),
                              "maximum_outside_cpu0_3_percent": max(values)}
    protected = read(NOTES / "paper_before.json")
    assert all(digest(PAPER / row["path"]) == row["sha256"] for row in protected)
    stems = ("single_runtime", "batch_runtime", "memory_peak", "scaling_required", "scaling_optional", "scaling_blocks")
    figure_paths = [PAPER / "figures" / f"{stem}.{ext}" for stem in stems for ext in ("png", "svg", "pdf")]
    assert all(path.is_file() and path.stat().st_size > 1000 for path in figure_paths)
    assert len(list((PAPER / "figures").iterdir())) == 18
    review = read(RESULTS / "collection" / "figure_review.json")
    assert review["status"] == "passed" and review["pdf_pages"] == {stem: 1 for stem in stems}
    assert "No broken requirements found." in log_text(RESULTS / "validation_logs" / "pip_check.log")
    cleanup = read(NOTES / "cleanup.json")
    report = {
        "status": "passed", "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Fresh Python 3.14.9 measurements only; the manuscript and existing arXiv project/output are frozen.",
        "python_executable": sys.executable, "python": sys.version, "packages": expected,
        "numpy": environment["numpy"], "scipy": environment["scipy"],
        "previous_commit": cleanup["previous_commit"], "cleaned_old_files": len(cleanup["files"]),
        "cleanup_method": cleanup["cleanup_method"],
        "python_3_11_run": "not started", "paper_updated": False,
        "unchanged_paper_files": len(protected), "sample_summary_checks": counts,
        "enumeration_solver_calls": 27320, "strict_batch_outputs": batch["checked_outputs"],
        "objective_checks": objective_counts, "memory_measurements": len(memory),
        "package_test_suites": release["suites"], "cpu_load": cpu_summary,
        "paper_harness_tests_passed": 11,
        "figure_visual_review": review,
        "figures": [p.relative_to(PAPER).as_posix() for p in figure_paths],
        "script_sha256": {p.name: digest(p) for p in PAPER.glob("*.py")},
        "deferred": ["Python 3.11.17 experiments and comparison figure", "Manuscript tables/text and arXiv rebuild", "Full paper/data consistency check after the manuscript is updated"],
    }
    evidence = RESULTS / "collection"
    evidence.mkdir(exist_ok=True)
    write(evidence / "cleanup.json", cleanup)
    write(evidence / "paper_before.json", protected)
    if Path(__file__).resolve() != (evidence / "finish_collection.py").resolve():
        shutil.copy2(__file__, evidence / "finish_collection.py")
    write(RESULTS / "collection_audit.json", report)
    inventory = sorted(p for p in RESULTS.rglob("*") if p.is_file() and p.name != "file_manifest.json") + figure_paths
    write(evidence / "file_manifest.json", {
        p.relative_to(PAPER).as_posix(): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in inventory
    })
    print(json.dumps({k: report[k] for k in ("status", "sample_summary_checks", "enumeration_solver_calls", "strict_batch_outputs", "unchanged_paper_files")}))


if __name__ == "__main__":
    action = sys.argv[1]
    {"promote": promote, "figures": figures, "audit": audit}[action]()
