"""Check the archived data, manuscript tables, narrative values, and recorded binaries.

This uses only the Python standard library and does not rerun native solvers.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER / "results"
SOLVERS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "jvs", "jvc", "scipy")
MEMORY = ("jvrust", "jvhybrid", "jvcpp", "jvx", "scipy")
FORMULATIONS = ("jvrust", "jvcpp", "scipy_square", "scipy_dummies", "scipy_capped")
BATCH_CASES = ("batch64x64x80", "batch16x128x160_optional", "batch1024x8x10")
BATCH_ORDER = ("jvrust_loop", "jvrust_batch_t1", "jvrust_batch_t4", "jvhybrid_batch_t1",
               "jvhybrid_batch_t4", "jvcpp_batch_t1", "jvcpp_batch_t4", "jvx_batch_t1",
               "jvx_batch_t4", "scipy_loop")


def read_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measurement_script(name, expected_sha256):
    """Resolve the exact script used for a measurement, including preserved sources."""
    for path in (PAPER / name, PAPER / "archive" / "measurement-scripts" / name):
        if path.is_file() and digest(path) == expected_sha256:
            return path
    raise AssertionError(f"No preserved measurement script matches {name}: {expected_sha256}")


def numeric_rows(text):
    rows = []
    for line in text.splitlines():
        if line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            try:
                rows.append((cells[0], [float(value) for value in cells[1:]]))
            except ValueError:
                pass
    return rows


def section(manuscript, start, end):
    return manuscript.split(start, 1)[1].split(end, 1)[0]


def check_summary(samples, summaries, rounds):
    groups = defaultdict(list)
    for row in samples:
        groups[(row["case"], row["solver"])].append(float(row["nanoseconds"]))
    assert len(groups) == len(summaries)
    for row in summaries:
        values = sorted(groups[(row["case"], row["solver"])])
        assert len(values) == int(row["rounds"]) == rounds == 9
        for field, expected in (("median_ns", statistics.median(values)),
                                ("q25_ns", values[2]), ("q75_ns", values[6])):
            assert math.isclose(float(row[field]), expected, rel_tol=1e-12)
    return {key: statistics.median(values) for key, values in groups.items()}


def check_binaries(environment, runtime):
    assert set(environment["native_modules"]) == set(runtime["native_modules"])
    for name, metadata in environment["native_modules"].items():
        assert metadata["sha256"] == runtime["native_modules"][name]["sha256"], name


def check_batch_validation(batch_validation, environment, runtime):
    recheck_environment = batch_validation["environment"]
    assert batch_validation["status"] == "passed"
    assert recheck_environment["experiment_sha256"] == digest(PAPER / "run_experiments.py")
    assert recheck_environment["seed"] == environment["seed"]
    assert recheck_environment["rounds"] == 0
    assert recheck_environment["batch_affinity"] == environment["batch_affinity"]
    for key in ("python", "numpy", "scipy"):
        assert recheck_environment[key] == environment[key], key
    check_binaries(recheck_environment, runtime)
    assert {name: info["version"] for name, info in recheck_environment["packages"].items()} == \
        {name: info["version"] for name, info in runtime["packages"].items()}
    from run_experiments import batch_names, batch_specs
    specs = batch_specs()
    names = batch_names(len(recheck_environment["batch_affinity"]))
    assert len(batch_validation["checks"]) == len(specs)
    for row, spec in zip(batch_validation["checks"], specs, strict=True):
        assert row["case"] == spec["case"]
        assert row["problems"] == spec["b"] and row["solvers"] == len(names)
        assert row["checked_outputs"] == {name: spec["b"] for name in names}
        assert row["objective_checks_passed"]
        assert re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])
    assert batch_validation["checked_outputs"] == sum(
        sum(row["checked_outputs"].values()) for row in batch_validation["checks"]
    ) == 11040


def main():
    manuscript = (PAPER / "manuscript.md").read_text(encoding="utf-8")
    runtime = read_json("runtime.json")
    environment = read_json("environment.json")
    measured_script = measurement_script("run_experiments.py", environment["experiment_sha256"])
    assert digest(PAPER / "record_runtime.py") == runtime["record_script_sha256"]
    check_binaries(environment, runtime)
    assert {name: info["version"] for name, info in environment["packages"].items()} == \
        {name: info["version"] for name, info in runtime["packages"].items()}

    # Main study: samples, summaries, and the three numeric tables.
    single = check_summary(read_csv("single_samples.csv"), read_csv("single_summary.csv"), environment["rounds"])
    batch = check_summary(read_csv("batch_samples.csv"), read_csv("batch_summary.csv"), environment["rounds"])
    from run_experiments import case_specs
    cases = [spec["case"] for spec in case_specs()]
    single_table = numeric_rows(section(manuscript, "### 6.2 ", "### 6.3 "))
    assert len(single_table) == len(cases) == 12
    for (_, values), case in zip(single_table, cases):
        assert values == [float(f"{single[case, solver] / 1000:.2f}") for solver in SOLVERS], case
    batch_table = numeric_rows(section(manuscript, "### 6.3 ", "### 6.4 "))
    assert len(batch_table) == len(BATCH_ORDER)
    for (_, values), solver in zip(batch_table, BATCH_ORDER):
        assert values == [float(f"{batch[case, solver] / 1e6:.3f}") for case in BATCH_CASES], solver
    memory = {(row["case"], row["solver"]): row["additional_peak"] / 2**20 for row in read_json("memory.json")}
    memory_table = numeric_rows(section(manuscript, "### 6.4 ", "### 6.5 "))
    assert len(memory_table) == len(MEMORY)
    for (_, values), solver in zip(memory_table, MEMORY):
        assert values == [float(f"{memory[case, solver]:.3f}") for case in ("required", "optional")]

    validation = read_json("validation.json")
    assert not validation["failures"]
    assert validation["solver_calls"] == 16800 and validation["infeasible_calls"] == 80
    assert validation["maximum_absolute_objective_error"] == 0
    checks = read_json("benchmark_checks.json")
    assert all(row["objective_checks_passed"] for group in checks.values() for row in group)
    # Strict rechecks have separate provenance from both historical timing runs.
    batch_validation = read_json("batch_validation.json")
    check_batch_validation(batch_validation, environment, runtime)
    archived_root = PAPER / "archive" / "python-3.11.16"
    archived_batch_validation = json.loads(
        (archived_root / "batch-revalidation" / "batch_validation.json").read_text(encoding="utf-8")
    )
    archived_validation_environment = json.loads(
        (archived_root / "paper-results" / "environment.json").read_text(encoding="utf-8")
    )
    archived_validation_runtime = json.loads(
        (archived_root / "paper-results" / "runtime.json").read_text(encoding="utf-8")
    )
    measurement_script("run_experiments.py", archived_validation_environment["experiment_sha256"])
    check_batch_validation(archived_batch_validation, archived_validation_environment,
                           archived_validation_runtime)
    tests = read_json("test_validation.json")
    assert all(suite["exit_code"] == 0 for suite in tests["suites"].values())
    test_counts = {name: sum(suite["reported_pass_counts"]) for name, suite in tests["suites"].items()}

    # Formulation comparison.
    formulation_environment = read_json("formulations/environment.json")
    assert digest(PAPER / "compare_optional.py") == formulation_environment["supplement_sha256"]
    assert formulation_environment["experiment_sha256"] == environment["experiment_sha256"]
    check_binaries(formulation_environment, runtime)
    formulation_samples = read_csv("formulations/samples.csv")
    formulations = check_summary(formulation_samples, read_csv("formulations/summary.csv"),
                                 formulation_environment["rounds"])
    datasets = read_json("formulations/datasets.json")
    formulation_checks = read_json("formulations/checks.json")
    assert len(datasets) == 10 and len(formulation_samples) == 450 and len(formulation_checks) == 200
    assert all(math.isclose(row["objective"], row["reference_objective"], rel_tol=1e-10, abs_tol=1e-10)
               for row in formulation_checks)
    formulation_table = numeric_rows(section(manuscript, "### 6.5 ", "### 6.6 "))
    assert len(formulation_table) == len(datasets)
    for (_, values), case in zip(formulation_table, datasets):
        counts = [row["pairs"] for row in formulation_checks
                  if row["case"] == case["case"] and row["solver"] == "jvrust"]
        assert len(counts) == case["instances"] == 4 and statistics.mean(counts) == case["mean_pairs"]
        expected = [min(case["n"], case["m"]) - case["mean_pairs"]] + [
            formulations[case["case"], method] / 1000 for method in FORMULATIONS]
        assert values == [float(f"{value:.2f}") for value in expected], case["case"]
    formulation_validation = read_json("formulations/validation.json")
    assert formulation_validation["status"] == "passed"
    assert formulation_validation["grid_solver_calls"] == 10500
    assert formulation_validation["targeted_solver_calls"] == 20
    assert formulation_validation["maximum_absolute_grid_objective_error"] == 0

    # Scaling experiment.
    scaling_environment = read_json("scaling/environment.json")
    assert scaling_environment["status"] == "completed"
    assert digest(PAPER / "benchmark_scaling.py") == scaling_environment["scaling_script_sha256"]
    assert digest(PAPER / "compare_optional.py") == scaling_environment["optional_script_sha256"]
    assert scaling_environment["experiment_sha256"] == environment["experiment_sha256"]
    check_binaries(scaling_environment, runtime)
    scaling_datasets = read_json("scaling/datasets.json")
    scaling_checks = read_json("scaling/checks.json")
    scaling_samples = read_csv("scaling/samples.csv")
    scaling = check_summary(scaling_samples, read_csv("scaling/summary.csv"), scaling_environment["rounds"])
    from benchmark_scaling import specifications
    expected_specs = list(specifications())
    assert len(scaling_datasets) == len(expected_specs) == scaling_environment["completed_cases"] == 47
    assert len(scaling_checks) == scaling_environment["objective_checks"]
    assert len(scaling_samples) == scaling_environment["timing_samples"]
    assert all(math.isclose(row["objective"], row["reference_objective"], rel_tol=1e-10, abs_tol=1e-10)
               for row in scaling_checks)
    for case, spec in zip(scaling_datasets, expected_specs):
        for key, value in spec.items():
            assert case[key] == (None if key == "limit" and not math.isfinite(value) else value)
        assert case["instances"] == len(case["input_sha256"]) == 4
        methods = scaling_environment[case["regime"] + "_methods"]
        case_checks = [row for row in scaling_checks if row["case"] == case["case"]]
        assert {(row["solver"], row["instance"]) for row in case_checks} == {
            (method, index) for method in methods for index in range(4)}
        counts = [row["pairs"] for row in case_checks if row["solver"] == "jvrust"]
        assert statistics.mean(counts) == case["mean_pairs"]
    for stem in ("single_runtime", "batch_runtime", "memory_peak",
                 "scaling_required", "scaling_optional", "scaling_blocks", "scaling_blocks_runs"):
        for folder, suffix in (("figures", "png"), ("figures", "svg"), ("latex/images", "pdf")):
            assert (PAPER / folder / f"{stem}.{suffix}").is_file(), (stem, suffix)

    # Sequential run manifest.
    run = read_json("rerun/rerun.json")
    assert run["status"] == "completed"
    assert [entry["stage"] for entry in run["suites"]] == ["main", "optional", "scaling"]
    for entry, script, run_environment in zip(run["suites"],
            ("run_experiments.py", "compare_optional.py", "benchmark_scaling.py"),
            (environment, formulation_environment, scaling_environment)):
        assert entry["exit_code"] == 0
        measurement_script(script, entry["script_sha256"])
        if entry["stage"] == "main":
            assert entry["script_sha256"] == environment["experiment_sha256"]
        assert entry["started_utc"] <= run_environment["recorded_utc"] <= entry["completed_utc"]
    cpu_load = read_csv("rerun/cpu_load.csv")
    assert {row["stage"] for row in cpu_load} >= {"main", "optional", "scaling"}

    # Narrative values must match the archived measurements.
    from narrative import expected_sentences
    sentences = expected_sentences(single, batch, formulations, scaling, datasets, scaling_datasets, test_counts)
    for heading, sentence in sentences:
        assert manuscript.count(sentence) == 1, (heading, sentence)
        # Catch stale text left behind an updated sentence, such as a trailing digit or comma.
        following = manuscript[manuscript.index(sentence) + len(sentence):][:1]
        assert not following.isdigit(), (heading, sentence, following)
        assert not (sentence.endswith(".") and following and following not in " \n"), (heading, sentence, following)

    # Appendix A: tables and sentences from the archived Python 3.11.16 run.
    from narrative import ARCHIVE, appendix_sentences
    from update_manuscript_tables import all_table_values
    archived_environment = json.loads((ARCHIVE / "environment.json").read_text(encoding="utf-8"))
    assert archived_environment["python"].startswith("3.11.16")
    for name in ("single", "batch", "formulations/", "scaling/"):
        stem = name if name.endswith("/") else name + "_"
        with (ARCHIVE / (stem + "samples.csv")).open(newline="") as stream:
            archived_samples = list(csv.DictReader(stream))
        with (ARCHIVE / (stem + "summary.csv")).open(newline="") as stream:
            check_summary(archived_samples, list(csv.DictReader(stream)), 9)
    expected = all_table_values()
    appendix_rows = 0
    for heading, following in (("A.1", "### A.2 "), ("A.2", "### A.3 "), ("A.3", "### A.4 "), ("A.4", None)):
        text = manuscript.split(f"### {heading} ", 1)[1]
        rows = numeric_rows(text.split(following, 1)[0] if following else text)
        values, places = expected[heading]
        assert [row[1] for row in rows] == [[float(f"{v:.{places}f}") for v in row] for row in values], heading
        appendix_rows += len(rows)
    appendix = appendix_sentences()
    for heading, sentence in appendix:
        assert sentence in manuscript, (heading, sentence)
    for stem in ("manuscript.md", "README.md", "figures/single_runtime.png"):
        assert (ARCHIVE.parent / "paper-snapshot" / stem).is_file(), stem

    # Manuscript hygiene.
    for target in re.findall(r"\]\(([^)]+)\)", manuscript):
        if not target.startswith(("http://", "https://", "#")):
            assert (PAPER / target).is_file(), target
    assert manuscript.count("$$") % 2 == 0
    assert not re.search(r"\b(?:TODO|TBD|FIXME)\b|\{\{", manuscript)
    assert not re.search(r"\b[0-9a-f]{40}\b", manuscript), "commit hashes are not cited"
    # The paper is shared alone: no artifact paths or files outside embedded figures.
    prose = re.sub(r"!\[[^\]]*\]\(figures/[^)]+\)", "", manuscript)
    assert not re.search(r"\b(?:archive|results|tmp|latex|output|figures)/|README|[A-Za-z]:\\", prose)
    assert len(re.findall(r"^@", (PAPER / "references.bib").read_text(), re.M)) == 19
    for path in PAPER.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    report = {
        "status": "passed",
        "scope": "Artifact consistency; native computations are not rerun by this check.",
        "packages": {name: info["version"] for name, info in runtime["packages"].items()},
        "timing_samples": {"single": len(read_csv("single_samples.csv")), "batch": len(read_csv("batch_samples.csv")),
                           "formulations": len(formulation_samples), "scaling": len(scaling_samples)},
        "checked_table_rows": len(single_table) + len(batch_table) + len(memory_table) + len(formulation_table),
        "checked_narrative_sentences": len(sentences),
        "checked_appendix_table_rows": appendix_rows,
        "checked_appendix_sentences": len(appendix),
        "batch_validation_outputs": batch_validation["checked_outputs"],
        "archived_batch_validation_outputs": archived_batch_validation["checked_outputs"],
        "measurement_experiment_script": str(measured_script.relative_to(PAPER)).replace("\\", "/"),
        "measurement_experiment_sha256": environment["experiment_sha256"],
        "enumeration_solver_calls": validation["solver_calls"] + formulation_validation["grid_solver_calls"]
                                    + formulation_validation["targeted_solver_calls"],
        "scaling_settings": len(scaling_datasets), "scaling_objective_checks": len(scaling_checks),
        "cpu_utilization_observations": len(cpu_load),
        "passed_tests": test_counts,
        "artifact_script_hashes": {path.name: digest(path) for path in sorted(PAPER.glob("*.py"))},
        "manuscript_sha256": digest(PAPER / "manuscript.md"),
    }
    (RESULTS / "artifact_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
