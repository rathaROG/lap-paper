"""Audit both frozen collections and the revised arXiv publication.

Shared validation helpers remain available to the historical collection scripts.
No solver measurements are rerun.
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
    """Validate both completed collections and the current publication."""
    from paper_evidence import verify
    verify()


if __name__ == "__main__":
    main()
