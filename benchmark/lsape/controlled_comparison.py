"""Compare original and specialized LSAPE in one freshly measured session.

This script never builds or installs software and imports no other assignment
package. SciPy is an untimed objective reference. Importing this module performs
no solver calls, measurements, environment changes, or output writes.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as dt
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
import traceback
import zipfile

from redaction import redact


ROOT = Path(__file__).resolve().parent
SEED = 20261027
INPUTS_PER_CASE = 4
METHOD_APIS = {
    "original": "solve",
    "one_sided_materialized": "solve_one_sided",
    "implicit_scan": "solve_implicit_scan",
    "specialized_cached": "solve_specialized",
}
PRIMARY_METHODS = ("original", "specialized_cached")
np = None


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def matrix_digest(cost):
    return hashlib.sha256(cost.tobytes(order="C")).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(redact(value), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def effective_penalty(limit):
    """Match the bridge's endpoint rounding, including subnormal limits."""
    half = float(limit) / 2.0
    return half, half + half


def load_runtime():
    global np
    previous = {}
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        previous[name] = os.environ.get(name)
        os.environ[name] = "1"
    build = ROOT / "build"
    sys.path.insert(0, str(build))
    np = importlib.import_module("numpy")
    scipy = importlib.import_module("scipy")
    lsa = importlib.import_module("scipy.optimize").linear_sum_assignment
    bridge = importlib.import_module("_lsape_native")
    binary = Path(bridge.__file__).resolve()
    if binary.parent != build.resolve():
        raise RuntimeError(f"Bridge was loaded outside this experiment's build directory: {binary}")
    methods = {}
    for name, attribute in METHOD_APIS.items():
        function = getattr(bridge, attribute)
        if not callable(function):
            raise TypeError(f"Required bridge API is not callable: {attribute}")
        methods[name] = function
    return bridge, scipy, lsa, methods, previous


def pin_process():
    """Request CPU 0 only for this process, without changing system settings."""
    if os.name != "nt":
        return {"applied": False, "reason": "CPU affinity request is Windows-specific"}
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)
    ]
    kernel.GetProcessAffinityMask.restype = ctypes.c_int
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = ctypes.c_int
    handle = kernel.GetCurrentProcess()
    original, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(original), ctypes.byref(system)):
        return {"requested_logical_cpu": 0, "applied": False,
                "error": ctypes.get_last_error()}
    applied = bool(kernel.SetProcessAffinityMask(handle, 1))
    return {"requested_logical_cpu": 0, "applied": applied,
            "original_process_mask": original.value, "system_mask": system.value,
            "error": None if applied else ctypes.get_last_error()}


def check_upstream_archive():
    """Compare every extracted upstream file with its original ZIP member."""
    archive = ROOT / "src" / "lsape-v4.zip"
    extracted = (ROOT / "src" / "upstream-v4").resolve()
    records, failures, seen = [], [], set()
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename.replace("\\", "/"))
            target = (extracted / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(extracted):
                failures.append({"member": entry.filename, "reason": "Unsafe member path"})
                continue
            key = relative.as_posix()
            if key in seen:
                failures.append({"member": key, "reason": "Duplicate member path"})
                continue
            seen.add(key)
            content = source.read(entry)
            expected = hashlib.sha256(content).hexdigest()
            actual = digest(target) if target.is_file() else None
            equal = actual == expected
            records.append({"member": key, "bytes": len(content),
                            "archive_sha256": expected, "extracted_sha256": actual,
                            "identical": equal})
            if not equal:
                failures.append({"member": key, "reason": "Missing or changed extracted file"})
    extras = sorted(
        p.relative_to(extracted).as_posix()
        for p in extracted.rglob("*") if p.is_file()
        and p.relative_to(extracted).as_posix() not in seen
    )
    if extras:
        failures.append({"reason": "Files absent from original archive", "files": extras})
    return {"archive": str(archive), "archive_sha256": digest(archive),
            "extracted_root": str(extracted), "file_count": len(records),
            "all_members_identical": not failures, "failures": failures,
            "members": records}


def source_files():
    extensions = {".py", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".inl"}
    # Top-level experiment sources include the executed driver and bridge builder.
    files = {p.resolve() for p in ROOT.iterdir()
             if p.is_file() and p.suffix.lower() in extensions}
    for path in (ROOT / "src").rglob("*"):
        if path.is_file() and (
            path.suffix.lower() in extensions
            or path.name.lower().startswith(("license", "licence", "copying", "readme"))
        ):
            files.add(path.resolve())
    files.add((ROOT / "src" / "lsape-v4.zip").resolve())
    return sorted(files)


def resolve_build_input(name, candidates):
    relative = Path(name)
    direct = relative if relative.is_absolute() else ROOT / relative
    if direct.is_file():
        return direct.resolve()
    matches = [p for p in candidates if p.name == relative.name]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot uniquely resolve build provenance input {name!r}")
    return matches[0]


def snapshot_provenance(dest, bridge):
    build_record_path = ROOT / "build" / "provenance.json"
    build_record = json.loads(build_record_path.read_text(encoding="utf-8"))
    binary = Path(bridge.__file__).resolve()
    if build_record.get("returncode") != 0:
        raise RuntimeError("Build provenance does not describe a successful build")
    binary_hash = digest(binary)
    if binary_hash != build_record.get("output_sha256"):
        raise RuntimeError("Loaded bridge bytes differ from successful build provenance")
    candidates = source_files()
    declared = build_record.get("input_sha256", {})
    if not declared:
        raise RuntimeError("Build provenance has no source input hashes")
    checks = []
    for name, expected in declared.items():
        path = resolve_build_input(name, candidates)
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"Source changed since the recorded build: {name}")
        checks.append({"recorded_name": name, "path": str(path), "sha256": actual})
    required = [ROOT / "lsape_bridge.cpp", ROOT / "build_bridge.py",
                ROOT / "src" / "lsape-v4.zip",
                ROOT / "src" / "specialized" / "hungarian-lsape-uniform.hh",
                ROOT / "src" / "upstream-v4" / "lsape" / "include" / "hungarian-lsape.hh"]
    checked_paths = {Path(item["path"]).resolve() for item in checks}
    for path in required:
        if path.resolve() not in checked_paths:
            raise RuntimeError(f"Required build input lacks a matching recorded hash: {path}")
    files = set(candidates)
    files.update([binary, build_record_path.resolve()])
    for name in ("build.log", "compile_bridge.cmd"):
        path = ROOT / "build" / name
        if path.is_file():
            files.add(path.resolve())
    manifest = []
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        copied = dest / "provenance" / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
        original_hash = digest(path)
        if digest(copied) != original_hash:
            raise RuntimeError(f"Provenance copy differs from its source: {relative}")
        manifest.append({"path": relative.as_posix(), "sha256": original_hash,
                         "bytes": path.stat().st_size,
                         "snapshot": copied.relative_to(dest).as_posix()})
    write_json(dest / "provenance_manifest.json", manifest)
    return {"loaded_bridge": str(binary), "loaded_bridge_sha256": binary_hash,
            "build": build_record, "verified_build_inputs": checks,
            "snapshot_manifest": "provenance_manifest.json", "snapshot_files": len(manifest)}


def check_domain(cost, limit):
    if not isinstance(cost, np.ndarray) or cost.dtype != np.float64:
        raise TypeError("All comparison inputs must be float64 NumPy arrays")
    if cost.ndim != 2 or not cost.flags.c_contiguous:
        raise ValueError("All comparison inputs must be two-dimensional and C-contiguous")
    if not math.isfinite(limit) or limit < 0 or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("This comparison supports only finite nonnegative costs and limits")


def objective(cost, limit, pairs):
    n, m = cost.shape
    if not isinstance(pairs, np.ndarray) or pairs.dtype != np.int64:
        raise TypeError("Expected an int64 NumPy pair array")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"Expected Kx2 pair output, received {pairs.shape}")
    rows, cols = pairs[:, 0], pairs[:, 1]
    if len(pairs) > min(n, m):
        raise ValueError("Too many real pairs")
    if (np.any(rows < 0) or np.any(rows >= n)
            or np.any(cols < 0) or np.any(cols >= m)):
        raise ValueError("Pair index out of bounds")
    if len(np.unique(rows)) != len(rows) or len(np.unique(cols)) != len(cols):
        raise ValueError("Repeated row or column in pair output")
    selected = cost[rows, cols]
    if not np.isfinite(selected).all():
        raise ValueError("Selected a non-finite cost")
    half, _ = effective_penalty(limit)
    result = float(selected.sum(dtype=np.float64) + half * (n + m - 2 * len(pairs)))
    if not math.isfinite(result):
        raise ValueError("Non-finite reconstructed objective")
    return result


def reference_objective(cost, limit, lsa):
    _, effective = effective_penalty(limit)
    rows, cols = lsa(np.minimum(cost, effective))
    keep = cost[rows, cols] < effective
    pairs = np.column_stack((rows[keep], cols[keep])).astype(np.int64, copy=False)
    return objective(cost, limit, pairs)


def enumerate_objective(cost, limit):
    """Exhaustive partial-matchings oracle, guarded against large inputs."""
    n, m = cost.shape
    if n > 4 or m > 4:
        raise ValueError("Exhaustive enumeration is limited to dimensions at most 4")
    half, _ = effective_penalty(limit)
    best = math.inf

    def visit(i, used, selected, count):
        nonlocal best
        if i == n:
            best = min(best, selected + half * (n + m - 2 * count))
            return
        visit(i + 1, used, selected, count)
        for j in range(m):
            if not (used >> j) & 1:
                visit(i + 1, used | (1 << j), selected + float(cost[i, j]), count + 1)

    visit(0, 0, 0.0, 0)
    return float(best)


def require_equal(actual, expected, exact, context):
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise ValueError(f"{context}: non-finite value ({actual}, {expected})")
    equal = actual == expected if exact else bool(np.isclose(
        actual, expected, rtol=1e-10, atol=1e-10
    ))
    if not equal:
        raise ValueError(f"{context}: {actual!r} != {expected!r} (exact={exact})")


def validate(cost, limit, expected, methods, *, exact=False):
    check_domain(cost, limit)
    original_hash = matrix_digest(cost)
    outcomes = {}
    for name, solve in methods.items():
        # Exercise the exact no-dual entrypoint used for timing as well as its
        # diagnostic mode. Equality of pair identities is intentionally absent.
        pairs = solve(cost, limit)
        value = objective(cost, limit, pairs)
        require_equal(value, expected, exact, f"{name} objective")
        if matrix_digest(cost) != original_hash:
            raise ValueError(f"{name} changed its input in the normal call")
        dual_pairs, dual = solve(cost, limit, return_dual=True)
        diagnostic_value = objective(cost, limit, dual_pairs)
        dual = float(dual)
        require_equal(diagnostic_value, expected, exact, f"{name} diagnostic objective")
        require_equal(dual, expected, exact, f"{name} dual in endpoint objective F")
        if matrix_digest(cost) != original_hash:
            raise ValueError(f"{name} changed its input in the diagnostic call")
        outcomes[name] = {"objective": value, "pairs": len(pairs),
                          "diagnostic_objective": diagnostic_value,
                          "diagnostic_pairs": len(dual_pairs), "dual_F": dual,
                          "objective_absolute_error": abs(value - expected),
                          "dual_absolute_error": abs(dual - expected),
                          "exact_comparison": exact, "input_unchanged": True}
    return outcomes


def small_cases():
    rng = np.random.default_rng(SEED + 1)
    for n, m in ((1, 1), (2, 2), (2, 3), (3, 2), (3, 3)):
        for limit in (0.0, 0.5, 1.0):
            for repeat in range(3):
                cost = rng.integers(0, 9, size=(n, m)).astype(np.float64) / 8
                yield f"enumeration_{n}x{m}_L{limit}_{repeat}", cost, limit
    targets = [
        ("rejection_before_optimization", [[0.0, 0.75], [0.75, 2.0]], 1.0),
        ("single_row", [[0.75, 0.125, 0.5]], 0.5),
        ("single_column", [[0.75], [0.125], [0.5]], 0.5),
        ("all_rejected", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 0.25),
        ("all_equal_below_limit", [[0.125] * 3] * 2, 0.5),
        ("all_equal_at_limit", [[0.5] * 3] * 2, 0.5),
        ("all_zero_limit_zero", [[0.0] * 3] * 2, 0.0),
        ("equality_ties", [[0.25, 0.5, 1.0], [0.5, 0.5, 0.25]], 0.5),
        ("wide", [[0.0, 0.5, 0.75, 1.0], [0.25, 0.125, 0.5, 0.75]], 0.5),
        ("tall", [[0.0, 0.25], [0.5, 0.125], [0.75, 0.5], [1.0, 0.75]], 0.5),
    ]
    for name, values, limit in targets:
        yield name, np.array(values, dtype=np.float64, order="C"), limit
    for shape in ((0, 0), (0, 3), (3, 0)):
        for limit in (0.0, 0.5):
            yield f"empty_{shape[0]}x{shape[1]}_L{limit}", np.empty(shape, dtype=np.float64), limit
    tiny = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
    for multiplier in (1, 2, 3, 5, 7):
        limit = multiplier * tiny
        for name, values in (
            ("square", [[0.0, tiny], [tiny, 2 * tiny]]),
            ("wide", [[tiny, 2 * tiny, 4 * tiny]]),
            ("tall", [[tiny], [2 * tiny], [4 * tiny]]),
        ):
            yield f"subnormal_{name}_L{multiplier}ulp", np.array(values, dtype=np.float64), limit
    yield "subnormal_empty_tall", np.empty((3, 0), dtype=np.float64), 3 * tiny


def workload_cases():
    rng = np.random.default_rng(SEED)
    specs = [
        ("uniform128x160", 128, 160, 0.5, "uniform"),
        ("square128", 128, 128, 0.5, "uniform"),
        ("reject128x160", 128, 160, 0.005, "uniform"),
        ("wide64x512", 64, 512, 0.005, "uniform"),
        ("tall512x64", 512, 64, 0.005, "transpose"),
        ("blocks256x320", 256, 320, 0.5, "blocks"),
        ("blocks1024x1280", 1024, 1280, 0.5, "blocks"),
        ("ties128x160", 128, 160, 0.5, "ties"),
    ]
    wide = None
    for name, n, m, limit, kind in specs:
        if kind == "transpose":
            if wide is None:
                raise RuntimeError("The paired wide inputs have not been generated")
            inputs = [np.ascontiguousarray(cost.T) for cost in wide]
        else:
            inputs = [rng.random((n, m)) for _ in range(INPUTS_PER_CASE)]
            for cost in inputs:
                if kind == "blocks":
                    sr, sc = 3 * n // 4, m // 4
                    cost[:sr, sc:] += 3
                    cost[sr:, :sc] += 3
                elif kind == "ties":
                    cost[:] = np.floor(cost * 8) / 8
        if name == "wide64x512":
            wide = inputs
        half, effective = effective_penalty(limit)
        yield {"name": name, "shape": [n, m], "limit": limit, "kind": kind,
               "endpoint_penalty": half, "effective_row_penalty": effective,
               "paired_transpose_of": "wide64x512" if kind == "transpose" else None}, inputs


def time_cycles(solve, inputs, limit, cycles):
    enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter_ns()
        for _ in range(cycles):
            for cost in inputs:
                solve(cost, limit)
        return time.perf_counter_ns() - start
    finally:
        if enabled:
            gc.enable()


def calibrate(solve, inputs, limit, target_ns):
    for cost in inputs:
        solve(cost, limit)
    cycles = 1
    while True:
        elapsed = time_cycles(solve, inputs, limit, cycles)
        if elapsed >= target_ns or cycles >= 100000:
            return cycles, elapsed
        cycles = min(100000, max(cycles + 1,
            int(cycles * target_ns / max(elapsed, 1) * 1.1)))


def write_summary(dest, report):
    fields = ["case", "rows", "columns", "limit", "method", "rounds",
              "q1_us", "median_us", "q3_us", "original_over_method"]
    with (dest / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            for name, stats in case.get("methods", {}).items():
                if "median_us" not in stats:
                    continue
                writer.writerow({"case": case["name"], "rows": case["shape"][0],
                    "columns": case["shape"][1], "limit": case["limit"], "method": name,
                    "rounds": len(stats["round_average_us"]),
                    "q1_us": stats["q1_us"], "median_us": stats["median_us"],
                    "q3_us": stats["q3_us"],
                    "original_over_method": stats["original_over_method"]})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate small fixtures and pilot inputs without timing")
    parser.add_argument("--output", type=Path,
                        help="New output directory; existing directories are never overwritten")
    parser.add_argument("--rounds", type=int, default=9,
                        help="Shuffled timing rounds per case (default: 9)")
    parser.add_argument("--target-ms", type=float, default=30.0,
                        help="Calibration target per complete-input-cycle block (default: 30)")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if not math.isfinite(args.target_ms) or args.target_ms <= 0:
        parser.error("--target-ms must be finite and positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = (args.output or ROOT / "results" / ("controlled-" + stamp)).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    report_path = dest / "report.json"
    report = {
        "schema_version": 1, "status": "started", "started_utc": utc_now(),
        "stage": "initialization", "output_directory": str(dest),
        "validate_only": args.validate_only, "seed": SEED,
        "rounds": args.rounds, "target_block_ms": args.target_ms,
        "inputs_per_case": INPUTS_PER_CASE,
        "scope": "Original versus specialized LSAPE; one freshly measured process/session/environment",
        "primary_methods": list(PRIMARY_METHODS), "method_apis": METHOD_APIS,
        "controls": ["one_sided_materialized", "implicit_scan"],
        "control_interpretation": {
            "original_to_one_sided_materialized": "Formulation control using the upstream solver",
            "one_sided_materialized_to_implicit_scan": "Uniform/implicit specialization, including preprocessing of known-zero column minima, border storage, and possible scratch-allocation changes; not a pure storage ablation",
            "implicit_scan_to_specialized_cached": "Cached rejection-candidate control within the same specialized implementation",
        },
        "objective": "F = sum selected original costs + half*(N+M-2K), half=double(L/2), p=half+half",
        "input_contract": "Same finite nonnegative float64 C-contiguous inputs; no dimension reorientation",
        "timed_work": "Python call; shared adapter validation, column-major input conversion, representation construction, per-call solver workspaces, solving, output consistency checks, and new int64 pair output; no total-cost calculation",
        "untimed_work": "Input generation; exhaustive or capped-SciPy references; external feasibility, objective, and dual checks; hashes; warmup; calibration",
        "comparison_contract": "All four kernels are called through one bridge binary and build; both original and specialized paths copy inputs into column-major storage. Specialization is not a zero-copy interface comparison.",
        "ratio_direction": "original_over_method > 1 means that method has lower within-session median latency",
        "validation_policy": {"small_fixtures": "Exact objective and dual equality, including subnormal limits",
                              "pilot_inputs": {"rtol": 1e-10, "atol": 1e-10},
                              "pair_identity_required": False,
                              "input_hashes_checked": True},
        "limitations": [
            "Exploratory designed workloads, not an application-weighted sample",
            "One process/session/environment; no cross-platform or independent-session replication",
            "No comparison with other assignment packages or historical timing sessions",
            "No kernel-phase latency breakdown, memory measurement, or batching study",
            "No signed costs, forbidden edges, or required-matching workload",
            "IQRs describe round averages, not per-instance confidence intervals",
            "Original and specialized representations differ only by a small cost-matrix border in storage size; both copy real input costs",
        ],
        "small_validation": [], "cases": [],
    }
    write_json(report_path, report)
    try:
        report["stage"] = "loading_runtime"
        bridge, scipy, lsa, methods, previous_threads = load_runtime()
        report.update({"python": sys.version, "executable": sys.executable,
                       "platform": platform.platform(), "processor": platform.processor(),
                       "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                       "thread_environment": {"requested": 1, "previous": previous_threads},
                       "affinity": pin_process(), "driver_sha256": digest(__file__)})
        report["stage"] = "upstream_integrity"
        archive_check = check_upstream_archive()
        write_json(dest / "upstream_archive_check.json", archive_check)
        report["upstream_integrity"] = {
            "record": "upstream_archive_check.json", "file_count": archive_check["file_count"],
            "all_members_identical": archive_check["all_members_identical"],
            "archive_sha256": archive_check["archive_sha256"],
        }
        if not archive_check["all_members_identical"]:
            raise RuntimeError("Extracted upstream files differ from the supplied ZIP archive")
        report["stage"] = "build_provenance"
        report["provenance"] = snapshot_provenance(dest, bridge)
        write_json(report_path, report)
        report["stage"] = "small_validation"
        for name, cost, limit in small_cases():
            report["active_case"] = name
            check_domain(cost, limit)
            expected = enumerate_objective(cost, limit)
            reference = reference_objective(cost, limit, lsa)
            require_equal(reference, expected, True, f"{name} capped-SciPy reference")
            half, effective = effective_penalty(limit)
            record = {"name": name, "shape": list(cost.shape), "limit": limit,
                      "endpoint_penalty": half, "effective_row_penalty": effective,
                      "costs": cost.tolist(), "input_sha256": matrix_digest(cost),
                      "expected": expected,
                      "outcomes": validate(cost, limit, expected, methods, exact=True)}
            report["small_validation"].append(record)
        report["small_validation_cases"] = len(report["small_validation"])
        write_json(report_path, report)
        print(f"Small validation: {len(report['small_validation'])} cases passed", flush=True)
        schedule = random.Random(SEED + 2)
        for spec, inputs in workload_cases():
            report["active_case"] = spec["name"]
            report["stage"] = "pilot_validation"
            setting = {**spec, "input_sha256": [matrix_digest(c) for c in inputs],
                       "validation": [], "methods": {}, "round_order": [],
                       "status": "validating"}
            report["cases"].append(setting)
            write_json(report_path, report)
            print("Checking " + spec["name"], flush=True)
            for index, cost in enumerate(inputs):
                check_domain(cost, spec["limit"])
                expected = reference_objective(cost, spec["limit"], lsa)
                setting["validation"].append({"input_index": index, "expected": expected,
                    "outcomes": validate(cost, spec["limit"], expected, methods)})
            setting["status"] = "validated"
            write_json(report_path, report)
            if args.validate_only:
                continue
            report["stage"] = "calibration"
            for name, solve in methods.items():
                cycles, elapsed = calibrate(solve, inputs, spec["limit"], args.target_ms * 1e6)
                setting["methods"][name] = {"cycles_per_round": cycles,
                    "calibration_elapsed_ns": elapsed,
                    "round_average_us": [], "round_elapsed_ns": []}
            report["stage"] = "timing"
            setting["status"] = "timing"
            write_json(report_path, report)
            for round_index in range(args.rounds):
                report["active_round"] = round_index
                order = list(methods)
                schedule.shuffle(order)
                setting["round_order"].append(order)
                for name in order:
                    stats = setting["methods"][name]
                    elapsed = time_cycles(methods[name], inputs, spec["limit"],
                                          stats["cycles_per_round"])
                    stats["round_elapsed_ns"].append(elapsed)
                    stats["round_average_us"].append(
                        elapsed / (1000 * stats["cycles_per_round"] * len(inputs)))
                if [matrix_digest(c) for c in inputs] != setting["input_sha256"]:
                    raise ValueError("A timed call changed an input matrix")
                write_json(report_path, report)
            for stats in setting["methods"].values():
                q1, median, q3 = np.quantile(stats["round_average_us"], [0.25, 0.5, 0.75])
                if median <= 0:
                    raise ValueError("Non-positive timing median")
                stats.update(q1_us=float(q1), median_us=float(median), q3_us=float(q3))
            original = setting["methods"]["original"]["median_us"]
            for stats in setting["methods"].values():
                stats["original_over_method"] = original / stats["median_us"]
            specialized = setting["methods"]["specialized_cached"]["median_us"]
            setting["primary_original_over_specialized"] = original / specialized
            setting["input_unchanged_after_timing"] = True
            setting["status"] = "complete"
            write_json(report_path, report)
            write_summary(dest, report)
            print(f"{spec['name']}: original {original:.3f} us; specialized {specialized:.3f} us; "
                  f"original/specialized {original / specialized:.3f}", flush=True)
        report.pop("active_case", None)
        report.pop("active_round", None)
        report["stage"] = "complete"
        report["status"] = "validated_only" if args.validate_only else "complete"
        report["completed_utc"] = utc_now()
        write_json(report_path, report)
        write_summary(dest, report)
        (ROOT / "results").mkdir(exist_ok=True)
        try:
            pointer = os.path.relpath(report_path, ROOT)
        except ValueError:  # output on another drive; write_json redacts the absolute path
            pointer = str(report_path)
        write_json(ROOT / "results" / "latest-controlled.json",
                   {"report": pointer, "status": report["status"], "completed_utc": report["completed_utc"]})
        print(redact("Saved " + str(report_path)), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["failed_utc"] = utc_now()
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
        write_json(report_path, report)
        print(redact(f"Comparison failed during {report['stage']}: {error}\nReport: {report_path}"),
              file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
