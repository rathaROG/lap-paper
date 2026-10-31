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

from .redaction import redact


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


def load_runtime(build_dir):
    global np
    previous = {}
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        previous[name] = os.environ.get(name)
        os.environ[name] = "1"
    build = Path(build_dir).resolve()
    record = json.loads((build / "provenance.json").read_text(encoding="utf-8"))
    if record.get("python_version", "").split()[0] != platform.python_version():
        raise RuntimeError("LSAPE build requires the same Python patch version")
    if record.get("output_sha256") != digest(build / "_lsape_native.pyd"):
        raise RuntimeError("LSAPE binary differs from build provenance")
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
    build_dir = Path(bridge.__file__).resolve().parent
    build_record_path = build_dir / "provenance.json"
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
    from .build_provenance import verified_builder
    builder, builder_evidence = verified_builder(build_dir, ROOT, build_record)
    checks = []
    for name, expected in declared.items():
        path = builder if name == "build_bridge.py" else resolve_build_input(name, candidates)
        actual = digest(path)
        if actual != expected:
            raise RuntimeError(f"Source changed since the recorded build: {name}")
        checks.append({"recorded_name": name, "path": str(path), "sha256": actual})
    required = [ROOT / "lsape_bridge.cpp", builder,
                ROOT / "src" / "lsape-v4.zip",
                ROOT / "src" / "specialized" / "hungarian-lsape-uniform.hh",
                ROOT / "src" / "upstream-v4" / "lsape" / "include" / "hungarian-lsape.hh"]
    checked_paths = {Path(item["path"]).resolve() for item in checks}
    for path in required:
        if path.resolve() not in checked_paths:
            raise RuntimeError(f"Required build input lacks a matching recorded hash: {path}")
    files = set(candidates)
    files.update([binary, build_record_path.resolve(), builder, *builder_evidence])
    for name in ("build.log", "compile_bridge.cmd"):
        path = build_dir / name
        if path.is_file():
            files.add(path.resolve())
    manifest = []
    for path in sorted(files):
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("build") / path.relative_to(build_dir)
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

