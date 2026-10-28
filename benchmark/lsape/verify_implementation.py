"""Targeted correctness/coverage verification, never a timing benchmark.

Uses the existing compiled extension, NumPy and SciPy; installs nothing.
Checks final-state identity, full primal/dual certificates, two SciPy
formulations, and independent exhaustive enumeration of small matchings.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy

from redaction import redact

VARIANTS = ("original", "one_sided", "implicit_scan", "specialized", "specialized_padded")
FUNCTIONS = ("solve", "solve_one_sided", "solve_implicit_scan", "solve_specialized", "solve_specialized_padded")
SEED = 20261027


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bits(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    if out.dtype.kind == "f":
        out[out == 0] = 0.0
        return out.view(np.uint64)
    return out


def objective(C: np.ndarray, half: float, pairs: np.ndarray) -> float:
    return math.fsum(float(C[i, j]) for i, j in pairs) + half * (sum(C.shape) - 2 * len(pairs))


def certificate(C: np.ndarray, L: float, state: dict, variant: str) -> dict:
    n, m = C.shape
    h = L / 2.0
    p = 2.0 * h
    rho, varrho, u, v = (state[k] for k in ("rho", "varrho", "u", "v"))
    pairs = state["pairs"]
    assert rho.shape == (n,) and varrho.shape == (m,)
    assert u.shape == (n + 1,) and v.shape == (m + 1,)
    assert pairs.dtype == np.int64 and pairs.ndim == 2 and pairs.shape[1] == 2
    assert np.all((rho >= 0) & (rho <= m)) and np.all((varrho >= 0) & (varrho <= n))
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert u[n] == 0 and v[m] == 0
    rows = np.flatnonzero(rho < m)
    expected = np.column_stack((rows, rho[rows])).astype(np.int64)
    assert np.array_equal(pairs, expected)
    assert len(np.unique(pairs[:, 1])) == len(pairs)
    for i, j in pairs:
        assert varrho[j] == i
    for j, i in enumerate(varrho):
        if i < n:
            assert rho[i] == j
    deletion, insertion = (h, h) if variant == "original" else (p, 0.0)
    # Scale-aware tolerance: no large absolute floor hides errors at tiny costs.
    # Tiny subnormal fixtures intentionally receive zero tolerance when this
    # product underflows: their integer-subnormal arithmetic is exact.
    scale = max(float(np.max(C, initial=0)), h, p, 0.0)
    tolerance = 512 * np.finfo(float).eps * max(n + m, 1) * scale
    feasibility = max(float(np.max(u[:n, None] + v[None, :m] - C, initial=0)),
                      float(np.max(u[:n] - deletion, initial=0)),
                      float(np.max(v[:m] - insertion, initial=0)))
    tightness = max((abs(float(C[i, j] - (u[i] + v[j]))) for i, j in pairs), default=0.0)
    rejected = float(np.max(np.abs(u[:n][rho == m] - deletion), initial=0))
    unused = float(np.max(np.abs(v[:m][varrho == n] - insertion), initial=0))
    assert max(feasibility, tightness, rejected, unused) <= tolerance, (
        variant, feasibility, tightness, rejected, unused, tolerance)
    primal = objective(C, h, pairs)
    dual = math.fsum(map(float, u[:n])) + math.fsum(map(float, v[:m]))
    if variant != "original":
        dual += h * (m - n)
    assert abs(primal - dual) <= tolerance * max(n + m, 1), (variant, primal, dual)
    assert abs(primal - state["dual_F"]) <= tolerance * max(n + m, 1)
    return {"primal": primal, "tolerance": tolerance * max(n + m, 1),
            "max_certificate_residual": max(feasibility, tightness, rejected, unused)}


def scipy_objectives(C: np.ndarray, L: float) -> tuple[float, float]:
    n, m = C.shape
    h, p = L / 2.0, 2.0 * (L / 2.0)
    oriented = C if n <= m else C.T
    ri, ci = linear_sum_assignment(np.minimum(oriented, p))
    keep = oriented[ri, ci] < p
    capped = math.fsum(map(float, oriented[ri[keep], ci[keep]])) + h * (n + m - 2 * int(keep.sum()))
    # Independent symmetric endpoint construction, no one-sided cost shift.
    square = np.full((n + m, n + m), h, dtype=np.float64)
    square[:n, :m] = C
    square[n:, m:] = 0
    si, sj = linear_sum_assignment(square)
    expanded = math.fsum(map(float, square[si, sj]))
    return capped, expanded


def exhaustive_objective(C: np.ndarray, L: float) -> tuple[float, int]:
    """Visit every partial-injection state; reject or match each row independently."""
    n, m = C.shape
    h = L / 2.0

    @lru_cache(None)
    def visit(i: int, used: int) -> float:
        if i == n:
            return h * (m - used.bit_count())
        best = h + visit(i + 1, used)
        for j in range(m):
            if not used & (1 << j):
                best = min(best, float(C[i, j]) + visit(i + 1, used | (1 << j)))
        return best

    return visit(0, 0), visit.cache_info().currsize


def fixtures():
    for n, m in ((0, 0), (0, 3), (4, 0), (1, 0), (0, 1)):
        yield f"empty_{n}_{m}", np.empty((n, m)), 0.1
    for L in (0.0, 0.1, 1.0 / 3.0, 1.0):
        p = 2 * (L / 2)
        yield f"zero_{L}", np.zeros((4, 5)), L
        yield f"negative_zero_{L}", np.full((3, 4), -0.0), L
        yield f"entry_rejection_{L}", np.array([[p], [0.8 * p]]), L
        yield f"dual_rejection_fallback_{L}", np.array([[0, 2*p], [0, 2*p]]), L
        yield f"dual_rejection_fast_{L}", np.array([[0.75*p, 2*p], [0.75*p, 2*p]]), L
        yield f"free_column_{L}", np.array([[0, 0.2*p], [0, 0.3*p]]), L
        for direction in (0.0, np.inf):
            h = p / 2
            q = np.nextafter(h, direction)
            yield f"half_boundary_{L}_{direction}", np.array([[h, 2*p], [q, 2*p]]), L
    tiny = np.nextafter(0.0, 1.0)
    for k in (1, 2, 3, 4, 8):
        yield f"subnormal_{k}", np.array([[0, tiny*k], [tiny, tiny*(k+1)]]), tiny*k
    bound = np.finfo(float).max / (16 * (1 + 1 + 1))
    yield "large_guard_inside", np.array([[bound / 4]]), bound / 2


def stress_cases(count: int):
    rng = np.random.default_rng(SEED)
    limits = np.array([0.1, 1/3, 0.2, 0.7, 1.1, 1.0, 0.01])
    for index in range(count):
        n = int(rng.integers(1, 65))
        m = n if index % 3 == 0 else int(rng.integers(1, 65))
        if index % 3 == 1:
            n, m = min(n, m), max(n, m)
        elif index % 3 == 2:
            n, m = max(n, m), min(n, m)
        L = float(limits[index % len(limits)])
        p, h = 2*(L/2), L/2
        mode = index % 6
        if mode == 0:
            C = rng.random((n, m)) * (2*p)
        elif mode in (1, 2):
            center = p if mode == 1 else h
            C = center * (1 + rng.uniform(-1e-12, 1e-12, (n, m)))
        elif mode == 3:
            values = np.array([0, np.nextafter(h, 0), h, np.nextafter(h, np.inf),
                               np.nextafter(p, 0), p, np.nextafter(p, np.inf), 2*p])
            C = rng.choice(values, (n, m))
        elif mode == 4:
            C = np.full((n, m), 2*p)
            C[:, :max(1, m//4)] = rng.choice([0.25*p, h, 0.75*p], (n, max(1, m//4)))
        else:
            C = rng.integers(0, 9, (n, m)).astype(float) * (p/4)
            C[0, 0] = -0.0
        yield f"stress_{index:05d}_mode{mode}", C, L


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).resolve().parent / "build")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stress-count", type=int, default=10000)
    args = parser.parse_args()
    if args.stress_count < 10000:
        raise SystemExit("Audited verification requires at least 10,000 stress instances")
    root = Path(__file__).resolve().parent
    out = (args.output_dir or root / "runs" / ("verification-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.build_dir.resolve() / "_lsape_native.pyd"
    for path in (Path(__file__).resolve(), root / "lsape_bridge.cpp", root / "build_bridge.py",
                 root / "src/specialized/hungarian-lsape-uniform.hh",
                 root / "src/upstream-v4/lsape/include/hungarian-lsape.hh", root / "README.md"):
        if path.is_file():
            destination = out / "sources" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    shutil.copy2(binary, out / binary.name)
    shutil.copy2(args.build_dir / "provenance.json", out / "build-provenance.json")
    provenance = json.loads((args.build_dir / "provenance.json").read_text(encoding="utf-8"))
    inputs = {
        "lsape_bridge.cpp": root / "lsape_bridge.cpp",
        "build_bridge.py": root / "build_bridge.py",
        "lsape-v4.zip": root / "src/lsape-v4.zip",
        "hungarian-lsape.hh": root / "src/upstream-v4/lsape/include/hungarian-lsape.hh",
        "src/specialized/hungarian-lsape-uniform.hh": root / "src/specialized/hungarian-lsape-uniform.hh",
    }
    integrity = {key: {"expected": provenance["input_sha256"].get(key), "actual": file_hash(path)}
                 for key, path in inputs.items()}
    integrity["_lsape_native.pyd"] = {"expected": provenance.get("output_sha256"), "actual": file_hash(binary)}
    integrity_passed = all(item["expected"] == item["actual"] for item in integrity.values())
    (out / "provenance-check.json").write_text(json.dumps(redact({"passed": integrity_passed, "checks": integrity}), indent=2) + "\n", encoding="utf-8")
    if not integrity_passed:
        raise SystemExit(redact(f"Refusing stale/mismatched build; see {out / 'provenance-check.json'}"))
    spec = importlib.util.spec_from_file_location("_lsape_native", binary)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    counts = {v: Counter() for v in VARIANTS}
    unknown_counter_keys = {v: set() for v in VARIANTS}
    log_file = (out / "verification.log").open("w", encoding="utf-8")

    def log(message):
        message = redact(message)
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()
    summary = {"status": "running", "seed": SEED, "started_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
               "binary_sha256": file_hash(binary), "source_sha256": {}, "stress_requested": args.stress_count,
               "cases": 0, "stress_cases": 0, "enumerated_cases": 0, "enumerated_states": 0,
               "certificates": 0, "state_identity_comparisons": 0, "timed_api_identity_checks": 0,
               "scipy_capped_checks": 0, "scipy_square_checks": 0, "domain_rejections": 0,
               "max_certificate_residual": 0.0, "interpretation": "Finite sampled validation; not universal floating-point verification."}
    summary["provenance_verified_before_loading"] = True
    summary["subnormal_check_policy"] = "No normal-number tolerance floor; exact residual/objective equality when relative tolerance underflows to zero."
    for path in (out / "sources").rglob("*"):
        if path.is_file():
            summary["source_sha256"][str(path.relative_to(out / "sources"))] = file_hash(path)
    case_file = (out / "cases.jsonl").open("w", encoding="utf-8")
    current = None
    try:
        def check(name, C, L, exhaustive=False, category="fixture"):
            nonlocal current
            C = np.asarray(C, dtype=np.float64, order="C")
            current = (name, C, L)
            states = {v: native.diagnose(C, L, variant=v) for v in VARIANTS}
            answers = {}
            for v, function_name in zip(VARIANTS, FUNCTIONS):
                state = states[v]
                cert = certificate(C, L, state, v)
                answers[v] = cert["primal"]
                summary["max_certificate_residual"] = max(summary["max_certificate_residual"], cert["max_certificate_residual"])
                summary["certificates"] += 1
                ordinary = getattr(native, function_name)(C, L)
                assert np.array_equal(ordinary, state["pairs"]), (v, "diagnostic/timed mismatch")
                summary["timed_api_identity_checks"] += 1
                counters = state["counters"]
                counts[v]["cases"] += 1
                counts[v]["bridge_empty_cases"] += bool(counters["bridge_empty"])
                counts[v]["kernel_no_search_cases"] += bool(counters["no_search"] and not counters["bridge_empty"])
                counts[v]["kernel_search_cases"] += not counters["no_search"]
                counts[v]["cases_with_unavailable_internal_counters"] += any(value is None for value in counters.values())
                for key, value in counters.items():
                    if value is None:
                        unknown_counter_keys[v].add(key)
                    elif not isinstance(value, bool):
                        counts[v][key] += value
                if v not in ("original", "one_sided"):
                    assert counters["searches"] == sum(counters[k] for k in ("free_column", "rejection_row_entry", "rejection_dual_update"))
                    assert counters["dual_updates"] == sum(counters[k] for k in ("cache_fast_path", "cache_fallback", "scan_rejection"))
            for v in VARIANTS[2:]:
                for field in ("rho", "varrho", "u", "v"):
                    assert np.array_equal(canonical_bits(states["one_sided"][field]), canonical_bits(states[v][field])), (name, v, field, "state identity")
                    summary["state_identity_comparisons"] += 1
            assert states["one_sided"]["leading_dimension"] == C.shape[0] + 1
            assert states["specialized_padded"]["leading_dimension"] == C.shape[0] + 1
            capped, square = scipy_objectives(C, L)
            tolerance = max(certificate(C, L, states[v], v)["tolerance"] for v in VARIANTS)
            for value in answers.values():
                assert abs(value - capped) <= tolerance and abs(value - square) <= tolerance, (name, value, capped, square, tolerance)
            summary["scipy_capped_checks"] += 1
            summary["scipy_square_checks"] += 1
            if exhaustive:
                reference, visited = exhaustive_objective(C, L)
                assert all(abs(value - reference) <= tolerance for value in answers.values())
                summary["enumerated_cases"] += 1
                summary["enumerated_states"] += visited
            summary["cases"] += 1
            summary["stress_cases"] += category == "stress"
            case_file.write(json.dumps({"case": name, "category": category, "shape": list(C.shape), "L": L,
                "input_sha256": hashlib.sha256(C.tobytes()).hexdigest(), "objectives": answers,
                "counters": {v: states[v]["counters"] for v in VARIANTS}}) + "\n")
            summary["mxcsr_observed"] = states["original"]["mxcsr"]

        for name, C, L in fixtures():
            check(name, C, L, exhaustive=max(C.shape, default=0) <= 6)
        rng = np.random.default_rng(SEED + 1)
        for n in range(7):
            for m in range(7):
                for q, L in enumerate((0.1, 1/3, 1.0)):
                    C = rng.integers(0, 9, (n, m)).astype(float) * (L/4)
                    check(f"exhaustive_{n}_{m}_{q}", C, L, exhaustive=True, category="exhaustive")
        for name, C, L in stress_cases(args.stress_count):
            check(name, C, L, category="stress")
            if summary["stress_cases"] % 1000 == 0:
                case_file.flush()
                log(f"Verified {summary['stress_cases']} stress matrices")
        # Rejection and arithmetic-environment contracts are shared by all arms.
        bound = np.finfo(float).max / 48
        invalid = [(np.array([[x]]), 1.0) for x in (-1.0, np.inf, np.nan, bound*2)]
        invalid += [(np.ones((1, 1)), x) for x in (-1.0, np.inf, np.nan, bound*2)]
        invalid += [(np.array([[1+2j]]), 1.0), (np.array([["1"]]), 1.0)]
        for C, L in invalid:
            for function_name in FUNCTIONS:
                try:
                    getattr(native, function_name)(C, L)
                except (ValueError, TypeError, OverflowError):
                    summary["domain_rejections"] += 1
                else:
                    raise AssertionError((function_name, "invalid input was accepted", C, L))
        for v in ("specialized", "specialized_padded"):
            for key in ("free_column", "rejection_row_entry", "rejection_dual_update", "cache_fast_path", "cache_fallback"):
                assert counts[v][key] > 0, (v, "uncovered branch", key)
        assert counts["implicit_scan"]["scan_rejection"] > 0
        # Explicit boundary arithmetic demonstrates why <=p/2 uses the scan.
        assert 1.0 - 0.5 == 1.0 - np.nextafter(0.5, 0.0)
        summary["half_boundary_rounding_demonstrated"] = True
        summary["status"] = "passed"
    except BaseException as error:
        summary["status"] = "failed"
        summary["error"] = repr(error)
        summary["traceback"] = traceback.format_exc()
        if current is not None:
            name, C, L = current
            np.save(out / "failure-costs.npy", C)
            summary["failure_case"] = {"name": name, "L": L, "shape": list(C.shape)}
        log(summary["traceback"])
    finally:
        case_file.close()
        summary["coverage"] = {v: dict(c) for v, c in counts.items()}
        for variant, keys in unknown_counter_keys.items():
            for key in keys:
                summary["coverage"][variant][key] = None
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (out / "report.json").write_text(json.dumps(redact(summary), indent=2) + "\n", encoding="utf-8")
        log(f"Verification {summary['status']}: {out / 'report.json'}")
        log_file.close()
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
