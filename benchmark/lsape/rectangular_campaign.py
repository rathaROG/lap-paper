"""Predeclared rectangular LSAPE campaign, one fresh process per session.

Write the immutable grid first with --write-manifest. A measurement invocation
requires that manifest and a session index. Importing this module runs no solver,
measurement, installation, compilation, or file write.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

import controlled_comparison as common
from redaction import redact

ROOT = Path(__file__).resolve().parent
GRID_ID = "lsape-rectangular-v1-67"
METHODS = {
    "original": ("solve", "original", "LSAPE-v4"),
    "one_sided_materialized": ("solve_one_sided", "one_sided", "LSAPE-v4 [p,0]"),
    "implicit_scan": ("solve_implicit_scan", "implicit_scan", "LSAPE-v4-1S/scan"),
    "specialized_cached": ("solve_specialized", "specialized", "LSAPE-v4-1S/tracked"),
    "specialized_padded": ("solve_specialized_padded", "specialized_padded", "LSAPE-v4-1S/tracked/padded"),
}
COMPARISONS = {
    "original_over_padded": ("original", "specialized_padded"),
    "original_over_one_sided": ("original", "one_sided_materialized"),
    "one_sided_over_padded": ("one_sided_materialized", "specialized_padded"),
    "scan_over_compact": ("implicit_scan", "specialized_cached"),
    "compact_over_padded": ("specialized_cached", "specialized_padded"),
    "padded_over_compact": ("specialized_padded", "specialized_cached"),
    "original_over_compact": ("original", "specialized_cached"),
    "original_over_scan": ("original", "implicit_scan"),
}
np = None


def canonical_hash(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def grid_manifest(seed):
    cases = []

    def add(block, family, n, m, regime, distribution="uniform", paired=True):
        if regime == "fixed":
            limit = 0.5
        elif regime == "size_dependent":
            limit = 2.0 / max(n, m)
        elif regime == "threshold_ties":
            limit = 2.0 * 2.0 ** (-math.ceil(math.log2(max(n, m))))
        else:
            raise ValueError(regime)
        group = f"{block}_{family}_{n}x{m}_{regime}"
        orientations = [(n, m, False)]
        if paired and n != m:
            orientations.append((m, n, True))
        for rows, cols, transpose in orientations:
            subset = (
                (block == "A" and rows == cols == 1024 and regime == "fixed")
                or (block == "B" and {rows, cols} == {512, 640} and regime == "fixed")
                or (block == "D" and {rows, cols} == {256, 1024} and regime == "fixed")
                or (block == "E" and {rows, cols} == {64, 16384})
                or (block == "C" and {rows, cols} == {1024, 1280})
                or (block == "F" and (rows, cols) == (256, 320))
            )
            cases.append({
                "case": f"{block}_{family}_{rows}x{cols}_{regime}",
                "block": block, "family": family, "shape": [rows, cols],
                "base_shape": [n, m], "transpose": transpose,
                "orientation": "square" if rows == cols else ("wide" if rows < cols else "tall"),
                "regime": regime, "limit": limit, "distribution": distribution,
                "input_group": group, "paper_subset": subset,
            })

    for n in (128, 256, 512, 1024):
        for regime in ("fixed", "size_dependent"):
            add("A", "square", n, n, regime)
    for n in (128, 256, 512, 1024):
        for regime in ("fixed", "size_dependent"):
            add("B", "moderate5", n, 5 * n // 4, regime)
    for n in (128, 256, 512, 1024):
        add("C", "congested5", n, 5 * n // 4, "fixed", "blocks")
    for n in (64, 128, 256, 512):
        for regime in ("fixed", "size_dependent"):
            add("D", "moderate4", n, 4 * n, regime)
    for m in (1024, 4096, 16384):
        for regime in ("fixed", "size_dependent"):
            add("E", "extreme", 64, m, regime)
    for n, m in ((256, 256), (256, 320), (256, 1024), (64, 4096)):
        add("F", "threshold_ties", n, m, "threshold_ties", "threshold_ties")
    if len(cases) != 67 or sum(c["paper_subset"] for c in cases) != 12:
        raise RuntimeError("The predeclared grid or paper subset changed")
    return {
        "schema_version": 1, "grid_id": GRID_ID, "base_seed": seed,
        "case_count": 67, "paper_subset_count": 12,
        "planned_sessions": [1, 2, 3], "inputs_per_case": 4,
        "rounds": 9, "target_block_ms": 30.0,
        "methods": {k: {"api": v[0], "diagnostic_variant": v[1], "label": v[2]}
                    for k, v in METHODS.items()},
        "primary_comparison": "original_over_padded",
        "comparisons": {k: {"numerator": v[0], "denominator": v[1]}
                        for k, v in COMPARISONS.items()},
        "practical_ratio_band": [0.97, 1.03],
        "scope": "All 67 cases retained; no optional limit-sweep block or post-measurement case selection",
        "generator": {
            "seed_derivation": "First 128 SHA256 bits of grid_id|base_seed|session_index|input_group|input_index",
            "uniform": "Independent U[0,1) binary64 costs",
            "blocks": "Generate in the wide orientation; first 3/4 of rows prefer first 1/4 of columns; add 3 in both off-diagonal blocks",
            "threshold_ties": "h=2^(-ceil(log2(max(shape)))); C=h*floor(U/h); L=2h",
            "pairing": "Each nonsquare case and its exact transpose share base values within session and regime",
            "limit_regimes": "Fixed and size-dependent regimes use separate generated values and different objectives",
            "sizes": "Fresh matrices for each shape; no nested inputs across sizes",
        },
        "sessions": "Three fresh processes within the current day/window; no designed or recorded reboot or independent-day replication",
        "cases": cases,
    }


def manifest_document(seed):
    payload = grid_manifest(seed)
    return {"manifest_sha256": canonical_hash(payload), "manifest": payload}


def read_manifest(path, requested_seed):
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["manifest"]
    if canonical_hash(payload) != document["manifest_sha256"]:
        raise ValueError("Frozen manifest digest does not match its contents")
    seed = payload["base_seed"] if requested_seed is None else requested_seed
    if document != manifest_document(seed):
        raise ValueError("Manifest differs from this predeclared grid, methods, or seed")
    return document


def inputs_for_case(spec, manifest, session):
    inputs, records = [], []
    n, m = spec["base_shape"]
    for index in range(manifest["inputs_per_case"]):
        key = f"{manifest['grid_id']}|{manifest['base_seed']}|{session}|{spec['input_group']}|{index}"
        seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:32], 16)
        rng = np.random.default_rng(seed)
        base = rng.random((n, m))
        if spec["distribution"] == "blocks":
            sr, sc = 3 * n // 4, m // 4
            base[:sr, sc:] += 3
            base[sr:, :sc] += 3
        elif spec["distribution"] == "threshold_ties":
            h = spec["limit"] / 2.0
            base[:] = h * np.floor(base / h)
        cost = np.ascontiguousarray(base.T) if spec["transpose"] else base
        inputs.append(cost)
        records.append({"input_index": index, "seed": seed,
                        "base_sha256": common.matrix_digest(base),
                        "input_sha256": common.matrix_digest(cost)})
    return inputs, records


def state_hash(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def canonical_state(diag):
    result = {}
    for name in ("rho", "varrho", "u", "v"):
        value = np.asarray(diag[name])
        if name in ("u", "v"):
            if value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} potential array")
            value = value.copy()
            value[value == 0] = 0.0  # Normalize the harmless sign of zero only.
            result[name] = value.view(np.uint64)
        else:
            if value.dtype != np.int64:
                raise TypeError(f"Expected int64 {name} mapping")
            result[name] = value
    return result


def require_same_state(reference, actual, method):
    for name in ("rho", "varrho", "u", "v"):
        a, b = reference[name], actual[name]
        if a.shape != b.shape:
            raise ValueError(f"{method} {name} state shape differs from one-sided upstream")
        unequal = np.flatnonzero(a != b)
        if len(unequal):
            i = int(unequal[0])
            raise ValueError(f"{method} differs from one-sided upstream in {name}[{i}]: "
                             f"{int(b.flat[i])} versus {int(a.flat[i])}; potentials reported as bits")


def certificate(cost, limit, diagnostic, method):
    n, m = cost.shape
    u = np.asarray(diagnostic["u"])
    v = np.asarray(diagnostic["v"])
    rho = np.asarray(diagnostic["rho"])
    varrho = np.asarray(diagnostic["varrho"])
    if u.shape != (n + 1,) or v.shape != (m + 1,) or rho.shape != (n,) or varrho.shape != (m,):
        raise ValueError("Diagnostic arrays have unexpected dimensions")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Non-finite diagnostic potentials")
    half, p = common.effective_penalty(limit)
    if diagnostic["half"] != half or diagnostic["penalty"] != p:
        raise ValueError("Diagnostic endpoint rounding differs from the declared objective")
    row_bound, col_bound = (half, half) if method == "original" else (p, 0.0)
    scale = max(1.0, float(np.max(np.abs(cost))), float(np.max(np.abs(u))),
                float(np.max(np.abs(v))), abs(p))
    tolerance = 1e-10 * (1.0 + scale)
    edge_violation = 0.0
    for first in range(0, n, 256):
        last = min(n, first + 256)
        violations = u[first:last, None] + v[None, :m] - cost[first:last]
        edge_violation = max(edge_violation, float(np.max(violations)))
    pairs = diagnostic["pairs"]
    rows, cols = pairs[:, 0], pairs[:, 1]
    matched_tightness = float(np.max(np.abs(cost[rows, cols] - u[rows] - v[cols]))) if len(pairs) else 0.0
    rejected = rho == m
    unused = varrho == n
    residuals = {
        "edge_feasibility": edge_violation,
        "row_bound": max(0.0, float(np.max(u[:n] - row_bound))),
        "column_bound": max(0.0, float(np.max(v[:m] - col_bound))),
        "matched_tightness": matched_tightness,
        "rejected_row_tightness": float(np.max(np.abs(u[:n][rejected] - row_bound))) if rejected.any() else 0.0,
        "unused_column_tightness": float(np.max(np.abs(v[:m][unused] - col_bound))) if unused.any() else 0.0,
        "null_potential": max(abs(float(u[n])), abs(float(v[m]))),
    }
    if any(value > tolerance for value in residuals.values()):
        raise ValueError(f"{method} primal-dual certificate failed: {residuals}, tolerance={tolerance}")
    correction = 0.0 if method == "original" else half * (m - n)
    reconstructed_dual = float(u[:n].sum(dtype=np.float64) + v[:m].sum(dtype=np.float64) + correction)
    common.require_equal(reconstructed_dual, float(diagnostic["dual_F"]), False,
                         f"{method} diagnostic potential sum")
    return {"passed": True, "tolerance": tolerance, "maximum_residual": max(residuals.values()),
            "residuals": residuals, "reconstructed_dual_F": reconstructed_dual}


def validate_input(cost, limit, expected, solves, diagnose):
    common.check_domain(cost, limit)
    original_hash = common.matrix_digest(cost)
    records, states = {}, {}
    n, m = cost.shape
    for method, solve in solves.items():
        normal_pairs = solve(cost, limit)
        value = common.objective(cost, limit, normal_pairs)
        common.require_equal(value, expected, False, f"{method} ordinary-call objective")
        if common.matrix_digest(cost) != original_hash:
            raise ValueError(f"{method} ordinary call mutated the input")
        diag = diagnose(cost, limit, variant=METHODS[method][1])
        diagnostic_value = common.objective(cost, limit, diag["pairs"])
        common.require_equal(diagnostic_value, expected, False, f"{method} diagnostic objective")
        common.require_equal(float(diag["dual_F"]), expected, False, f"{method} dual F")
        if not np.array_equal(normal_pairs, diag["pairs"]):
            raise ValueError(f"{method} timed and diagnostic entrypoints return different pairs")
        if common.matrix_digest(cost) != original_hash:
            raise ValueError(f"{method} diagnostic call mutated the input")
        expected_ld = n if method in ("implicit_scan", "specialized_cached") else n + 1
        if diag["leading_dimension"] != expected_ld:
            raise ValueError(f"{method} leading dimension differs from the planned layout control")
        state = canonical_state(diag)
        states[method] = state
        k = len(normal_pairs)
        records[method] = {
            "objective": value, "dual_F": float(diag["dual_F"]), "pairs": k,
            "row_rejection_fraction": (n - k) / n,
            "smaller_side_unmatched_fraction": (min(n, m) - k) / min(n, m),
            "certificate": certificate(cost, limit, diag, method),
            "state_sha256": {key: state_hash(array) for key, array in state.items()},
            "leading_dimension": diag["leading_dimension"],
            "magnitude_bound": float(diag["magnitude_bound"]),
            "counters": diag["counters"], "input_unchanged": True,
            "diagnostic_matches_ordinary_pairs": True,
        }
    for method in ("implicit_scan", "specialized_cached", "specialized_padded"):
        require_same_state(states["one_sided_materialized"], states[method], method)
        records[method]["bitwise_state_equals_one_sided_upstream"] = True
    records["one_sided_materialized"]["bitwise_state_equals_one_sided_upstream"] = True
    return records


def aggregate_counters(inputs, method):
    records = [item["methods"][method]["counters"] for item in inputs]
    keys = sorted(set().union(*(record.keys() for record in records)))
    result = {}
    for key in keys:
        values = [record.get(key) for record in records]
        result[key] = None if any(v is None for v in values) else sum(int(v) for v in values)
    fast, fallback = result.get("cache_fast_path"), result.get("cache_fallback")
    result["cache_fast_path_share"] = (
        fast / (fast + fallback) if fast is not None and fallback is not None and fast + fallback else None
    )
    result["diagnostic_inputs"] = len(inputs)
    return result


def extra_provenance(dest, manifest_path):
    extras = [ROOT / "README.md", ROOT / "src" / "upstream_download_check.json"]
    manifest_path_out = dest / "provenance_manifest.json"
    inventory = json.loads(manifest_path_out.read_text(encoding="utf-8"))
    for source in extras:
        if not source.is_file():
            raise FileNotFoundError(f"Required documentation/provenance file is missing: {source}")
        relative = source.relative_to(ROOT)
        target = dest / "provenance" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        inventory.append({"path": relative.as_posix(), "sha256": common.digest(source),
                          "bytes": source.stat().st_size, "snapshot": target.relative_to(dest).as_posix()})
    target = dest / "provenance" / "frozen_campaign_manifest.json"
    shutil.copyfile(manifest_path, target)
    inventory.append({"path": str(manifest_path), "sha256": common.digest(manifest_path),
                      "bytes": manifest_path.stat().st_size, "snapshot": target.relative_to(dest).as_posix()})
    common.write_json(manifest_path_out, inventory)
    return json.loads((ROOT / "src" / "upstream_download_check.json").read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "results" / "rectangular-grid-v1.json")
    parser.add_argument("--write-manifest", action="store_true",
                        help="Freeze the 67-case plan without importing or calling solvers")
    parser.add_argument("--session-index", type=int, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, help="Base seed; must match the frozen manifest")
    parser.add_argument("--output", type=Path, help="New output directory; never overwrite an existing run")
    parser.add_argument("--validate-only", action="store_true", help="Validate all campaign inputs without timing")
    args = parser.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be nonnegative")
    if not args.write_manifest and args.session_index is None:
        parser.error("--session-index is required for a campaign invocation")
    return args


def main(argv=None):
    global np
    args = parse_args(argv)
    manifest_path = args.manifest.resolve()
    if args.write_manifest:
        document = manifest_document(args.seed if args.seed is not None else 20261027)
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding="utf-8")) != document:
                raise SystemExit("Existing manifest differs; it will not be overwritten")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            common.write_json(manifest_path, document)
        print(f"Frozen {GRID_ID}: 67 cases, 12 preselected table rows; {document['manifest_sha256']}")
        print(redact(str(manifest_path)))
        return 0
    document = read_manifest(manifest_path, args.seed)
    manifest = document["manifest"]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = (args.output or ROOT / "results" / f"rectangular-s{args.session_index}-{stamp}").resolve()
    dest.mkdir(parents=True, exist_ok=False)
    path = dest / "report.json"
    report = {
        "schema_version": 1, "status": "started", "stage": "initialization",
        "started_utc": common.utc_now(), "session_index": args.session_index,
        "pid": os.getpid(), "validate_only": args.validate_only,
        "manifest_sha256": document["manifest_sha256"], "manifest": manifest,
        "driver_sha256": common.digest(__file__), "shared_harness_sha256": common.digest(common.__file__),
        "output_directory": str(dest), "cases": [],
        "primary_comparison": "original_over_padded",
        "timed_work": "Complete Python calls: shared input validation, per-call column-major conversion, representation/workspace allocation, solver, consistency checks and int64 pair construction; no diagnostic instrumentation or objective summation",
        "untimed_work": "Generation, capped-SciPy reference, full primal-dual certificate, state-equivalence checks, counters, input hashes, warmup and calibration",
        "decomposition": {
            "original_to_one_sided": "Same upstream kernel and layout; the one-sided border changes formulation and search behavior",
            "one_sided_to_padded": "Uniform-penalty specialization with the same fully materialized border and leading dimension; omits known-zero preprocessing, specializes search, and has disclosed parameter/index/code-generation differences",
            "scan_to_compact_tracked": "Rejection-candidate caching within the same compact specialized implementation",
            "compact_tracked_to_padded_tracked": "Leading dimension and materialized border control; strided access and cache mapping can change, not merely storage size",
        },
        "limitations": [
            "All three planned sessions are fresh processes in the current day/window; no reboot or independent-day replication is claimed",
            "Four fixed inputs per case per session; newly seeded inputs across sessions",
            "Paired round ratios and their intervals are descriptive; thermal drift and serial correlation remain possible",
            "The 3% ratio band is practical and descriptive, not a statistical-equivalence conclusion",
            "No other assignment package is a timed comparator; no historical timing is used",
            "No floor arm or kernel-only timing; common adapter work can reduce apparent relative kernel differences",
            "No signed, forbidden, required-matching, application-trace, memory, or batching claims",
            "Supplied and independently downloaded archive equality does not identify this 2017 v4 code as the implementation of the 2016 report",
        ],
    }
    common.write_json(path, report)
    try:
        report["stage"] = "runtime_and_provenance"
        bridge, scipy, lsa, _, previous_threads = common.load_runtime()
        np = common.np
        solves = {name: getattr(bridge, attributes[0]) for name, attributes in METHODS.items()}
        diagnose = getattr(bridge, "diagnose")
        if not all(callable(fn) for fn in solves.values()) or not callable(diagnose):
            raise TypeError("Missing callable campaign API; no fallback is permitted")
        report.update({"python": sys.version, "executable": sys.executable,
                       "platform": common.platform.platform(), "processor": common.platform.processor(),
                       "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
                       "affinity": common.pin_process(),
                       "thread_environment": {"requested": 1, "previous": previous_threads}})
        audit = common.check_upstream_archive()
        common.write_json(dest / "upstream_archive_check.json", audit)
        if not audit["all_members_identical"]:
            raise ValueError("Upstream extracted files differ from the supplied archive")
        report["upstream_integrity"] = {"file_count": audit["file_count"], "all_members_identical": True,
                                        "archive_sha256": audit["archive_sha256"]}
        report["provenance"] = common.snapshot_provenance(dest, bridge)
        report["upstream_download_check"] = extra_provenance(dest, manifest_path)
        common.write_json(path, report)
        schedule = random.Random(manifest["base_seed"] + 10000 * args.session_index + 2)
        for ordinal, spec in enumerate(manifest["cases"], 1):
            report["stage"], report["active_case"] = "validation", spec["case"]
            inputs, identities = inputs_for_case(spec, manifest, args.session_index)
            case = {**spec, "input_identities": identities, "diagnostics": [],
                    "methods": {}, "round_order": [], "status": "validating"}
            report["cases"].append(case)
            print(f"Session {args.session_index}: {ordinal}/67 {spec['case']}", flush=True)
            common.write_json(path, report)
            for index, cost in enumerate(inputs):
                expected = common.reference_objective(cost, spec["limit"], lsa)
                results = validate_input(cost, spec["limit"], expected, solves, diagnose)
                case["diagnostics"].append({"input_index": index, "expected": expected, "methods": results})
            case["diagnostic_coverage"] = {method: aggregate_counters(case["diagnostics"], method)
                                           for method in METHODS}
            case["status"] = "validated"
            common.write_json(path, report)
            if args.validate_only:
                continue
            report["stage"] = "calibration"
            for method, solve in solves.items():
                cycles, elapsed = common.calibrate(solve, inputs, spec["limit"], manifest["target_block_ms"] * 1e6)
                case["methods"][method] = {"cycles_per_round": cycles,
                    "calibration_elapsed_ns": elapsed, "round_average_us": [], "round_elapsed_ns": []}
            report["stage"], case["status"] = "timing", "timing"
            common.write_json(path, report)
            hashes = [item["input_sha256"] for item in identities]
            for round_index in range(manifest["rounds"]):
                report["active_round"] = round_index
                order = list(solves)
                schedule.shuffle(order)
                case["round_order"].append(order)
                for method in order:
                    stats = case["methods"][method]
                    elapsed = common.time_cycles(solves[method], inputs, spec["limit"], stats["cycles_per_round"])
                    stats["round_elapsed_ns"].append(elapsed)
                    stats["round_average_us"].append(elapsed / (1000 * stats["cycles_per_round"] * len(inputs)))
                if [common.matrix_digest(cost) for cost in inputs] != hashes:
                    raise ValueError("A timed solver call modified an input")
            for stats in case["methods"].values():
                q1, median, q3 = np.quantile(stats["round_average_us"], [0.25, 0.5, 0.75])
                if median <= 0:
                    raise ValueError("Non-positive measured latency")
                stats.update(q1_us=float(q1), median_us=float(median), q3_us=float(q3))
            case["status"] = "complete"
            case["input_unchanged_after_timing"] = True
            common.write_json(path, report)
            o = case["methods"]["original"]["median_us"]
            p = case["methods"]["specialized_padded"]["median_us"]
            print(f"  LSAPE-v4 {o:.3f} us; padded adaptation {p:.3f} us; median ratio {o/p:.3f}", flush=True)
        report.pop("active_case", None)
        report.pop("active_round", None)
        report["status"] = "validated_only" if args.validate_only else "complete"
        report["stage"], report["completed_utc"] = "complete", common.utc_now()
        report["completed_cases"] = len(report["cases"])
        common.write_json(path, report)
        print(redact("Saved " + str(path)), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", failed_utc=common.utc_now(),
                      error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        common.write_json(path, report)
        print(redact(f"Campaign failed during {report['stage']}: {error}\nReport: {path}"), file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
