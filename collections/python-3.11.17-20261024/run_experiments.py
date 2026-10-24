"""Reproduce the controlled experiments accompanying the manuscript.

The native rectangular solver with implicit rejection is measured through its three
implementations: JVRUST (laprust; also LAPX lapjvr), JVHYBRID (LAPX lapjvh) and JVCPP
(lapcpp). Baselines are SciPy and LAPX's JVX, JVS and JVC families. All public APIs
receive equivalent float64 costs and objectives. The installed packages are used;
their identities are recorded by record_runtime.py.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import functools
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

PAPER = Path(__file__).resolve().parent
SOLVERS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "jvs", "jvc", "scipy")
LABELS = {"jvrust": "JVRUST", "jvhybrid": "JVHYBRID", "jvcpp": "JVCPP", "jvx": "JVX",
          "jvs": "JVS", "jvc": "JVC", "scipy": "SciPy", "jvrust_lapx": "JVRUST (LAPX lapjvr)"}
# The proposed solver's implementations. JVRUST ships in laprust and as LAPX lapjvr;
# validation covers both packages, timing uses laprust.
NATIVE = ("jvrust", "jvrust_lapx", "jvhybrid", "jvcpp")
NATIVE_MODULES = ("laprust._lapjv", "lapcpp._lapjv", "lap._lapjv_rh", "lap._lapjv",
                  "lap._lapjvx", "lap._lapjvs", "lap._lapjvc")
np = None
lap = None
lapcpp = None
laprust = None
lsa = None
psutil = None


def initialize():
    global np, lap, lapcpp, laprust, lsa, psutil
    import numpy
    import lap as lap_module
    import lapcpp as lapcpp_module
    import laprust as laprust_module
    import psutil as psutil_module
    from scipy.optimize import linear_sum_assignment
    np, lap, lapcpp, laprust = numpy, lap_module, lapcpp_module, laprust_module
    lsa, psutil = linear_sum_assignment, psutil_module


def native_function(name, batch=False):
    """The public function behind each implementation of the proposed solver."""
    suffix = "_batch" if batch else ""
    return {"jvrust": getattr(laprust, "lapjv" + suffix),
            "jvrust_lapx": getattr(lap, "lapjvr" + suffix),
            "jvhybrid": getattr(lap, "lapjvh" + suffix),
            "jvcpp": getattr(lapcpp, "lapjv" + suffix)}[name]


def augmented(cost, limit):
    n, m = cost.shape
    out = np.empty((n + m, n + m), dtype=np.float64)
    out[:n, :m] = cost
    out[:n, m:] = limit / 2
    out[n:, :m] = limit / 2
    out[n:, m:] = 0
    return out


def solve(name, cost, limit=math.inf):
    """Return int64 pairs; preserve each backend's native pair ordering."""
    if name in NATIVE:
        return native_function(name)(
            cost, extend_cost=True, cost_limit=limit, return_cost=False,
            precision="float64", output="pairs",
        )
    if name == "jvx":
        return lap.lapjvxa(
            cost, extend_cost=True, cost_limit=limit, return_cost=False,
        ).astype(np.int64, copy=False)
    finite = math.isfinite(limit)
    work = augmented(cost, limit) if finite else cost
    if name == "jvs":
        pairs = lap.lapjvsa(
            work, extend_cost=True, return_cost=False, prefer_float32=False,
        )
    elif name == "jvc":
        rows, cols = lap.lapjvc(work, return_cost=False)
        pairs = np.column_stack((rows, cols))
    elif name == "scipy":
        rows, cols = lsa(work)
        pairs = np.column_stack((rows, cols))
    else:
        raise ValueError(name)
    if finite:
        pairs = pairs[(pairs[:, 0] < cost.shape[0]) & (pairs[:, 1] < cost.shape[1])]
    return pairs.astype(np.int64, copy=False)


def objective(cost, pairs, limit):
    selected = float(cost[pairs[:, 0], pairs[:, 1]].sum(dtype=np.float64))
    if math.isfinite(limit):
        selected += limit / 2 * (sum(cost.shape) - 2 * len(pairs))
    return selected


def check_pairs(cost, pairs, limit):
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert pairs.dtype == np.int64
    assert len(pairs) <= min(cost.shape)
    assert len(np.unique(pairs[:, 0])) == len(pairs)
    assert len(np.unique(pairs[:, 1])) == len(pairs)
    assert np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < cost.shape[0]))
    assert np.all((pairs[:, 1] >= 0) & (pairs[:, 1] < cost.shape[1]))
    assert np.all(np.isfinite(cost[pairs[:, 0], pairs[:, 1]]))
    if not math.isfinite(limit):
        assert len(pairs) == min(cost.shape)
    return objective(cost, pairs, limit)


def brute_objective(cost, limit):
    """Independent exact enumeration on small binary-rational instances."""
    a = cost if cost.shape[0] <= cost.shape[1] else cost.T
    r, c = a.shape

    @functools.lru_cache(None)
    def visit(i, used):
        if i == r:
            return 0.0
        best = limit + visit(i + 1, used) if math.isfinite(limit) else math.inf
        for j in range(c):
            if not used & (1 << j) and np.isfinite(a[i, j]):
                best = min(best, float(a[i, j]) + visit(i + 1, used | (1 << j)))
        return best

    answer = visit(0, 0)
    if math.isfinite(limit):
        answer += limit / 2 * (c - r)
    return answer


def validate(seed):
    rng = np.random.default_rng(seed)
    report = {
        "seed": seed, "instances": 0, "solver_calls": 0, "infeasible_calls": 0,
        "maximum_absolute_objective_error": 0.0, "failures": [],
        "input_dtypes": ["float32", "float64"],
        "cost_grid": "integers from -16 through 16 divided by 8",
        "limits": ["infinity", -1.0, 0.0, 1.0, 2.0],
        "forbidden_probability": 0.2,
        "solvers": [LABELS[name] for name in NATIVE],
    }
    for dtype in (np.float32, np.float64):
        for n in range(6):
            for m in range(7):
                for repetition in range(10):
                    cost = (rng.integers(-16, 17, size=(n, m)) / 8).astype(dtype)
                    cost[rng.random((n, m)) < 0.2] = np.inf
                    for limit in (math.inf, -1.0, 0.0, 1.0, 2.0):
                        expected = brute_objective(cost, limit)
                        report["instances"] += 1
                        for name in NATIVE:
                            report["solver_calls"] += 1
                            try:
                                pairs = native_function(name)(
                                    cost, extend_cost=True, cost_limit=limit,
                                    return_cost=False, output="pairs", precision="native",
                                )
                                actual = check_pairs(cost, pairs, limit)
                                if not math.isfinite(expected):
                                    raise AssertionError("infeasible instance returned a matching")
                                error = abs(actual - expected)
                                report["maximum_absolute_objective_error"] = max(
                                    report["maximum_absolute_objective_error"], error
                                )
                                assert error == 0.0, (expected, actual)
                            except ValueError as exc:
                                if math.isinf(expected):
                                    report["infeasible_calls"] += 1
                                else:
                                    report["failures"].append(
                                        {"solver": name, "shape": [n, m],
                                         "limit": str(limit), "error": str(exc)}
                                    )
                            except Exception as exc:
                                report["failures"].append(
                                    {"solver": name, "shape": [n, m],
                                     "limit": str(limit), "error": repr(exc)}
                                )
    return report


def case_specs():
    return [
        {"case": "square128", "n": 128, "m": 128, "limit": math.inf},
        {"case": "square512", "n": 512, "m": 512, "limit": math.inf},
        {"case": "wide64x512", "n": 64, "m": 512, "limit": math.inf},
        {"case": "tall512x64", "n": 512, "m": 64, "limit": math.inf},
        {"case": "near256x320", "n": 256, "m": 320, "limit": math.inf},
        {"case": "blocks256x320", "n": 256, "m": 320, "limit": math.inf, "blocks": True},
        {"case": "ties128x160", "n": 128, "m": 160, "limit": math.inf, "ties": True},
        {"case": "optional128x160", "n": 128, "m": 160, "limit": 0.5, "forbidden": 0.5},
        {"case": "optional256x320", "n": 256, "m": 320, "limit": 0.5, "forbidden": 0.5},
        {"case": "tight128x160", "n": 128, "m": 160, "limit": 0.05, "forbidden": 0.5},
        {"case": "negative128x160", "n": 128, "m": 160, "limit": -0.25,
         "forbidden": 0.2, "signed": True},
        {"case": "optional_square128", "n": 128, "m": 128, "limit": 0.5, "forbidden": 0.5},
    ]


def make_cost(spec, rng):
    cost = rng.random((spec["n"], spec["m"]))
    if spec.get("signed"):
        cost = 2 * cost - 1
    if spec.get("blocks"):
        # Three quarters of rows compete for one quarter of columns.
        split_r = 3 * spec["n"] // 4
        split_c = spec["m"] // 4
        cost[:split_r, split_c:] += 3
        cost[split_r:, :split_c] += 3
    if spec.get("ties"):
        cost = np.floor(cost * 8) / 8
    if spec.get("forbidden"):
        mask = rng.random(cost.shape) < spec["forbidden"]
        if not math.isfinite(spec["limit"]):
            np.fill_diagonal(mask, False)
        cost[mask] = np.inf
    return np.ascontiguousarray(cost, dtype=np.float64)


@contextlib.contextmanager
def no_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def timed_block(call, inputs, number):
    with no_gc():
        start = time.perf_counter_ns()
        for _ in range(number):
            for item in inputs:
                call(item)
        elapsed = time.perf_counter_ns() - start
    return elapsed / (number * len(inputs))


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(samples, groups):
    result = []
    for keys in sorted({tuple(row[k] for k in groups) for row in samples}):
        selected = [row for row in samples if tuple(row[k] for k in groups) == keys]
        values = np.asarray([row["nanoseconds"] for row in selected])
        result.append({
            **dict(zip(groups, keys)), "median_ns": float(np.median(values)),
            "q25_ns": float(np.percentile(values, 25)),
            "q75_ns": float(np.percentile(values, 75)), "rounds": len(values),
        })
    return result


def single_experiment(options):
    rng = np.random.default_rng(options.seed + 1)
    order_rng = random.Random(options.seed + 2)
    rows, checks = [], []
    for spec in case_specs():
        inputs = [make_cost(spec, rng) for _ in range(options.instances)]
        reference = [objective(a, solve("scipy", a, spec["limit"]), spec["limit"]) for a in inputs]
        calls = {}
        numbers = {}
        for name in SOLVERS:
            calls[name] = functools.partial(solve, name, limit=spec["limit"])
            for index, a in enumerate(inputs):
                pairs = calls[name](a)
                actual = check_pairs(a, pairs, spec["limit"])
                if not np.isclose(actual, reference[index], rtol=1e-10, atol=1e-10):
                    raise AssertionError((spec["case"], name, index, actual, reference[index]))
            probe = timed_block(calls[name], inputs, 1)
            numbers[name] = max(1, min(100000, round(options.block_seconds * 1e9 / (probe * len(inputs)))))
        for round_index in range(options.rounds):
            order = list(SOLVERS)
            order_rng.shuffle(order)
            for position, name in enumerate(order):
                elapsed = timed_block(calls[name], inputs, numbers[name])
                rows.append({
                    "case": spec["case"], "solver": name, "n": spec["n"], "m": spec["m"],
                    "limit": str(spec["limit"]), "round": round_index, "position": position,
                    "instances": len(inputs), "cycles": numbers[name], "nanoseconds": elapsed,
                })
        checks.append({"case": spec["case"], "instances": len(inputs),
                       "solvers": len(SOLVERS), "objective_checks_passed": True})
        print("Single", spec["case"], flush=True)
    return rows, checks


def batch_call(name, costs, limit):
    if name.endswith("_loop"):
        solver = name.removesuffix("_loop")
        return [solve(solver, a, limit) for a in costs]
    family, suffix = name.rsplit("_batch_t", 1)
    threads = int(suffix)
    if family in NATIVE:
        return native_function(family, batch=True)(
            costs, extend_cost=True, cost_limit=limit, return_cost=False,
            n_threads=threads, precision="float64", output="pairs",
        )
    if family == "jvx":
        return lap.lapjvxa_batch(
            costs, extend_cost=True, cost_limit=limit, return_cost=False,
            n_threads=threads,
        )
    raise ValueError(name)


BATCH_FAMILIES = ("jvrust", "jvhybrid", "jvcpp", "jvx")


def batch_specs():
    return [
        {"case": "batch64x64x80", "b": 64, "n": 64, "m": 80, "limit": math.inf},
        {"case": "batch16x128x160_optional", "b": 16, "n": 128, "m": 160,
         "limit": 0.5, "forbidden": 0.5},
        # Many tiny problems, as in per-frame tracking, expose per-problem overhead.
        {"case": "batch1024x8x10", "b": 1024, "n": 8, "m": 10, "limit": math.inf},
    ]


def batch_names(batch_threads):
    names = ["jvrust_loop", "scipy_loop"]
    for family in BATCH_FAMILIES:
        names += [f"{family}_batch_t1", f"{family}_batch_t{batch_threads}"]
    return list(dict.fromkeys(names))


def check_batch(costs, actual, expected, limit):
    """Validate every requested output and return the number actually checked."""
    if not len(actual) == len(costs) == len(expected):
        raise AssertionError(
            f"batch length mismatch: costs={len(costs)}, outputs={len(actual)}, "
            f"references={len(expected)}"
        )
    checked = 0
    for a, pairs, reference in zip(costs, actual, expected, strict=True):
        value = check_pairs(a, pairs.astype(np.int64, copy=False), limit)
        assert np.isclose(value, reference, rtol=1e-10, atol=1e-10), (checked, value, reference)
        checked += 1
    return checked


def batch_experiment(options, batch_threads, *, validation_only=False):
    rng = np.random.default_rng(options.seed + 3)
    order_rng = random.Random(options.seed + 4)
    names = batch_names(batch_threads)
    rows, checks = [], []
    for spec in batch_specs():
        costs = np.stack([make_cost(spec, rng) for _ in range(spec["b"])])
        expected = [objective(a, solve("scipy", a, spec["limit"]), spec["limit"]) for a in costs]
        calls, numbers, checked_outputs = {}, {}, {}
        for name in names:
            calls[name] = functools.partial(batch_call, name, limit=spec["limit"])
            actual = calls[name](costs)
            checked_outputs[name] = check_batch(costs, actual, expected, spec["limit"])
            if not validation_only:
                probe = timed_block(calls[name], [costs], 1)
                numbers[name] = max(1, min(10000, round(options.block_seconds * 1e9 / probe)))
        for round_index in range(0 if validation_only else options.rounds):
            order = list(names)
            order_rng.shuffle(order)
            for position, name in enumerate(order):
                elapsed = timed_block(calls[name], [costs], numbers[name])
                rows.append({
                    "case": spec["case"], "solver": name, "b": spec["b"],
                    "n": spec["n"], "m": spec["m"], "limit": str(spec["limit"]),
                    "round": round_index, "position": position,
                    "cycles": numbers[name], "nanoseconds": elapsed,
                })
        checks.append({"case": spec["case"], "solvers": len(names),
                       "problems": len(costs), "checked_outputs": checked_outputs,
                       "input_sha256": hashlib.sha256(costs.tobytes()).hexdigest(),
                       "objective_checks_passed": True})
        print("Batch", spec["case"], flush=True)
    return rows, checks


def memory_child(options):
    process = psutil.Process()
    available = process.cpu_affinity()
    if available:
        process.cpu_affinity(available[:1])
    limit = 0.5 if options.memory_case == "optional" else math.inf
    solve(options.memory_solver, np.ones((4, 4)), limit)
    rng = np.random.default_rng(options.seed)
    cost = rng.random((256, 4096))
    gc.collect()
    before = process.memory_info()
    result = solve(options.memory_solver, cost, limit)
    after = process.memory_info()
    check_pairs(cost, result, limit)
    report = {
        "case": options.memory_case, "solver": options.memory_solver,
        "n": 256, "m": 4096, "limit": str(limit),
        "rss_before": before.rss, "rss_after": after.rss,
        "peak_before": getattr(before, "peak_wset", None),
        "peak_after": getattr(after, "peak_wset", None),
    }
    if report["peak_after"] is not None:
        report["additional_peak"] = max(0, report["peak_after"] - report["peak_before"])
    print(json.dumps(report), flush=True)


MEMORY_SOLVERS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "scipy")


def memory_experiment(options):
    reports = []
    for case in ("required", "optional"):
        for name in MEMORY_SOLVERS:
            command = [
                sys.executable, str(Path(__file__).resolve()), "--memory-solver", name,
                "--memory-case", case, "--seed", str(options.seed),
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            report = json.loads(result.stdout.strip().splitlines()[-1])
            reports.append(report)
            print("Memory", case, name, flush=True)
    return reports


def environment(options, single_affinity, batch_affinity):
    import scipy
    from threadpoolctl import threadpool_info
    native = {}
    for name in NATIVE_MODULES:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        native[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    cpu_name = platform.processor()
    if os.name == "nt":
        try:
            cpu_name = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_Processor).Name"], text=True,
            ).strip()
        except Exception:
            pass
    return {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "cpu": cpu_name,
        "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "packages": {name: {"version": module.__version__,
                            "path": str(Path(module.__file__).resolve().parent)}
                     for name, module in (("lapx", lap), ("lapcpp", lapcpp), ("laprust", laprust))},
        "single_affinity": single_affinity, "batch_affinity": batch_affinity,
        "threadpools": threadpool_info(), "seed": options.seed, "rounds": options.rounds,
        "instances_per_single_case": options.instances,
        "target_block_seconds": options.block_seconds,
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_modules": native,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "results")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--block-seconds", type=float, default=0.03)
    parser.add_argument("--validate-batch-only", action="store_true",
                        help="Check batch outputs and save batch_validation.json without timing.")
    parser.add_argument("--memory-solver", choices=SOLVERS)
    parser.add_argument("--memory-case", choices=("required", "optional"))
    options = parser.parse_args()
    initialize()
    if options.memory_solver:
        memory_child(options)
        return
    from threadpoolctl import threadpool_limits
    options.output.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    original_affinity = process.cpu_affinity()
    single_affinity = original_affinity[:1]
    batch_affinity = original_affinity[:4]
    try:
        if options.validate_batch_only:
            options.rounds = 0
            process.cpu_affinity(batch_affinity)
            with threadpool_limits(limits=1):
                _, checks = batch_experiment(options, len(batch_affinity), validation_only=True)
                report = {
                    "status": "passed",
                    "scope": "Batch correctness only; timing measurements were not rerun.",
                    "environment": environment(options, single_affinity, batch_affinity),
                    "checks": checks,
                    "checked_outputs": sum(sum(row["checked_outputs"].values()) for row in checks),
                }
            target = options.output / "batch_validation.json"
            target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print("Batch validation:", report["checked_outputs"], "outputs passed; saved", target)
            return
        process.cpu_affinity(single_affinity)
        with threadpool_limits(limits=1):
            validation = validate(options.seed)
            (options.output / "validation.json").write_text(
                json.dumps(validation, indent=2) + "\n", encoding="utf-8"
            )
            if validation["failures"]:
                raise AssertionError(validation["failures"][:5])
            print("Validation", validation["solver_calls"], "calls passed", flush=True)
            single, single_checks = single_experiment(options)
            csv_write(options.output / "single_samples.csv", single)
            csv_write(options.output / "single_summary.csv", summary(single, ["case", "solver"]))
            process.cpu_affinity(batch_affinity)
            batch, batch_checks = batch_experiment(options, len(batch_affinity))
            csv_write(options.output / "batch_samples.csv", batch)
            csv_write(options.output / "batch_summary.csv", summary(batch, ["case", "solver"]))
            metadata = environment(options, single_affinity, batch_affinity)
            (options.output / "environment.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            (options.output / "benchmark_checks.json").write_text(
                json.dumps({"single": single_checks, "batch": batch_checks}, indent=2) + "\n",
                encoding="utf-8",
            )
        memory = memory_experiment(options)
        (options.output / "memory.json").write_text(json.dumps(memory, indent=2) + "\n", encoding="utf-8")
    finally:
        process.cpu_affinity(original_affinity)
    print("Results saved:", options.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
