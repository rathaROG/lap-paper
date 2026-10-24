"""Collect a separate, resumable extension using the paper's existing adapters.

No manuscript, previous result, previous figure, or arXiv file is written.
All timings use four paired float64 inputs and nine shuffled rounds by default.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import functools
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import run_experiments as base
import compare_optional as optional

PAPER = Path(__file__).resolve().parent
REQUIRED = ("jvrust", "jvhybrid", "jvcpp", "scipy")
OPTIONAL = ("jvrust", "jvcpp", "scipy_capped", "scipy_dummies")
MEMORY_REQUIRED = ("jvrust", "jvcpp", "scipy", "jvx")
MEMORY_OPTIONAL = ("jvrust", "jvcpp", "scipy_capped", "scipy_dummies", "scipy_square")
SOURCE_NAMES = ("run_extended_experiments.py", "run_experiments.py", "compare_optional.py")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(memoryview(array)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def limit_of(spec):
    return math.inf if spec["limit"] is None else spec["limit"]


def solve(method, cost, limit):
    if method.startswith("scipy_"):
        return optional.solve(method, cost, limit)
    return base.solve(method, cost, limit)


def methods(spec):
    return REQUIRED if spec["limit"] is None else OPTIONAL


def plan(smoke=False):
    cases = []

    def add(family, n, m, limit, key, **extras):
        regime = "required" if limit is None else "optional"
        item = dict(family=family, n=n, m=m, regime=regime, limit=limit,
                    input_key=key, distribution="uniform")
        item.update(extras)
        item["case"] = f"{len(cases):03d}_{family}_{regime}_{n}x{m}"
        cases.append(item)

    squares = (32,) if smoke else (1000, 2000, 3000, 5000)
    rectangles = ((24, 48),) if smoke else ((1000, 2000), (2000, 3000), (4000, 5000))
    extremes = ((8, 128),) if smoke else ((64, 4096), (64, 8192), (64, 16384))
    for n in squares:
        for limit in (None, 2 / n):
            add("large_square", n, n, limit, f"large_square_{n}")
    for family, shapes in (("large_rectangle", rectangles), ("extreme_rectangle", extremes)):
        for n, m in shapes:
            for rows, cols in ((n, m), (m, n)):
                for limit in (None, 2 / max(n, m)):
                    add(family, rows, cols, limit, f"{family}_{n}x{m}")
    sizes = ((24, 32),) if smoke else ((256, 320), (512, 640))
    for n, m in sizes:
        for congestion in ((.75,) if smoke else (.25, .5, .625, .75, .9)):
            for limit in (None, .5):
                add("congestion", n, m, limit, f"congestion_{n}x{m}",
                    distribution="blocks", row_fraction=congestion, column_fraction=.25)
        for forbidden in ((.9,) if smoke else (0., .5, .9, .99)):
            for limit in (None, 2 / m):
                add("forbidden", n, m, limit, f"forbidden_{n}x{m}",
                    forbidden=forbidden, planted_feasible=True)
        for levels in ((2,) if smoke else (1, 2, 8, 256)):
            for limit in (None, .25):
                add("ties", n, m, limit, f"ties_{n}x{m}", levels=levels)
    n, m = (24, 32) if smoke else (256, 320)
    for rows, cols in ((n, m), (m, n)):
        for limit in (None, -.25):
            add("signed", rows, cols, limit, f"signed_{n}x{m}",
                signed=True, forbidden=.5, planted_feasible=True)
    for n, m in (((24, 32),) if smoke else ((128, 160), (256, 320))):
        for distribution in ("uniform", "blocks"):
            for limit in ((.001, .5) if smoke else (0., .0005, .001, .002, .005, .01, .02, .05, .1, .5, 4.)):
                extra = dict(forbidden=.5) if distribution == "uniform" else dict(row_fraction=.75, column_fraction=.25)
                add("rejection", n, m, limit, f"rejection_{distribution}_{n}x{m}",
                    distribution=distribution, **extra)
    memory = []
    widths = (64,) if smoke else (256, 512, 1024, 2048, 4096)
    n = 16 if smoke else 256
    for m in widths:
        orientations = ((n, m),) if smoke or n == m else ((n, m), (m, n))
        for rows, cols in orientations:
            for limit in (None, .5):
                item = dict(case=f"memory_{'required' if limit is None else 'optional'}_{rows}x{cols}",
                            n=rows, m=cols, limit=limit, regime="required" if limit is None else "optional")
                for method in (MEMORY_REQUIRED if limit is None else MEMORY_OPTIONAL):
                    for trial in range(2 if smoke else 5):
                        memory.append(dict(item, solver=method, trial=trial, job=f"{item['case']}_{method}_{trial}"))
    return dict(timing=cases, memory=memory)


def make_input(spec, seed, instance):
    """Paired costs across limits/structure levels; tall inputs are exact transposes."""
    np = base.np
    small, large = sorted((spec["n"], spec["m"]))
    key = int(hashlib.sha256(spec["input_key"].encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([seed, key, instance])
    cost = rng.random((small, large))
    if spec.get("signed"):
        cost = 2 * cost - 1
    if "levels" in spec:
        cost = np.floor(cost * spec["levels"]) / spec["levels"]
    if spec["distribution"] == "blocks":
        r = int(spec["row_fraction"] * small)
        c = int(spec["column_fraction"] * large)
        cost[:r, c:] += 3
        cost[r:, :c] += 3
    if "forbidden" in spec:
        mask = rng.random((small, large)) < spec["forbidden"]
        if spec.get("planted_feasible"):
            np.fill_diagonal(mask, False)
        cost[mask] = np.inf
    canonical_hash = array_digest(cost)
    if spec["n"] > spec["m"]:
        cost = np.ascontiguousarray(cost.T)
    return cost, canonical_hash


def checked_objective(cost, pairs, limit, expected):
    value = base.check_pairs(cost, pairs, limit)
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError((value, expected))
    return value


def validate_edges():
    np = base.np
    cases = [(f"empty_{n}x{m}", np.empty((n, m)), limit)
             for n, m in ((0, 0), (0, 5), (5, 0)) for limit in (math.inf, .5)]
    narrow = np.array([[-1., 0., .5, 2., np.inf]])
    cases += [(f"single_{orientation}_{limit}", a, limit)
              for orientation, a in (("wide", narrow), ("tall", narrow.T.copy()))
              for limit in (math.inf, -.25, .5)]
    cases += [(f"forbidden_{n}x{m}_{limit}", np.full((n, m), np.inf), limit)
              for n, m in ((3, 5), (5, 3)) for limit in (math.inf, .5)]
    cases += [("all_zero", np.zeros((4, 6)), math.inf),
              ("all_zero_optional", np.zeros((4, 6)), .5),
              ("all_equal_to_limit", np.full((4, 6), .5), .5),
              ("signed", np.array([[-1., -.25, np.inf], [.5, -1., .25]]), -.25),
              ("required_infeasible", np.array([[0., 1.], [np.inf, np.inf]]), math.inf),
              ("optional_infeasible_row", np.array([[0., 1.], [np.inf, np.inf]]), .5),
              ("penalty_boundary", np.array([[np.nextafter(.5, 0.), .5, np.nextafter(.5, 1.)],
                                            [.5, np.nextafter(.5, 1.), np.nextafter(.5, 0.)]]), .5)]
    checks = []
    for name, cost, limit in cases:
        expected = base.brute_objective(cost, limit)
        candidates = (REQUIRED if math.isinf(limit) else OPTIONAL + ("jvhybrid",)) + ("jvrust_lapx",)
        for method in candidates:
            try:
                pairs = solve(method, cost, limit)
            except ValueError:
                if math.isfinite(expected):
                    raise
                checks.append(dict(case=name, solver=method, status="expected_infeasible"))
                continue
            assert math.isfinite(expected), (name, method, "infeasible problem returned pairs")
            actual = checked_objective(cost, pairs, limit, expected)
            checks.append(dict(case=name, solver=method, status="passed", pairs=len(pairs),
                               objective=actual, reference_objective=expected))
    return dict(status="passed", instances=len(cases), checks=checks,
                method="Independent exhaustive objective enumeration; pair identity is not required for ties.")


def timing_worker(output, index):
    manifest = read(output / "plan.json")
    spec = manifest["timing"][index]
    cfg = manifest["configuration"]
    base.initialize()
    base.psutil.Process().cpu_affinity([cfg["cpu"]])
    from threadpoolctl import threadpool_limits
    path = output / "cases" / (spec["case"] + ".json")
    report = dict(spec=spec, status="running", started_utc=utc(), samples=[], checks=[], methods={}, inputs=[])
    save(path, report)
    with threadpool_limits(limits=1):
        inputs = []
        for i in range(cfg["instances"]):
            a, canonical = make_input(spec, cfg["seed"], i)
            inputs.append(a)
            report["inputs"].append(dict(instance=i, sha256=array_digest(a), canonical_sha256=canonical,
                                         forbidden_fraction=float(base.np.isinf(a).mean())))
        limit = limit_of(spec)
        reference_method = "scipy" if spec["limit"] is None else "scipy_capped"
        expected = [base.check_pairs(a, solve(reference_method, a, limit), limit) for a in inputs]
        calls, cycles = {}, {}
        for method in methods(spec):
            report["active_method"] = method
            save(path, report)
            start = time.perf_counter()
            calls[method] = functools.partial(solve, method, limit=limit)
            for i, a in enumerate(inputs):
                pairs = calls[method](a)
                value = checked_objective(a, pairs, limit, expected[i])
                report["checks"].append(dict(case=spec["case"], solver=method, instance=i, pairs=len(pairs),
                                             objective=value, reference_objective=expected[i]))
            probe = base.timed_block(calls[method], inputs, 1)
            cycles[method] = max(1, min(100000, round(cfg["block_seconds"] * 1e9 / (probe * len(inputs)))))
            projected = probe * len(inputs) * cycles[method] * cfg["rounds"] / 1e9
            status = "budget_excluded" if projected > cfg["method_budget_seconds"] else "ready"
            report["methods"][method] = dict(status=status, probe_ns=probe, cycles=cycles[method],
                projected_timing_seconds=projected, validation_and_probe_seconds=time.perf_counter()-start,
                measured_seconds=0.)
            print(f"{spec['case']} {method}: validated; projected {projected:.2f}s; {status}", flush=True)
            save(path, report)
        order_rng = random.Random(cfg["seed"] + 10000 + index)
        for round_index in range(cfg["rounds"]):
            order = list(methods(spec))
            order_rng.shuffle(order)
            for position, method in enumerate(order):
                info = report["methods"][method]
                if info["status"] not in ("ready", "running"):
                    continue
                info["status"] = "running"
                report["active_method"] = method
                save(path, report)
                start = time.perf_counter()
                elapsed = base.timed_block(calls[method], inputs, cycles[method])
                info["measured_seconds"] += time.perf_counter() - start
                report["samples"].append(dict(case=spec["case"], solver=method, round=round_index,
                    position=position, instances=len(inputs), cycles=cycles[method], nanoseconds=elapsed))
                if info["measured_seconds"] > cfg["method_budget_seconds"] and round_index + 1 < cfg["rounds"]:
                    info["status"] = "budget_exceeded"
                elif round_index + 1 == cfg["rounds"]:
                    info["status"] = "completed"
                save(path, report)
        assert [array_digest(a) for a in inputs] == [row["sha256"] for row in report["inputs"]], "Solver changed input costs"
        report.update(status="completed", completed_utc=utc(), active_method=None)
        save(path, report)


def memory_worker(output, index):
    import gc
    manifest = read(output / "plan.json")
    spec, cfg = manifest["memory"][index], manifest["configuration"]
    base.initialize()
    process = base.psutil.Process()
    process.cpu_affinity([cfg["cpu"]])
    from threadpoolctl import threadpool_limits
    limit = limit_of(spec)
    with threadpool_limits(limits=1):
        # Warm imports/workspace setup before preparing the resident input.
        solve(spec["solver"], base.np.ones((4, 4)), limit)
        rng = base.np.random.default_rng([cfg["seed"], 30000, spec["n"], spec["m"], spec["trial"]])
        cost = rng.random((spec["n"], spec["m"]))
        input_hash = array_digest(cost)  # memoryview: no extra dense copy before measuring.
        gc.collect()
        before = process.memory_info()
        pairs = solve(spec["solver"], cost, limit)
        after = process.memory_info()
        # The reference and all checking run AFTER capturing the measured peak.
        reference = "scipy" if spec["limit"] is None else "scipy_capped"
        expected = base.check_pairs(cost, solve(reference, cost, limit), limit)
        value = checked_objective(cost, pairs, limit, expected)
    r, c = sorted((spec["n"], spec["m"]))
    buffers = {"jvrust": 0, "jvcpp": 0, "scipy": 0, "jvx": c*c*8 if r != c else 0,
               "scipy_capped": r*c*8, "scipy_dummies": r*(c+r)*8, "scipy_square": (r+c)**2*8}
    report = dict(spec, status="passed", pid=os.getpid(), python_executable=sys.executable,
                  input_sha256=input_hash, input_bytes=cost.nbytes, rss_before=before.rss, rss_after=after.rss,
                  peak_before=before.peak_wset, peak_after=after.peak_wset,
                  additional_peak=max(0, after.peak_wset-before.peak_wset),
                  baseline_peak_headroom=max(0, before.peak_wset-before.rss),
                  transformed_dense_buffer_bytes=buffers[spec["solver"]],
                  objective=value, reference_objective=expected, completed_utc=utc())
    save(output / "memory_cases" / (spec["job"] + ".json"), report)


def aggregate(output):
    samples, valid, datasets, checks, statuses = [], [], [], [], []
    for path in sorted((output / "cases").glob("*.json")):
        data = read(path)
        samples.extend(data["samples"])
        checks.extend(data["checks"])
        datasets.append(dict(data["spec"], inputs=data["inputs"], status=data["status"]))
        for method, info in data["methods"].items():
            statuses.append(dict(case=data["spec"]["case"], solver=method, **info))
            if data["status"] == "completed" and info["status"] == "completed":
                valid.extend(row for row in data["samples"] if row["solver"] == method)
    if samples:
        base.csv_write(output / "samples.csv", samples)
    if valid:
        base.csv_write(output / "summary.csv", base.summary(valid, ("case", "solver")))
    save(output / "datasets.json", datasets)
    save(output / "checks.json", checks)
    save(output / "method_status.json", statuses)
    memory = [read(p) for p in sorted((output / "memory_cases").glob("*.json"))]
    save(output / "memory_samples.json", memory)
    memory_summary = []
    for case, solver in sorted({(row["case"], row["solver"]) for row in memory}):
        group = [row for row in memory if (row["case"], row["solver"]) == (case, solver)]
        values = base.np.array([row["additional_peak"] for row in group])
        first = group[0]
        memory_summary.append(dict(case=case, solver=solver, n=first["n"], m=first["m"], regime=first["regime"],
            trials=len(group), median_bytes=float(base.np.median(values)), q25_bytes=float(base.np.percentile(values,25)),
            q75_bytes=float(base.np.percentile(values,75)), min_bytes=int(values.min()), max_bytes=int(values.max()),
            input_bytes=first["input_bytes"], transformed_dense_buffer_bytes=first["transformed_dense_buffer_bytes"]))
    if memory_summary:
        base.csv_write(output / "memory_summary.csv", memory_summary)
    return dict(timing_cases=len(datasets), timing_samples=len(samples), completed_method_settings=len({(r['case'],r['solver']) for r in valid}),
                memory_trials=len(memory), objective_checks=len(checks),
                excluded_method_settings=sum(r["status"] not in ("completed", "ready", "running") for r in statuses))


def protected_files():
    paths = []  # Original paper and both 3.14 collections are protected by the outer collection manifest.
    for directory in ("results", "figures", "latex", "templates", "output/arxiv"):
        paths.extend(p for p in (PAPER / directory).rglob("*") if p.is_file())
    return {p.relative_to(PAPER).as_posix(): digest(p) for p in sorted(paths)}


def verify_protected(output):
    for name, expected in read(output / "protected_files.json").items():
        assert digest(PAPER / name) == expected, f"Existing artifact changed: {name}"


def controller(args):
    base.initialize()
    output = args.output.resolve()
    expected_versions = {"lapx": "0.11.1", "lapcpp": "1.0.0", "laprust": "1.0.0"}
    assert sys.version_info[:3] == (3, 11, 17), "This collection is restricted to Python 3.11.17."
    assert Path(sys.executable).resolve() == Path(r"C:\dev\exc\python\p311\python.exe").resolve()
    assert {name: importlib.metadata.version(name) for name in expected_versions} == expected_versions
    process = base.psutil.Process()
    affinity = process.cpu_affinity()
    hashes = {name: digest(PAPER / name) for name in SOURCE_NAMES}
    if args.resume:
        manifest = read(output / "plan.json")
        assert read(output / "run.json")["source_sha256"] == hashes, "Measured scripts changed; use a new output directory."
        assert manifest["configuration"]["smoke"] == args.smoke
        verify_protected(output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = plan(args.smoke)
        manifest["configuration"] = dict(seed=20261024, instances=4, rounds=9, block_seconds=.03, cpu=affinity[0],
            smoke=args.smoke, method_budget_seconds=240, case_timeout_seconds=1200, memory_timeout_seconds=120,
            precision="float64", objective_rtol=1e-10, objective_atol=1e-10)
        manifest["protocol"] = {
            "required_methods": REQUIRED, "optional_methods": OPTIONAL,
            "omissions": "JVX, JVS and JVC are outside the focused timing comparison; existing all-solver results are preserved. JVX remains a required-memory reference. JVHYBRID shares the finite-limit Rust path.",
            "pairing": "Four deterministic input matrices per timing case, shared by all methods. Input arrays are reused across threshold/structure levels and exact transposes across orientations.",
            "large_optional_limit": "L=2/max(N,M), chosen before timing to retain rejection-active inputs as size grows; actual cardinalities are recorded.",
            "timing": "Validation, data generation and calibration are outside timing. Nine shuffled rounds include adapters and output construction; each method uses a calibrated block targeting 30 ms.",
            "budgets": "Methods projected to require more than 240 timing seconds are excluded before rounds; methods exceeding that budget mid-run retain partial samples but have no summary. Whole-case wall timeout is 1200 seconds. All omissions remain explicit in method_status.json and logs.",
            "memory": "Five fresh processes per full condition, single CPU and one library thread. Resident float64 input prepared before baseline. High-water increments, baseline headroom and exact transformed-buffer sizes are recorded separately. Memory orientations use distribution-matched independent arrays to avoid a pre-baseline transpose allocation.",
        }
        save(output / "plan.json", manifest)
        save(output / "protected_files.json", protected_files())
    cfg = manifest["configuration"]
    env_options = argparse.Namespace(seed=cfg["seed"], rounds=cfg["rounds"], instances=cfg["instances"], block_seconds=cfg["block_seconds"])
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        environment = base.environment(env_options, [cfg["cpu"]], [])
    baseline = read(PAPER / "results" / "runtime.json")
    for name, binary in environment["native_modules"].items():
        assert binary["sha256"] == baseline["native_modules"][name]["sha256"]
    environment.update(executable=sys.executable, dependencies={name:importlib.metadata.version(name) for name in
        ("numpy", "scipy", "matplotlib", "psutil", "threadpoolctl")})
    save(output / "environment.json", environment)
    for name in ("cases", "memory_cases", "logs", "sources"):
        (output / name).mkdir(exist_ok=True)
    for name in SOURCE_NAMES:
        (output / "sources" / name).write_bytes((PAPER / name).read_bytes())
    run = dict(status="running", started_utc=utc(), python_executable=sys.executable,
               source_sha256=hashes, planned_timing_cases=len(manifest["timing"]), planned_memory_trials=len(manifest["memory"]), failures=[])
    save(output / "run.json", run)
    print(f"Plan: {len(manifest['timing'])} timing cases, {len(manifest['memory'])} fresh-process memory trials", flush=True)
    stop = threading.Event()
    state = {"stage": "validation"}

    def monitor():
        path = output / "cpu_load.csv"
        append = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            fields = ["utc", "stage", "mean_cpu_percent", "mean_outside_cpu0_3_percent"] + [f"cpu{i}_percent" for i in range(base.psutil.cpu_count())]
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not append:
                writer.writeheader()
            while not stop.is_set():
                values = base.psutil.cpu_percent(interval=1, percpu=True)
                writer.writerow(dict(utc=utc(), stage=state["stage"], mean_cpu_percent=sum(values)/len(values),
                    mean_outside_cpu0_3_percent=sum(values[4:])/len(values[4:]),
                    **{f"cpu{i}_percent":value for i,value in enumerate(values)}))
                stream.flush()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        process.cpu_affinity([cfg["cpu"]])
        with threadpool_limits(limits=1):
            save(output / "edge_validation.json", validate_edges())
        # Controller and monitoring work stay off the timed CPU; each child explicitly selects CPU 0.
        if len(affinity) > 4:
            process.cpu_affinity(affinity[4:])
        for index, spec in enumerate(manifest["timing"]):
            path = output / "cases" / (spec["case"] + ".json")
            if args.resume and path.exists() and read(path)["status"] == "completed":
                continue
            state["stage"] = spec["case"]
            print(f"Timing {index+1}/{len(manifest['timing'])}: {spec['case']}", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "timing", "--index", str(index), "--output", str(output)]
            logpath = output / "logs" / (spec["case"] + ".log")
            with logpath.open("w", encoding="utf-8") as log:
                try:
                    result = subprocess.run(command, cwd=PAPER, stdout=log, stderr=subprocess.STDOUT,
                                            timeout=cfg["case_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    data = read(path) if path.exists() else dict(spec=spec,samples=[],checks=[],inputs=[],methods={})
                    data.update(status="timeout", completed_utc=utc())
                    save(path, data)
                    run["failures"].append(dict(case=spec["case"], status="timeout"))
                    aggregate(output)
                    save(output / "run.json", run)
                    continue
            if result.returncode:
                run["failures"].append(dict(case=spec["case"], exit_code=result.returncode))
                raise RuntimeError(f"Timing failed: {spec['case']}; see {logpath}")
            aggregate(output)
        state["stage"] = "memory"
        for index, spec in enumerate(manifest["memory"]):
            path = output / "memory_cases" / (spec["job"] + ".json")
            if args.resume and path.exists() and read(path)["status"] == "passed":
                continue
            if index % 5 == 0:
                print(f"Memory {index+1}/{len(manifest['memory'])}: {spec['case']} {spec['solver']}", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "memory", "--index", str(index), "--output", str(output)]
            with (output / "logs" / "memory.log").open("a", encoding="utf-8") as log:
                log.write(spec["job"] + "\n")
                log.flush()
                result = subprocess.run(command, cwd=PAPER, stdout=log, stderr=subprocess.STDOUT, timeout=cfg["memory_timeout_seconds"])
            if result.returncode:
                run["failures"].append(dict(case=spec["job"], exit_code=result.returncode))
                raise RuntimeError(f"Memory trial failed: {spec['job']}")
            if index % 10 == 0:
                aggregate(output)
        run.update(aggregate(output))
        verify_protected(output)
        run.update(status="completed" if not run["failures"] else "completed_with_exclusions", completed_utc=utc())
    except BaseException as exc:
        run["error"] = repr(exc)
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        process.cpu_affinity(affinity)
        if run["status"] == "running":
            run.update(status="incomplete", interrupted_utc=utc())
        save(output / "run.json", run)
    print(json.dumps(run, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("timing", "memory"))
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.worker:
        {"timing": timing_worker, "memory": memory_worker}[args.worker](args.output.resolve(), args.index)
    else:
        controller(args)


if __name__ == "__main__":
    main()
