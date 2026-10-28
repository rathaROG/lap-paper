"""Base suite: Tables 4-6 and Figures 3-5 (Appendix A1, A2, B1-B4).

Standalone port of the paper's `run_experiments.py`. It measures:

* validation: exhaustive enumeration on small binary-rational inputs, float32
  and float64, five limits, forbidden entries (all four native interfaces);
* single: 12 required and finite-limit cases x 7 solvers (Table 4, Figure 3);
* batch: 3 workloads, sequential loops and 1/4-worker batch APIs (Table 5,
  Figure 4);
* memory: high-water increment for one 256 x 4096 solve per fresh process,
  required and finite-limit (Table 6, Figure 5).

Seeds, generators, adapters, objectives and the timing loop are those of the
measured program; `--check-inputs` proves the regenerated inputs are
byte-identical to the paper's. Needs numpy, scipy, psutil, threadpoolctl,
lapx, laprust and lapcpp. About 1 minute with the defaults.

    python run.py --output results-base
    python run.py --check-inputs
    python run.py --output smoke --rounds 1 --block-seconds 0.001 --skip-memory
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
import re
import site
import subprocess
import sys
import sysconfig
import tempfile
import time

SUITE = "base"
INPUT_DIGESTS = {
    # sha256 of "case:input_sha256" lines, from the paper's saved measurement records
    "single": "605a4fddc780f4f7be52619395830e7906184bffaabbb8b0a137eae132605b95",
    "batch": "2d153f269137aaa93c655092c5f97f3274b36a75f683d263b58172125805a292",
}

# ---- BEGIN SHARED PAPER ADAPTERS ---------------------------------------------
# Identical in base/, formulations/, scaling/ and extended/ run.py so that each
# suite runs on its own; ../check_consistency.py verifies the copies match.
# Solver calls, cost transformations, objectives and the timing loop are those
# of the paper's measured programs (LAPX 0.12.0, laprust/lapcpp 1.1.0).

SOLVERS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "jvs", "jvc", "scipy")
LABELS = {"jvrust": "JVRUST", "jvrust_lapx": "JVRUST (LAPX lapjvr)", "jvhybrid": "JVHYBRID",
          "jvcpp": "JVCPP", "jvx": "JVX", "jvs": "JVS", "jvc": "JVC", "scipy": "SciPy",
          "scipy_square": "SciPy square", "scipy_dummies": "SciPy dummy cols",
          "scipy_capped": "SciPy capped"}
# JVRUST ships in laprust and as LAPX lapjvr; validation covers both, timing uses laprust.
NATIVE = ("jvrust", "jvrust_lapx", "jvhybrid", "jvcpp")
NATIVE_MODULES = ("laprust._lapjv", "lapcpp._lapjv", "lap._lapjv_rh", "lap._lapjv",
                  "lap._lapjvx", "lap._lapjvs", "lap._lapjvc")
np = lap = lapcpp = laprust = lsa = psutil = None


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
    """Square (n+m) expansion with L/2 rejection blocks: the classic dense formulation."""
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
    if name == "scipy_capped":
        # Capping precedes optimization; filtering an uncapped solve is wrong.
        rows, cols = lsa(np.minimum(cost, limit))
        keep = cost[rows, cols] < limit  # Reject equality ties consistently.
        return np.column_stack((rows[keep], cols[keep])).astype(np.int64, copy=False)
    if name == "scipy_dummies":
        transpose = cost.shape[0] > cost.shape[1]
        oriented = cost.T if transpose else cost
        r, c = oriented.shape
        work = np.empty((r, c + r), dtype=np.float64)
        work[:, :c] = oriented
        work[:, c:] = limit
        rows, cols = lsa(work)
        keep = cols < c
        rows, cols = rows[keep], cols[keep]
        return np.column_stack((cols, rows) if transpose else (rows, cols)).astype(np.int64, copy=False)
    if name == "scipy_square":
        name = "scipy"
    finite = math.isfinite(limit)
    work = augmented(cost, limit) if finite else cost
    if name == "jvs":
        pairs = lap.lapjvsa(work, extend_cost=True, return_cost=False, prefer_float32=False)
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
    """Selected costs plus L/2 per unmatched row and column (the square-expanded objective F)."""
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


def make_cost(spec, rng):
    """Generator of the base, formulation and scaling suites (uniform [0,1) by default)."""
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


def array_sha256(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def input_digest(rows):
    """Digest of ordered (case, input hash) rows, comparable with INPUT_DIGESTS."""
    return hashlib.sha256("\n".join(f"{case}:{h}" for case, h in rows).encode()).hexdigest()


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
    """Mean nanoseconds per call over `number` passes through `inputs`, GC disabled."""
    with no_gc():
        start = time.perf_counter_ns()
        for _ in range(number):
            for item in inputs:
                call(item)
        elapsed = time.perf_counter_ns() - start
    return elapsed / (number * len(inputs))


def calibrated_cycles(probe_ns, count, block_seconds, cap=100000):
    """Passes per timed block so that one block takes about `block_seconds`."""
    return max(1, min(cap, round(block_seconds * 1e9 / (probe_ns * count))))


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


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# Saved records never contain local absolute paths: known roots become placeholders
# (<benchmark>, <python>, <site-packages>, <tmp>, ~, <cwd>), any other absolute path
# becomes <external>/<name>. Hashes identify files.
REDACT_ROOT = (Path(__file__).resolve().parents[1], "<benchmark>")
# URLs are protected before path matching, including path-like query values.
_URL = re.compile(r"(?:https?|ftp)://[^\s\"'<>]+", re.IGNORECASE)
# A directory component may contain spaces; the unquoted final component stops
# at whitespace so surrounding prose is retained. Quoted/whole paths handle
# filenames with spaces separately.
_COMPONENT = r"[^\s\"'<>|:\\/]+(?:[ \t]+[^\s\"'<>|:\\/]+)*"
_WINDOWS_ABS = re.compile(r"(?:(?<![A-Za-z0-9])|(?<=/I)|(?<=/Fo)|(?<=/Fe)|(?<=/LIBPATH:))"
                          r"[A-Za-z]:[\\/](?:" + _COMPONENT + r"[\\/])*[^\s\"'<>|\\/]*", re.IGNORECASE)
_QUOTED_ABS = re.compile(r"(?P<quote>[\"'])(?P<path>(?:[A-Za-z]:[\\/]|\\\\|/)[^\"'\r\n]*)(?P=quote)")
_WHOLE_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/][^\"'<>|\r\n]*$")
_WHOLE_UNC_ABS = re.compile(r"^\\\\[^\\/\s]+[\\/][^\"'<>|\r\n]*$")
_WHOLE_POSIX_ABS = re.compile(r"^/(?:[^/\r\n]+/)+[^\r\n]*$")
_UNC_ABS = re.compile(r"(?<![\\\w])\\\\[^\\/\s\"'<>|]+[\\/](?:" + _COMPONENT + r"[\\/])*[^\s\"'<>|\\/]*")
_POSIX_ABS = re.compile(r"(?:(?<![\w:/.>~-])|(?<=-I)|(?<=-L))/(?:(?:" + _COMPONENT + r")/)+[^\s\"'<>/]*")
_REDACT_ROOTS = None


def _redaction_roots():
    global _REDACT_ROOTS
    if _REDACT_ROOTS is None:
        sites = list(getattr(site, "getsitepackages", lambda: [])()) + [site.getusersitepackages()]
        sites += [sysconfig.get_paths().get(key) for key in ("purelib", "platlib")]
        # Later entries win for identical paths: Windows lists sys.prefix as a site directory.
        candidates = [(os.getcwd(), "<cwd>"), (Path.home(), "~"), (tempfile.gettempdir(), "<tmp>"),
                      *((p, "<site-packages>") for p in sites if p),
                      *((p, "<python>") for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)), REDACT_ROOT]
        chosen = {}
        for path, placeholder in candidates:
            try:
                text = str(Path(path).resolve())
            except (OSError, ValueError):
                continue
            chosen[text.lower() if os.name == "nt" else text] = (text, placeholder)
        roots = []
        for text, placeholder in chosen.values():
            parts = [part for part in re.split(r"[\\/]+", text) if part]
            if len(parts) < 2:
                continue  # never redact a bare drive or "/"
            body = r"[\\/]+".join(re.escape(part) for part in parts)
            lead = "" if re.fullmatch(r"[A-Za-z]:", parts[0]) else r"[\\/]"
            roots.append((len(text), re.compile(lead + body + r"(?=$|[\\/\s\"'])",
                                                re.IGNORECASE if os.name == "nt" else 0), placeholder))
        _REDACT_ROOTS = [(pattern, placeholder) for _, pattern, placeholder in sorted(roots, key=lambda r: -r[0])]
    return _REDACT_ROOTS


def _external_path(value):
    parts = [part for part in re.split(r"[\\/]", value.rstrip("\\/")) if part]
    return "<external>/" + parts[-1] if parts else "<external>"


def _external(match):
    return _external_path(match.group(0))


def redact(value):
    """Replace local paths without changing numeric data or discarding records."""
    if isinstance(value, dict):
        items = [(redact(key) if isinstance(key, str) else key, key, item) for key, item in value.items()]
        counts = {}
        for key, _, _ in items:
            counts[key] = counts.get(key, 0) + 1
        result = {}
        for key, original, item in items:
            if counts[key] > 1 or key in result:
                # Distinct external paths may have the same basename. Keep every
                # record and give its key a stable identity without exposing roots.
                identity = hashlib.sha256(str(original).encode("utf-8")).hexdigest()[:12]
                key = f"{key}#<id:{identity}>"
                while key in result:
                    key += "_"
            result[key] = redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if not isinstance(value, str) or not value:
        return value
    urls = []

    def protect_url(match):
        urls.append(match.group(0))
        return f"<url:{len(urls) - 1}>"

    value = _URL.sub(protect_url, value)
    for pattern, placeholder in _redaction_roots():
        value = pattern.sub(placeholder, value)
    value = _QUOTED_ABS.sub(lambda match: match.group("quote") + _external_path(match.group("path")) + match.group("quote"), value)
    if any(pattern.fullmatch(value) for pattern in (_WHOLE_WINDOWS_ABS, _WHOLE_UNC_ABS, _WHOLE_POSIX_ABS)):
        value = _external_path(value)
    else:
        value = _WINDOWS_ABS.sub(_external, value)
        value = _UNC_ABS.sub(_external, value)
        value = _POSIX_ABS.sub(_external, value)
    for index, url in enumerate(urls):
        value = value.replace(f"<url:{index}>", url)
    return value

def write_json(path, value):
    Path(path).write_text(json.dumps(redact(value), indent=2) + "\n", encoding="utf-8")


def fresh_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Choose a new output directory to preserve previous measurements: {redact(str(path))}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def select_cpus(single=None, group=None):
    """Paper defaults: single solves on the first allowed CPU, batches on the first four.

    Override with --cpu/--batch-cpus or the LAP_BENCH_SINGLE_CPU and
    LAP_BENCH_CPUS (comma-separated) environment variables of the Linux kit.
    On hybrid CPUs choose performance cores.
    """
    available = psutil.Process().cpu_affinity()
    if group is None and os.environ.get("LAP_BENCH_CPUS"):
        group = [int(v) for v in os.environ["LAP_BENCH_CPUS"].split(",")]
    group = list(group) if group else available[:4]
    if single is None:
        single = int(os.environ.get("LAP_BENCH_SINGLE_CPU", group[0]))
    if not set(group) | {single} <= set(available):
        raise SystemExit(f"Requested CPUs {sorted(set(group) | {single})} are not all in the allowed set {available}")
    return single, group


def snapshot_memory(process):
    """Current RSS and the process high-water counter (Windows peak working set, Linux ru_maxrss)."""
    info = process.memory_info()
    if sys.platform == "win32":
        return {"rss": int(info.rss), "peak": int(info.peak_wset), "counter": "psutil.memory_info.peak_wset"}
    if sys.platform.startswith("linux"):
        import resource
        return {"rss": int(info.rss), "peak": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                "counter": "getrusage(RUSAGE_SELF).ru_maxrss"}
    raise RuntimeError("Peak resident memory is implemented for Windows and Linux only")


def environment(extra=None):
    import scipy
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pools = None
    native = {}
    for name in NATIVE_MODULES:
        try:
            path = Path(importlib.import_module(name).__file__).resolve()
            native[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception as error:
            native[name] = {"error": repr(error)}
    cpu_name = platform.processor()
    if os.name == "nt":
        with contextlib.suppress(Exception):
            cpu_name = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True).strip()
    return redact({
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suite": SUITE, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cpu": cpu_name, "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "packages": {name: getattr(module, "__version__", None)
                     for name, module in (("lapx", lap), ("lapcpp", lapcpp), ("laprust", laprust))},
        "threadpools": pools, "native_modules": native, **(extra or {}),
    })
# ---- END SHARED PAPER ADAPTERS -----------------------------------------------


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
                                    report["maximum_absolute_objective_error"], error)
                                assert error == 0.0, (expected, actual)
                            except ValueError as exc:
                                if math.isinf(expected):
                                    report["infeasible_calls"] += 1
                                else:
                                    report["failures"].append({"solver": name, "shape": [n, m],
                                                               "limit": str(limit), "error": str(exc)})
                            except Exception as exc:
                                report["failures"].append({"solver": name, "shape": [n, m],
                                                           "limit": str(limit), "error": repr(exc)})
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
        {"case": "negative128x160", "n": 128, "m": 160, "limit": -0.25, "forbidden": 0.2, "signed": True},
        {"case": "optional_square128", "n": 128, "m": 128, "limit": 0.5, "forbidden": 0.5},
    ]


def single_inputs(seed, instances):
    rng = np.random.default_rng(seed + 1)
    for spec in case_specs():
        yield spec, [make_cost(spec, rng) for _ in range(instances)]


def single_experiment(options):
    order_rng = random.Random(options.seed + 2)
    rows, checks = [], []
    for spec, inputs in single_inputs(options.seed, options.instances):
        reference = [objective(a, solve("scipy", a, spec["limit"]), spec["limit"]) for a in inputs]
        calls, numbers = {}, {}
        for name in SOLVERS:
            calls[name] = functools.partial(solve, name, limit=spec["limit"])
            for index, a in enumerate(inputs):
                actual = check_pairs(a, calls[name](a), spec["limit"])
                if not np.isclose(actual, reference[index], rtol=1e-10, atol=1e-10):
                    raise AssertionError((spec["case"], name, index, actual, reference[index]))
            numbers[name] = calibrated_cycles(timed_block(calls[name], inputs, 1), len(inputs), options.block_seconds)
        for round_index in range(options.rounds):
            order = list(SOLVERS)
            order_rng.shuffle(order)
            for position, name in enumerate(order):
                elapsed = timed_block(calls[name], inputs, numbers[name])
                rows.append({"case": spec["case"], "solver": name, "n": spec["n"], "m": spec["m"],
                             "limit": str(spec["limit"]), "round": round_index, "position": position,
                             "instances": len(inputs), "cycles": numbers[name], "nanoseconds": elapsed})
        checks.append({"case": spec["case"], "instances": len(inputs), "solvers": len(SOLVERS),
                       "objective_checks_passed": True, "input_sha256": [array_sha256(a) for a in inputs]})
        print("Single", spec["case"], flush=True)
    return rows, checks


BATCH_FAMILIES = ("jvrust", "jvhybrid", "jvcpp", "jvx")


def batch_specs():
    return [
        {"case": "batch64x64x80", "b": 64, "n": 64, "m": 80, "limit": math.inf},
        {"case": "batch16x128x160_optional", "b": 16, "n": 128, "m": 160, "limit": 0.5, "forbidden": 0.5},
        # Many tiny problems, as in per-frame tracking, expose per-problem overhead.
        {"case": "batch1024x8x10", "b": 1024, "n": 8, "m": 10, "limit": math.inf},
    ]


def batch_inputs(seed):
    rng = np.random.default_rng(seed + 3)
    for spec in batch_specs():
        yield spec, np.stack([make_cost(spec, rng) for _ in range(spec["b"])])


def batch_call(name, costs, limit):
    if name.endswith("_loop"):
        solver = name.removesuffix("_loop")
        return [solve(solver, a, limit) for a in costs]
    family, suffix = name.rsplit("_batch_t", 1)
    threads = int(suffix)
    if family in NATIVE:
        return native_function(family, batch=True)(
            costs, extend_cost=True, cost_limit=limit, return_cost=False,
            n_threads=threads, precision="float64", output="pairs")
    if family == "jvx":
        return lap.lapjvxa_batch(costs, extend_cost=True, cost_limit=limit, return_cost=False, n_threads=threads)
    raise ValueError(name)


def batch_names(batch_threads):
    names = ["jvrust_loop", "scipy_loop"]
    for family in BATCH_FAMILIES:
        names += [f"{family}_batch_t1", f"{family}_batch_t{batch_threads}"]
    return list(dict.fromkeys(names))


def check_batch(costs, actual, expected, limit):
    if not len(actual) == len(costs) == len(expected):
        raise AssertionError(f"batch length mismatch: {len(costs)}, {len(actual)}, {len(expected)}")
    checked = 0
    for a, pairs, reference in zip(costs, actual, expected, strict=True):
        value = check_pairs(a, pairs.astype(np.int64, copy=False), limit)
        assert np.isclose(value, reference, rtol=1e-10, atol=1e-10), (checked, value, reference)
        checked += 1
    return checked


def batch_experiment(options, batch_threads):
    order_rng = random.Random(options.seed + 4)
    names = batch_names(batch_threads)
    rows, checks = [], []
    for spec, costs in batch_inputs(options.seed):
        expected = [objective(a, solve("scipy", a, spec["limit"]), spec["limit"]) for a in costs]
        calls, numbers, checked = {}, {}, {}
        for name in names:
            calls[name] = functools.partial(batch_call, name, limit=spec["limit"])
            checked[name] = check_batch(costs, calls[name](costs), expected, spec["limit"])
            numbers[name] = calibrated_cycles(timed_block(calls[name], [costs], 1), 1, options.block_seconds, cap=10000)
        for round_index in range(options.rounds):
            order = list(names)
            order_rng.shuffle(order)
            for position, name in enumerate(order):
                elapsed = timed_block(calls[name], [costs], numbers[name])
                rows.append({"case": spec["case"], "solver": name, "b": spec["b"], "n": spec["n"],
                             "m": spec["m"], "limit": str(spec["limit"]), "round": round_index,
                             "position": position, "cycles": numbers[name], "nanoseconds": elapsed})
        checks.append({"case": spec["case"], "solvers": len(names), "problems": len(costs),
                       "checked_outputs": checked, "input_sha256": hashlib.sha256(costs.tobytes()).hexdigest(),
                       "objective_checks_passed": True})
        print("Batch", spec["case"], flush=True)
    return rows, checks


MEMORY_SOLVERS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "scipy")


def memory_child(options):
    """One fresh process: warm up, prepare the resident input, then measure one solve."""
    process = psutil.Process()
    process.cpu_affinity([options.cpu])
    limit = 0.5 if options.memory_case == "optional" else math.inf
    solve(options.memory_solver, np.ones((4, 4)), limit)
    cost = np.random.default_rng(options.seed).random((256, 4096))
    gc.collect()
    before = snapshot_memory(process)
    result = solve(options.memory_solver, cost, limit)
    after = snapshot_memory(process)
    check_pairs(cost, result, limit)
    print(json.dumps({"case": options.memory_case, "solver": options.memory_solver, "n": 256, "m": 4096,
                      "limit": str(limit), "counter": before["counter"],
                      "rss_before": before["rss"], "rss_after": after["rss"],
                      "peak_before": before["peak"], "peak_after": after["peak"],
                      "additional_peak": max(0, after["peak"] - before["peak"])}), flush=True)


def memory_experiment(options, cpu):
    reports = []
    for case in ("required", "optional"):
        for name in MEMORY_SOLVERS:
            command = [sys.executable, str(Path(__file__).resolve()), "--memory-solver", name,
                       "--memory-case", case, "--seed", str(options.seed), "--cpu", str(cpu)]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            reports.append(json.loads(result.stdout.strip().splitlines()[-1]))
            print("Memory", case, name, flush=True)
    return reports


def check_inputs(options):
    single = [(spec["case"], array_sha256(a)) for spec, inputs in single_inputs(options.seed, options.instances)
              for a in inputs]
    batch = [(spec["case"], hashlib.sha256(costs.tobytes()).hexdigest()) for spec, costs in batch_inputs(options.seed)]
    ok = True
    for name, rows in (("single", single), ("batch", batch)):
        same = input_digest(rows) == INPUT_DIGESTS[name]
        ok &= same
        print(f"{name}: {len(rows)} inputs {'identical to the paper' if same else 'DIFFER from the paper'}")
    if (options.seed, options.instances) != (20261021, 4):
        print("(non-default --seed/--instances cannot match the paper digests)")
    return 0 if ok else 1


def print_tables(single, batch, memory):
    if single:
        table = {(r["case"], r["solver"]): r["median_ns"] / 1000 for r in summary(single, ["case", "solver"])}
        print("\nTable 4: median microseconds per matrix")
        print(f"{'case':20s}" + "".join(f"{LABELS[s]:>10s}" for s in SOLVERS))
        for spec in case_specs():
            print(f"{spec['case']:20s}" + "".join(f"{table[(spec['case'], s)]:10.1f}" for s in SOLVERS))
    if batch:
        table = {(r["case"], r["solver"]): r["median_ns"] / 1e6 for r in summary(batch, ["case", "solver"])}
        names = sorted({r["solver"] for r in batch}, key=lambda s: (s.split("_")[0], s))
        print("\nTable 5: median milliseconds per complete batch")
        for spec in batch_specs():
            print(f"  {spec['case']}: " + ", ".join(f"{n} {table[(spec['case'], n)]:.3f}" for n in names))
    if memory:
        print("\nTable 6: additional peak for 256 x 4096 (MiB)")
        for r in memory:
            print(f"  {r['case']:9s} {LABELS[r['solver']]:10s} {r['additional_peak'] / 2**20:8.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--block-seconds", type=float, default=0.03)
    parser.add_argument("--cpu", type=int, help="logical CPU for single solves and memory children")
    parser.add_argument("--batch-cpus", type=int, nargs=4, help="four logical CPUs for batch timing")
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--check-inputs", action="store_true", help="regenerate inputs and compare with the paper")
    parser.add_argument("--memory-solver", choices=SOLVERS, help=argparse.SUPPRESS)
    parser.add_argument("--memory-case", choices=("required", "optional"), help=argparse.SUPPRESS)
    options = parser.parse_args()
    initialize()
    if options.memory_solver:
        return memory_child(options)
    if options.check_inputs:
        return check_inputs(options)
    if options.output is None:
        parser.error("--output is required")
    output = fresh_output(options.output)
    from threadpoolctl import threadpool_limits
    process = psutil.Process()
    original = process.cpu_affinity()
    single_cpu, batch_cpus = select_cpus(options.cpu, options.batch_cpus)
    memory = []
    try:
        with threadpool_limits(limits=1):
            process.cpu_affinity([single_cpu])
            validation = validate(options.seed)
            write_json(output / "validation.json", validation)
            if validation["failures"]:
                raise AssertionError(validation["failures"][:5])
            print("Validation", validation["solver_calls"], "calls passed", flush=True)
            single, single_checks = single_experiment(options)
            csv_write(output / "single_samples.csv", single)
            csv_write(output / "single_summary.csv", summary(single, ["case", "solver"]))
            process.cpu_affinity(batch_cpus)
            batch, batch_checks = batch_experiment(options, len(batch_cpus))
            csv_write(output / "batch_samples.csv", batch)
            csv_write(output / "batch_summary.csv", summary(batch, ["case", "solver"]))
            write_json(output / "benchmark_checks.json", {"single": single_checks, "batch": batch_checks})
            write_json(output / "environment.json", environment({
                "single_affinity": [single_cpu], "batch_affinity": batch_cpus, "seed": options.seed,
                "rounds": options.rounds, "instances_per_single_case": options.instances,
                "target_block_seconds": options.block_seconds}))
        if not options.skip_memory:
            memory = memory_experiment(options, single_cpu)
            write_json(output / "memory.json", memory)
    finally:
        process.cpu_affinity(original)
    print_tables(single, batch, memory)
    print("\nResults saved:", redact(str(output)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
