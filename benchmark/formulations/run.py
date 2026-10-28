"""Formulation suite: Table 7 (Appendix A3, B5, B6).

Standalone port of the paper's `compare_optional.py`. Finite-limit (implicit
rejection) solves by JVRUST and JVCPP against three equivalent SciPy
formulations: square expansion, dummy columns and capped original-dimension
costs. Ten cases: the five finite-limit base cases (same inputs as the base
suite) plus five additional rejection-heavy and congested cases.

Validation: exhaustive enumeration on a binary-rational grid with five limits,
plus all-forbidden and all-equal boundary cases. `--check-inputs` proves the
regenerated inputs are byte-identical to the paper's. Under a minute.

    python run.py --output results-formulations
    python run.py --check-inputs
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

SUITE = "formulations"
INPUT_DIGESTS = {
    "formulations": "db48a7d60d31cf6f6a8151cc49c211fb4ecb009e68c94cfb6ebc301913f2c095",
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

METHODS = ("jvrust", "jvcpp", "scipy_square", "scipy_dummies", "scipy_capped")


def base_case_specs():
    """The base suite's 12 cases, consumed in order to reproduce their inputs."""
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


def datasets(seed, instances):
    # Consume every original case in the original order to reproduce its inputs.
    rng = np.random.default_rng(seed + 1)
    for spec in base_case_specs():
        inputs = [make_cost(spec, rng) for _ in range(instances)]
        if math.isfinite(spec["limit"]):
            yield dict(spec, source="original"), inputs
    rng = np.random.default_rng(seed + 101)
    extra = [
        {"case": "reject128x160_L005", "n": 128, "m": 160, "limit": .005, "forbidden": .5},
        {"case": "reject128x160_L001", "n": 128, "m": 160, "limit": .001, "forbidden": .5},
        {"case": "reject_tall160x128", "n": 160, "m": 128, "limit": .005, "forbidden": .5},
        {"case": "optional_wide64x512", "n": 64, "m": 512, "limit": .005, "forbidden": .5},
        {"case": "optional_blocks256x320", "n": 256, "m": 320, "limit": .5, "blocks": True},
    ]
    for spec in extra:
        yield dict(spec, source="additional"), [make_cost(spec, rng) for _ in range(instances)]


def validate(seed):
    rng = np.random.default_rng(seed)
    calls = instances = 0
    maximum_error = 0.0
    for n in range(6):
        for m in range(7):
            for _ in range(10):
                cost = rng.integers(-16, 17, size=(n, m)).astype(np.float64) / 8
                cost[rng.random((n, m)) < .2] = np.inf
                for limit in (-1.0, 0.0, .125, 1.0, 2.0):
                    expected = brute_objective(cost, limit)
                    instances += 1
                    for method in METHODS:
                        actual = check_pairs(cost, solve(method, cost, limit), limit)
                        maximum_error = max(maximum_error, abs(actual - expected))
                        assert actual == expected, (method, n, m, limit, actual, expected)
                        calls += 1
    # Explicit boundary cases, including all-forbidden and all-equality inputs.
    targeted = [
        (np.full((3, 5), np.inf), .5),
        (np.full((5, 3), .5), .5),
        (np.array([[0., .9], [.9, np.inf]]), 1.),
        (np.array([[-2., -1.], [-1., np.inf]]), -1.),
    ]
    for cost, limit in targeted:
        expected = brute_objective(cost, limit)
        for method in METHODS:
            actual = check_pairs(cost, solve(method, cost, limit), limit)
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return {"status": "passed", "seed": seed, "grid_instances": instances, "grid_solver_calls": calls,
            "targeted_solver_calls": len(targeted) * len(METHODS),
            "maximum_absolute_grid_objective_error": maximum_error,
            "scope": "float64, binary-rational grid; ties compare objectives, not pairs"}


def check_inputs(options):
    rows = [(spec["case"], array_sha256(a)) for spec, inputs in datasets(options.seed, options.instances)
            for a in inputs]
    same = input_digest(rows) == INPUT_DIGESTS["formulations"]
    print(f"formulations: {len(rows)} inputs {'identical to the paper' if same else 'DIFFER from the paper'}")
    return 0 if same else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--block-seconds", type=float, default=.03)
    parser.add_argument("--cpu", type=int, help="logical CPU for timing (default: first allowed)")
    parser.add_argument("--check-inputs", action="store_true", help="regenerate inputs and compare with the paper")
    options = parser.parse_args()
    initialize()
    if options.check_inputs:
        return check_inputs(options)
    if options.output is None:
        parser.error("--output is required")
    output = fresh_output(options.output)
    from threadpoolctl import threadpool_limits
    process = psutil.Process()
    original = process.cpu_affinity()
    cpu, _ = select_cpus(options.cpu)
    samples, checks, descriptions = [], [], []
    order_rng = random.Random(options.seed + 102)
    try:
        process.cpu_affinity([cpu])
        with threadpool_limits(limits=1):
            metadata = environment({"affinity": [cpu], "seed": options.seed, "rounds": options.rounds,
                                    "instances": options.instances, "target_block_seconds": options.block_seconds,
                                    "methods": METHODS})
            validation = validate(options.seed + 100)
            print("Independent optional validation passed:", validation["grid_solver_calls"], flush=True)
            for spec, inputs in datasets(options.seed, options.instances):
                limit = spec["limit"]
                reference = [objective(a, solve("scipy_square", a, limit), limit) for a in inputs]
                calls, cycles = {}, {}
                for method in METHODS:
                    calls[method] = functools.partial(solve, method, limit=limit)
                    for index, cost in enumerate(inputs):
                        pairs = calls[method](cost)
                        actual = check_pairs(cost, pairs, limit)
                        assert np.isclose(actual, reference[index], rtol=1e-10, atol=1e-10)
                        checks.append({"case": spec["case"], "solver": method, "instance": index,
                                       "pairs": len(pairs), "objective": actual,
                                       "reference_objective": reference[index]})
                    cycles[method] = calibrated_cycles(timed_block(calls[method], inputs, 1), len(inputs),
                                                       options.block_seconds)
                for round_index in range(options.rounds):
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    for position, method in enumerate(order):
                        elapsed = timed_block(calls[method], inputs, cycles[method])
                        samples.append({"case": spec["case"], "solver": method, "round": round_index,
                                        "position": position, "instances": len(inputs),
                                        "cycles": cycles[method], "nanoseconds": elapsed})
                counts = [r["pairs"] for r in checks if r["case"] == spec["case"] and r["solver"] == "jvrust"]
                mean_pairs = sum(counts) / len(counts)
                descriptions.append({**spec, "instances": len(inputs), "mean_pairs": mean_pairs,
                                     "mean_unmatched": min(spec["n"], spec["m"]) - mean_pairs,
                                     "input_sha256": [array_sha256(a) for a in inputs]})
                print(spec["case"], "mean pairs", mean_pairs, flush=True)
    finally:
        process.cpu_affinity(original)
    summaries = summary(samples, ("case", "solver"))
    csv_write(output / "samples.csv", samples)
    csv_write(output / "summary.csv", summaries)
    for name, value in (("environment", metadata), ("validation", validation),
                        ("checks", checks), ("datasets", descriptions)):
        write_json(output / f"{name}.json", value)
    table = {(r["case"], r["solver"]): r["median_ns"] / 1000 for r in summaries}
    print("\nTable 7: median microseconds per matrix; Unmatched = mean smaller-side unmatched rows")
    print(f"{'case':24s}" + "".join(f"{LABELS[m]:>17s}" for m in METHODS) + f"{'Unmatched':>11s}")
    for spec in descriptions:
        print(f"{spec['case']:24s}" + "".join(f"{table[(spec['case'], m)]:17.1f}" for m in METHODS)
              + f"{spec['mean_unmatched']:11.2f}")
    print("\nResults saved:", redact(str(output)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
