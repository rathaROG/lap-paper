"""Extended suite: Table 8 and Figures 9-15 (plus Figures A2 and B3 inputs).

Standalone port of the paper's `run_extended_experiments.py` (portable Linux
kit variant). 132 timing cases and 405 fresh-process memory trials:

* large squares 1000..5000 and large rectangles to 4000 x 5000, both
  orientations (Figures 9, 10); extreme 64 x 4096..16384 and transposes
  (Figure 11); required and finite-limit with L = 2/max(N, M);
* structure sweeps at 256 x 320 and 512 x 640: congestion fractions, forbidden
  probabilities, quantized ties, signed costs (Figures 12, 13);
* rejection sweeps over eleven limits, uniform and congested (Figure 14);
* memory: 256 x {256..4096} in both orientations, five fresh processes per
  condition (Figure 15).

Required methods: JVRUST, JVHYBRID, JVCPP, SciPy. Finite-limit methods:
JVRUST, JVCPP, capped SciPy, dummy-column SciPy. Each timing case runs in a
fresh process; budgets are 240 s per method/case, 1200 s per case and 120 s
per memory trial. Interrupted runs continue with --resume. `--check-inputs`
proves all regenerated inputs are byte-identical to the paper's. About 11
minutes on the paper's machine.

    python run.py --output results-extended
    python run.py --output results-extended --resume
    python run.py --output smoke-extended --smoke
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
import threading
import time

SUITE = "extended"
INPUT_DIGESTS = {
    "timing": "33a83919da40b5503bcfe61baa9def3859690ceb7f0db7c0f1d9d32f179402c8",
    "memory": "5ab6c267f46e4627130aa97bf51ece7c63db72c96a424b1bd257883b36ea6842",
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

REQUIRED = ("jvrust", "jvhybrid", "jvcpp", "scipy")
OPTIONAL = ("jvrust", "jvcpp", "scipy_capped", "scipy_dummies")
MEMORY_REQUIRED = ("jvrust", "jvcpp", "scipy", "jvx")
MEMORY_OPTIONAL = ("jvrust", "jvcpp", "scipy_capped", "scipy_dummies", "scipy_square")
SEED = 20261024


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(redact(data), indent=2, allow_nan=False) + "\n", encoding="utf-8")
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


def methods(spec):
    return REQUIRED if spec["limit"] is None else OPTIONAL


def plan(smoke=False):
    cases = []

    def add(family, n, m, limit, key, **extras):
        regime = "required" if limit is None else "optional"
        item = dict(family=family, n=n, m=m, regime=regime, limit=limit, input_key=key, distribution="uniform")
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
                add("forbidden", n, m, limit, f"forbidden_{n}x{m}", forbidden=forbidden, planted_feasible=True)
        for levels in ((2,) if smoke else (1, 2, 8, 256)):
            for limit in (None, .25):
                add("ties", n, m, limit, f"ties_{n}x{m}", levels=levels)
    n, m = (24, 32) if smoke else (256, 320)
    for rows, cols in ((n, m), (m, n)):
        for limit in (None, -.25):
            add("signed", rows, cols, limit, f"signed_{n}x{m}", signed=True, forbidden=.5, planted_feasible=True)
    for n, m in (((24, 32),) if smoke else ((128, 160), (256, 320))):
        for distribution in ("uniform", "blocks"):
            for limit in ((.001, .5) if smoke else (0., .0005, .001, .002, .005, .01, .02, .05, .1, .5, 4.)):
                extra = dict(forbidden=.5) if distribution == "uniform" else dict(row_fraction=.75, column_fraction=.25)
                add("rejection", n, m, limit, f"rejection_{distribution}_{n}x{m}", distribution=distribution, **extra)
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
    canonical_hash = array_sha256(cost)
    if spec["n"] > spec["m"]:
        cost = np.ascontiguousarray(cost.T)
    return cost, canonical_hash


def memory_input(spec, seed):
    rng = np.random.default_rng([seed, 30000, spec["n"], spec["m"], spec["trial"]])
    return rng.random((spec["n"], spec["m"]))


def checked_objective(cost, pairs, limit, expected):
    value = check_pairs(cost, pairs, limit)
    if not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError((value, expected))
    return value


def validate_edges():
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
        expected = brute_objective(cost, limit)
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
    spec, cfg = manifest["timing"][index], manifest["configuration"]
    initialize()
    psutil.Process().cpu_affinity([cfg["cpu"]])
    from threadpoolctl import threadpool_limits
    path = output / "cases" / (spec["case"] + ".json")
    report = dict(spec=spec, status="running", started_utc=utc(), samples=[], checks=[], methods={}, inputs=[])
    save(path, report)
    with threadpool_limits(limits=1):
        inputs = []
        for i in range(cfg["instances"]):
            a, canonical = make_input(spec, cfg["seed"], i)
            inputs.append(a)
            report["inputs"].append(dict(instance=i, sha256=array_sha256(a), canonical_sha256=canonical,
                                         forbidden_fraction=float(np.isinf(a).mean())))
        limit = limit_of(spec)
        reference_method = "scipy" if spec["limit"] is None else "scipy_capped"
        expected = [check_pairs(a, solve(reference_method, a, limit), limit) for a in inputs]
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
            probe = timed_block(calls[method], inputs, 1)
            cycles[method] = calibrated_cycles(probe, len(inputs), cfg["block_seconds"])
            projected = probe * len(inputs) * cycles[method] * cfg["rounds"] / 1e9
            status = "budget_excluded" if projected > cfg["method_budget_seconds"] else "ready"
            report["methods"][method] = dict(status=status, probe_ns=probe, cycles=cycles[method],
                                             projected_timing_seconds=projected,
                                             validation_and_probe_seconds=time.perf_counter() - start,
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
                elapsed = timed_block(calls[method], inputs, cycles[method])
                info["measured_seconds"] += time.perf_counter() - start
                report["samples"].append(dict(case=spec["case"], solver=method, round=round_index,
                                              position=position, instances=len(inputs), cycles=cycles[method],
                                              nanoseconds=elapsed))
                if info["measured_seconds"] > cfg["method_budget_seconds"] and round_index + 1 < cfg["rounds"]:
                    info["status"] = "budget_exceeded"
                elif round_index + 1 == cfg["rounds"]:
                    info["status"] = "completed"
                save(path, report)
        assert [array_sha256(a) for a in inputs] == [row["sha256"] for row in report["inputs"]], \
            "Solver changed input costs"
        report.update(status="completed", completed_utc=utc(), active_method=None)
        save(path, report)


def memory_worker(output, index):
    manifest = read(output / "plan.json")
    spec, cfg = manifest["memory"][index], manifest["configuration"]
    initialize()
    process = psutil.Process()
    process.cpu_affinity([cfg["cpu"]])
    from threadpoolctl import threadpool_limits
    limit = limit_of(spec)
    with threadpool_limits(limits=1):
        # Warm imports/workspace setup before preparing the resident input.
        solve(spec["solver"], np.ones((4, 4)), limit)
        cost = memory_input(spec, cfg["seed"])
        input_hash = array_sha256(cost)  # memoryview: no extra dense copy before measuring.
        gc.collect()
        before = snapshot_memory(process)
        pairs = solve(spec["solver"], cost, limit)
        after = snapshot_memory(process)
        # The reference and all checking run after capturing the measured peak.
        reference = "scipy" if spec["limit"] is None else "scipy_capped"
        expected = check_pairs(cost, solve(reference, cost, limit), limit)
        value = checked_objective(cost, pairs, limit, expected)
    r, c = sorted((spec["n"], spec["m"]))
    buffers = {"jvrust": 0, "jvcpp": 0, "scipy": 0, "jvx": c * c * 8 if r != c else 0,
               "scipy_capped": r * c * 8, "scipy_dummies": r * (c + r) * 8, "scipy_square": (r + c) ** 2 * 8}
    save(output / "memory_cases" / (spec["job"] + ".json"), dict(
        spec, status="passed", pid=os.getpid(), python_executable=sys.executable, input_sha256=input_hash,
        input_bytes=cost.nbytes, memory_counter=before["counter"], rss_before=before["rss"],
        rss_after=after["rss"], peak_before=before["peak"], peak_after=after["peak"],
        additional_peak=max(0, after["peak"] - before["peak"]),
        baseline_peak_headroom=max(0, before["peak"] - before["rss"]),
        transformed_dense_buffer_bytes=buffers[spec["solver"]], objective=value,
        reference_objective=expected, completed_utc=utc()))


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
        csv_write(output / "samples.csv", samples)
    if valid:
        csv_write(output / "summary.csv", summary(valid, ("case", "solver")))
    save(output / "datasets.json", datasets)
    save(output / "checks.json", checks)
    save(output / "method_status.json", statuses)
    memory = [read(p) for p in sorted((output / "memory_cases").glob("*.json"))]
    save(output / "memory_samples.json", memory)
    memory_summary = []
    for case, solver in sorted({(row["case"], row["solver"]) for row in memory}):
        group = [row for row in memory if (row["case"], row["solver"]) == (case, solver)]
        values = np.array([row["additional_peak"] for row in group])
        first = group[0]
        memory_summary.append(dict(case=case, solver=solver, n=first["n"], m=first["m"], regime=first["regime"],
            trials=len(group), median_bytes=float(np.median(values)), q25_bytes=float(np.percentile(values, 25)),
            q75_bytes=float(np.percentile(values, 75)), min_bytes=int(values.min()), max_bytes=int(values.max()),
            input_bytes=first["input_bytes"], transformed_dense_buffer_bytes=first["transformed_dense_buffer_bytes"]))
    if memory_summary:
        csv_write(output / "memory_summary.csv", memory_summary)
    return dict(timing_cases=len(datasets), timing_samples=len(samples),
                completed_method_settings=len({(r["case"], r["solver"]) for r in valid}),
                memory_trials=len(memory), objective_checks=len(checks),
                excluded_method_settings=sum(r["status"] not in ("completed", "ready", "running") for r in statuses))


def run_logged(command, log, timeout):
    """Keep worker tracebacks portable, including partial output on timeout."""
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log.write(redact(output))
        log.flush()
        raise
    log.write(redact(result.stdout or ""))
    log.flush()
    return result


def controller(args):
    initialize()
    output = args.output.resolve()
    process = psutil.Process()
    affinity = process.cpu_affinity()
    cpu, group = select_cpus(args.cpu)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.resume:
        manifest = read(output / "plan.json")
        assert read(output / "run.json")["source_sha256"] == source_hash, \
            "This script changed since the run started; use a new output directory."
        assert manifest["configuration"]["smoke"] == args.smoke
    else:
        output = fresh_output(output)
        manifest = plan(args.smoke)
        manifest["configuration"] = dict(seed=SEED, instances=4, rounds=9, block_seconds=.03, cpu=cpu,
            smoke=args.smoke, method_budget_seconds=240, case_timeout_seconds=1200, memory_timeout_seconds=120,
            precision="float64", objective_rtol=1e-10, objective_atol=1e-10, controller_excluded_cpus=group)
        manifest["protocol"] = {
            "required_methods": REQUIRED, "optional_methods": OPTIONAL,
            "pairing": "Four deterministic input matrices per timing case, shared by all methods. Input arrays are "
                       "reused across threshold/structure levels and exact transposes across orientations.",
            "large_optional_limit": "L=2/max(N,M), chosen before timing to retain rejection-active inputs as size "
                                    "grows; actual cardinalities are recorded.",
            "timing": "Validation, data generation and calibration are outside timing. Nine shuffled rounds include "
                      "adapters and output construction; each method uses a calibrated block targeting 30 ms.",
            "budgets": "Methods projected to need more than 240 timing seconds are excluded before rounds; methods "
                       "exceeding it mid-run keep partial samples but get no summary. Whole-case wall timeout is "
                       "1200 seconds. All omissions stay explicit in method_status.json and logs.",
            "memory": "Five fresh processes per condition, single CPU and one library thread. Resident float64 input "
                      "prepared before the baseline. High-water increments, baseline headroom and exact transformed-"
                      "buffer sizes are recorded separately.",
        }
        save(output / "plan.json", manifest)
    cfg = manifest["configuration"]
    if cfg["cpu"] != cpu and args.resume:
        cpu = cfg["cpu"]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        save(output / "environment.json", environment({"configuration": cfg}))
    for name in ("cases", "memory_cases", "logs", "sources"):
        (output / name).mkdir(exist_ok=True)
    (output / "sources" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    run = dict(status="running", started_utc=utc(), python_executable=sys.executable, source_sha256=source_hash,
               planned_timing_cases=len(manifest["timing"]), planned_memory_trials=len(manifest["memory"]),
               failures=[])
    save(output / "run.json", run)
    print(f"Plan: {len(manifest['timing'])} timing cases, {len(manifest['memory'])} fresh-process memory trials",
          flush=True)
    stop, state = threading.Event(), {"stage": "validation"}
    excluded = set(cfg.get("controller_excluded_cpus", [cfg["cpu"]]))

    def monitor():
        path = output / "cpu_load.csv"
        append = path.exists()
        with path.open("a", newline="", encoding="utf-8") as stream:
            fields = ["utc", "stage", "mean_cpu_percent", "mean_outside_benchmark_cpus_percent"] + \
                     [f"cpu{i}_percent" for i in range(psutil.cpu_count())]
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not append:
                writer.writeheader()
            while not stop.is_set():
                values = psutil.cpu_percent(interval=1, percpu=True)
                outside = [v for i, v in enumerate(values) if i not in excluded] or values
                writer.writerow(dict(utc=utc(), stage=state["stage"], mean_cpu_percent=sum(values) / len(values),
                                     mean_outside_benchmark_cpus_percent=sum(outside) / len(outside),
                                     **{f"cpu{i}_percent": v for i, v in enumerate(values)}))
                stream.flush()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        process.cpu_affinity([cfg["cpu"]])
        with threadpool_limits(limits=1):
            save(output / "edge_validation.json", validate_edges())
        # Controller and monitor stay off the benchmark CPUs; each worker pins its declared CPU.
        process.cpu_affinity([c for c in affinity if c not in excluded] or affinity)
        for index, spec in enumerate(manifest["timing"]):
            path = output / "cases" / (spec["case"] + ".json")
            if args.resume and path.exists() and read(path)["status"] == "completed":
                continue
            state["stage"] = spec["case"]
            print(f"Timing {index + 1}/{len(manifest['timing'])}: {spec['case']}", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "timing",
                       "--index", str(index), "--output", str(output)]
            logpath = output / "logs" / (spec["case"] + ".log")
            with logpath.open("w", encoding="utf-8") as log:
                try:
                    result = run_logged(command, log, cfg["case_timeout_seconds"])
                except subprocess.TimeoutExpired:
                    data = read(path) if path.exists() else dict(spec=spec, samples=[], checks=[], inputs=[], methods={})
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
                print(f"Memory {index + 1}/{len(manifest['memory'])}: {spec['case']} {spec['solver']}", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "memory",
                       "--index", str(index), "--output", str(output)]
            with (output / "logs" / "memory.log").open("a", encoding="utf-8") as log:
                log.write(spec["job"] + "\n")
                log.flush()
                result = run_logged(command, log, cfg["memory_timeout_seconds"])
            if result.returncode:
                run["failures"].append(dict(case=spec["job"], exit_code=result.returncode))
                raise RuntimeError(f"Memory trial failed: {spec['job']}")
            if index % 10 == 0:
                aggregate(output)
        run.update(aggregate(output))
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
    print_tables(output, manifest)
    print(json.dumps({k: v for k, v in run.items() if k != "failures"} | {"failures": len(run["failures"])},
                     indent=2), flush=True)


def print_tables(output, manifest):
    path = output / "summary.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        table = {(r["case"], r["solver"]): float(r["median_ns"]) / 1e6 for r in csv.DictReader(stream)}
    print("\nMedian milliseconds per matrix; ratio = method / reference (SciPy for R, capped SciPy for F)")
    for spec in manifest["timing"]:
        reference = "scipy" if spec["limit"] is None else "scipy_capped"
        ref = table.get((spec["case"], reference))
        cells = []
        for method in methods(spec):
            t = table.get((spec["case"], method))
            cells.append(f"{LABELS[method]} {t:.3f}" + (f" ({t / ref:.2f})" if ref else "") if t else
                         f"{LABELS[method]} -")
        print(f"{spec['case']:42s} " + ", ".join(cells))


def check_inputs(args):
    manifest = plan()
    timing = []
    for spec in manifest["timing"]:
        for i in range(4):
            timing.append((spec["case"], array_sha256(make_input(spec, SEED, i)[0])))
    memory = sorted((spec["job"], array_sha256(memory_input(spec, SEED))) for spec in manifest["memory"])
    ok = True
    for name, rows in (("timing", timing), ("memory", memory)):
        same = input_digest(rows) == INPUT_DIGESTS[name]
        ok &= same
        print(f"extended {name}: {len(rows)} inputs {'identical to the paper' if same else 'DIFFER from the paper'}")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="tiny grid for wiring checks; not paper timings")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", type=int, help="logical CPU for timing (default: first allowed)")
    parser.add_argument("--check-inputs", action="store_true", help="regenerate inputs and compare with the paper")
    parser.add_argument("--worker", choices=("timing", "memory"), help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return {"timing": timing_worker, "memory": memory_worker}[args.worker](args.output.resolve(), args.index)
    if args.check_inputs:
        initialize()
        return check_inputs(args)
    if args.output is None:
        parser.error("--output is required")
    return controller(args)


if __name__ == "__main__":
    sys.exit(main())
