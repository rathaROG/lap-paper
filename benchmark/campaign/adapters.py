"""Complete-call adapters for the new, separately identified dense campaign.

Input matrices are converted to float64 inside the call. Output is int64 (K,2).
The caller MUST run validation.validate_input on every frozen input before
calling these timing adapters. Extra generic NaN/-infinity scans are outside
the measured boundary; any checks inherent in the backend remain inside it.
No solver is chosen, imported, or loaded inside a measured call. Gates act on
the original costs, before any cap; decoding also checks that original gate.

The eleven cost and six shape constructions are adapted from Pylapy's
cost_extension.py / shape_extension.py, and the two sparse constructions from
its sparse.py (diag_row_no_pad, diag_split_sparse), inspected at pylapy-test revision
70fa1196aca1c7d0b026ebc919ac4dcce5564c99 (30 October 2026).
Upstream: https://github.com/raphaelreme/pylapy . Layout names are retained.
These are independently validated float64 adaptations, not old Pylapy timings.

MIT License, Copyright (c) 2021 raphaelreme
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import importlib
import math
from typing import Callable

import numpy as np


COST_LAYOUTS = (
    "fill_split", "diag_split", "diag_split_sparse", "fill_row_pad_zero",
    "fill_row_pad_inf", "fill_col_pad_zero", "fill_col_pad_inf",
    "diag_row_pad_zero", "diag_row_pad_inf", "diag_col_pad_0", "diag_col_pad_inf",
)
SHAPE_LAYOUTS = (
    "smallest_fill_zero", "smallest_fill_inf", "sum_fill_inf",
    "sum_split_inf", "sum_row_inf", "sum_col_inf",
)
REQUIRED_BACKENDS = ("scipy", "jvx", "jvs")
SPARSE_BACKENDS = ("csgraph", "lapmod")
NATIVE_METHODS = ("jvrust", "jvcpp", "jvhybrid", "jvx")
CANONICAL_METHODS = (
    "scipy", "scipy:capped", "scipy:dummies", "scipy:diagonal_dummies", "scipy:square",
    "jvrust", "jvcpp", "jvhybrid", "jvx", "jvs",
)


class UnsupportedMethod(ValueError):
    """The requested method/objective/backend combination is unavailable."""


class InfeasibleProblem(ValueError):
    """No admissible full smaller-side matching exists."""


class UnsupportedCostDomain(ValueError):
    """A finite sentinel or represented penalty cannot meet this contract."""


def method_ids() -> list[str]:
    return list(CANONICAL_METHODS) + [f"sparse:{backend}" for backend in SPARSE_BACKENDS] + [
        f"{kind}:{backend}:{layout}"
        for kind, layouts in (("cost", COST_LAYOUTS), ("shape", SHAPE_LAYOUTS))
        for backend in REQUIRED_BACKENDS for layout in layouts
    ]


def _contract(case: dict) -> tuple[float | None, float | None]:
    objective = case.get("objective", "required")
    if objective not in ("required", "optional", "gated_optional"):
        raise UnsupportedMethod(f"Objective {objective!r} is a separate validation-only contract")
    value = case.get("limit")
    if objective == "required":
        if value is not None and value != math.inf:
            raise ValueError("Required assignment must not specify a finite rejection penalty")
        limit = None
    else:
        if value is None or not math.isfinite(float(value)):
            raise UnsupportedCostDomain("Optional assignment requires a finite float64 L")
        limit = float(value)
        if (limit / 2) * 2 != limit:
            raise UnsupportedCostDomain("L/2 is not exactly representable in float64")
    gate = case.get("gate")
    if gate is not None:
        gate = float(gate)
        if math.isnan(gate) or gate == -math.inf:
            raise UnsupportedCostDomain("Gate must be finite or positive infinity")
        if gate == math.inf:
            gate = None
    return limit, gate


def as_cost(cost, shape: tuple[int, int] | None = None) -> np.ndarray:
    a = np.asarray(cost, dtype=np.float64)
    if a.ndim != 2 or (shape is not None and a.shape != shape):
        raise ValueError(f"Expected cost shape {shape or '(n,m)'}, got {a.shape}")
    return a


def admissible(cost: np.ndarray, gate: float | None) -> np.ndarray:
    mask = np.isfinite(cost)
    if gate is not None:
        mask &= cost <= gate
    return mask


def masked_cost(cost: np.ndarray, gate: float | None) -> np.ndarray:
    return cost if gate is None else np.where(cost <= gate, cost, np.inf)


def _batch_costs(costs, shape, gate):
    """Preserve an existing 3D input; list input remains a list for native APIs."""
    if isinstance(costs, np.ndarray):
        a = np.asarray(costs, dtype=np.float64)
        if a.ndim != 3 or a.shape[1:] != shape:
            raise ValueError(f"Expected batch shape (B,{shape[0]},{shape[1]}), got {a.shape}")
        return a, masked_cost(a, gate)
    arrays = [as_cost(a, shape) for a in costs]
    return arrays, [masked_cost(a, gate) for a in arrays]


def pairs64(pairs) -> np.ndarray:
    p = np.asarray(pairs)
    if p.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if p.ndim != 2 or p.shape[1] != 2 or p.dtype.kind not in "iu":
        raise ValueError("Backend did not return integral (K,2) assignment pairs")
    return p.astype(np.int64, copy=False)


def finite_sentinel(cost: np.ndarray, limit: float | None = None,
                    order: int | None = None) -> float:
    """A proved dominance bound, not an arbitrary large constant.

    Put N = transformed matching size and A = max(1, |finite costs|, |L|).
    Every transformed objective is q*M + v with integer q and |v| <= N*A.
    Therefore M > 2*N*A makes a change of one sentinel dominate any possible
    finite-cost change. A power of two strictly above 4*(N+1)*A is used; both
    that inequality and finite N*M are checked. This is an exact-real reduction
    bound, NOT a claim that a floating-point solver cannot lose small gaps.
    The represented objective is independently checked by the validation gate.
    """
    finite = cost[np.isfinite(cost)]
    scale = max(1.0, abs(float(limit or 0.0)),
                float(np.max(np.abs(finite))) if finite.size else 0.0)
    n = max(1, int(order if order is not None else sum(cost.shape)))
    bound = 4.0 * (n + 1) * scale
    if not math.isfinite(bound):
        raise UnsupportedCostDomain("Finite-sentinel dominance bound overflows float64")
    try:
        sentinel = math.ldexp(1.0, math.frexp(bound)[1])
    except OverflowError as error:
        raise UnsupportedCostDomain("No representable finite sentinel") from error
    if not sentinel > bound or not math.isfinite(n * sentinel):
        raise UnsupportedCostDomain("Sentinel or transformed objective would overflow")
    return sentinel


def cost_layout(cost: np.ndarray, limit: float, layout: str) -> tuple[np.ndarray, dict]:
    """Return the square layout and F_original = sum(transformed) + offset.

    For row layouts offset=L*(m-n)/2; for column layouts L*(n-m)/2.
    Their pad_inf variants have respectively m and n compulsory M entries,
    so subtract m*M or n*M from that offset. Split layouts have offset zero.
    A minimum-sentinel completion exists for every optional matching (including
    the empty matching). Any forbidden real/off-diagonal edge raises the number
    of sentinels above that minimum and is excluded by finite_sentinel's bound.
    In diag_split_sparse, the reverse of each retained real edge supplies the
    necessary zero edge in the lower-right block.
    """
    if layout not in COST_LAYOUTS:
        raise UnsupportedMethod(f"Unknown cost layout: {layout}")
    n, m = cost.shape
    size = n + m
    sentinel = finite_sentinel(cost, limit, size)
    real = np.where(np.isfinite(cost), cost, sentinel)
    out = np.zeros((size, size), dtype=np.float64)
    out[:n, :m] = real
    offset = 0.0
    if layout == "fill_split":
        out[:n, m:] = limit / 2
        out[n:, :m] = limit / 2
    elif layout in ("diag_split", "diag_split_sparse"):
        out[:n, m:] = sentinel
        out[n:, :m] = sentinel
        np.fill_diagonal(out[:n, m:], limit / 2)
        np.fill_diagonal(out[n:, :m], limit / 2)
        if layout == "diag_split_sparse":
            out[n:, m:] = np.where(np.isfinite(cost.T), 0.0, sentinel)
    elif "row" in layout:
        out[:n, m:] = limit if layout.startswith("fill") else sentinel
        if layout.startswith("diag"):
            np.fill_diagonal(out[:n, m:], limit)
        offset = limit * (m - n) / 2
        if layout.endswith("inf"):
            out[n:, :] = sentinel
            offset -= m * sentinel
    else:
        out[n:, :m] = limit if layout.startswith("fill") else sentinel
        if layout.startswith("diag"):
            np.fill_diagonal(out[n:, :m], limit)
        offset = limit * (n - m) / 2
        if layout.endswith("inf"):
            out[:, m:] = sentinel
            offset -= n * sentinel
    return out, {"layout": layout, "sentinel": sentinel, "objective_offset": offset,
                 "shape": list(out.shape), "cost_buffer_bytes": out.nbytes}


def shape_layout(cost: np.ndarray, layout: str) -> tuple[np.ndarray, dict]:
    """Required layouts, with offset subtracting compulsory padding costs.

    For a full original matching r=min(n,m), the sentinel coefficients are
    |n-m|, N-r, N-2r, n-r, m-r for smallest_inf, sum_fill, sum_split,
    sum_row, sum_col. Zero-padding has coefficient zero. On a feasible graph
    every invalid/short original matching has a strictly larger coefficient
    (or a forbidden edge in the fixed-cardinality smallest layout). The proved
    bound selects a full admissible matching before minimizing its real cost.
    Infeasibility is detected after decoding, never reported as an optional
    optimum. Literal infinity is NOT used for mandatory padding rows.
    """
    if layout not in SHAPE_LAYOUTS:
        raise UnsupportedMethod(f"Unknown shape layout: {layout}")
    n, m = cost.shape
    r, c = min(n, m), max(n, m)
    size = c if layout.startswith("smallest") else n + m
    sentinel = finite_sentinel(cost, order=size)
    fill = 0.0 if layout in ("smallest_fill_zero", "sum_row_inf", "sum_col_inf") else sentinel
    out = np.full((size, size), fill, dtype=np.float64)
    out[:n, :m] = np.where(np.isfinite(cost), cost, sentinel)
    if layout == "sum_split_inf":
        out[n:, m:] = 0
    elif layout == "sum_row_inf":
        out[:n, m:] = sentinel
    elif layout == "sum_col_inf":
        out[n:, :m] = sentinel
    count = {"smallest_fill_zero": 0, "smallest_fill_inf": c-r,
             "sum_fill_inf": n+m-r, "sum_split_inf": n+m-2*r,
             "sum_row_inf": n-r, "sum_col_inf": m-r}[layout]
    return out, {"layout": layout, "sentinel": sentinel, "objective_offset": -count*sentinel,
                 "shape": list(out.shape), "cost_buffer_bytes": out.nbytes}


def _positive_shift(minimum: float, strict: bool) -> float:
    """Power-of-two offset that makes every stored weight >= 0, or > 0 if strict."""
    if minimum > 0 or (minimum == 0 and not strict):
        return 0.0
    return math.ldexp(1.0, math.frexp(max(-minimum, .5))[1])


def _sparse_solver(backend: str) -> Callable:
    """Optional assignment on admissible edges only: cost < L after gating.

    csgraph receives Pylapy's diag_row_no_pad layout, (n, n+m), with one row
    sink priced L. lapmod receives diag_split_sparse, (n+m, n+m), with L/2 row
    and column sinks and zero-cost mirrors of the admissible edges. Every
    extended row is matched exactly once, so one common power-of-two shift
    preserves the optimum. It keeps csgraph weights positive, because SciPy
    drops stored zeros, and lapmod weights nonnegative, as lapmod requires.
    """
    try:
        sparse = importlib.import_module("scipy.sparse")
        if backend == "csgraph":
            match = importlib.import_module("scipy.sparse.csgraph").min_weight_full_bipartite_matching
        elif backend == "lapmod":
            lapmod = importlib.import_module("lap").lapmod
        else:
            raise UnsupportedMethod(f"Unknown sparse backend: {backend}")
    except (ImportError, AttributeError, OSError) as error:
        raise UnsupportedMethod(f"{backend} sparse backend unavailable: {error}") from error

    def solve(work: np.ndarray, limit: float) -> np.ndarray:
        n, m = work.shape
        rows, cols = np.nonzero(work < limit)
        values = work[rows, cols]
        least = float(values.min()) if values.size else math.inf
        if backend == "csgraph":
            shift = _positive_shift(min(least, limit), strict=True)
            graph = sparse.coo_array(
                (np.concatenate((values, np.full(n, limit))) + shift,
                 (np.concatenate((rows, np.arange(n))), np.concatenate((cols, np.arange(m, m + n))))),
                shape=(n, n + m)).tocsr()
            matched_rows, matched_cols = match(graph)
            keep = matched_cols < m
            return np.column_stack((matched_rows[keep], matched_cols[keep]))
        size = n + m
        shift = _positive_shift(min(least, limit / 2, 0.0), strict=False)
        graph = sparse.coo_array(
            (np.concatenate((values, np.zeros(len(values)), np.full(size, limit / 2))) + shift,
             (np.concatenate((rows, cols + n, np.arange(size))),
              np.concatenate((cols, rows + m, np.arange(m, size), np.arange(m))))),
            shape=(size, size)).tocsr()
        x, _ = lapmod(size, graph.data, graph.indptr, graph.indices, return_cost=False)
        x = np.asarray(x)[:n]
        keep = x < m
        return np.column_stack((np.flatnonzero(keep), x[keep]))
    return solve


def _required_solver(backend: str) -> Callable:
    try:
        if backend == "scipy":
            lsa = importlib.import_module("scipy.optimize").linear_sum_assignment
            return lambda a: np.column_stack(lsa(a)).astype(np.int64, copy=False)
        lap = importlib.import_module("lap")
        if backend == "jvx":
            fn = lap.lapjvxa
            return lambda a: pairs64(fn(a, extend_cost=True, cost_limit=math.inf, return_cost=False))
        if backend == "jvs":
            fn = lap.lapjvsa
            return lambda a: pairs64(fn(a, extend_cost=True, return_cost=False, prefer_float32=False))
    except (ImportError, AttributeError, OSError) as error:
        raise UnsupportedMethod(f"{backend} required backend unavailable: {error}") from error
    raise UnsupportedMethod(f"Unknown required backend: {backend}")


def _native_solver(backend: str, batch: bool, workers: int) -> Callable:
    package, function = {"jvrust": ("laprust", "lapjv"), "jvcpp": ("lapcpp", "lapjv"),
                         "jvhybrid": ("lap", "lapjvh"), "jvx": ("lap", "lapjvxa")}[backend]
    try:
        fn = getattr(importlib.import_module(package), function + ("_batch" if batch else ""))
    except (ImportError, AttributeError, OSError) as error:
        raise UnsupportedMethod(f"{backend} {'batch' if batch else 'single'} API unavailable: {error}") from error
    kwargs = {"extend_cost": True, "return_cost": False}
    if backend != "jvx":
        kwargs.update(precision="float64", output="pairs")
    if batch:
        kwargs["n_threads"] = workers
    return lambda a, limit: fn(a, cost_limit=limit, **kwargs)


def _build_record(index, language: str, arm: str) -> dict:
    aliases = {"w0": "reference", "w1": "fresh_state", "w2": "fresh_worker",
               "budget4": "reference", "default": "reference"}
    arm = aliases.get(arm.lower(), arm)
    if index is None:
        raise UnsupportedMethod("Controlled native method requires an explicit build index")
    if isinstance(index, (str, bytes)) or hasattr(index, "__fspath__"):
        import json
        from pathlib import Path
        try:
            index = json.loads(Path(index).read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise UnsupportedMethod(f"Controlled build index unavailable: {error}") from error
    records = index.get("builds", []) if isinstance(index, dict) else index
    if isinstance(records, dict):
        records = list(records.values())
    selected = [x for x in records if x.get("language") == language and x.get("arm_id") == arm]
    if len(selected) != 1:
        raise UnsupportedMethod(f"Need one controlled build for {language}/{arm}, found {len(selected)}")
    return selected[0]


def _decode(pairs, cost: np.ndarray, gate: float | None, limit: float | None,
            trim: bool = False) -> np.ndarray:
    p = pairs64(pairs)
    if trim:
        p = p[(p[:, 0] >= 0) & (p[:, 0] < cost.shape[0]) &
              (p[:, 1] >= 0) & (p[:, 1] < cost.shape[1])]
    if p.size:
        selected = cost[p[:, 0], p[:, 1]]
        keep = np.isfinite(selected)
        if gate is not None:
            keep &= selected <= gate
        if limit is not None:
            keep &= selected < limit
        p = p[keep]
    if limit is None and len(p) != min(cost.shape):
        raise InfeasibleProblem("Required solver did not find an admissible full matching")
    return p


def prepare_method(method_id: str, case: dict, build_index=None) -> Callable:
    """Prepare one fixed callable outside primary timing.

    A batch callable consumes a list/3D array and returns a list of pairs. Plain
    methods on a batch case use a serial loop; batch:* and workspace:* use the
    actual public batch API. ``scipy`` means ordinary required LSA and, for an
    optional case, the declared compact capped comparator.
    """
    limit, gate = _contract(case)
    shape = (int(case["rows"]), int(case["cols"]))
    batch = case.get("kind") == "batch"
    parts = method_id.split(":")
    info = {"method": method_id, "input_dtype": "float64", "output_dtype": "int64",
            "limit": limit, "effective_limit": limit, "gate": gate,
            "batch": batch, "pool_lifetime": None}
    if parts[0] == "loop":
        if not batch:
            raise UnsupportedMethod("loop: methods require a batch case")
        single_case = {**case, "kind": "single"}
        single = prepare_method(":".join(parts[1:]), single_case, build_index)
        call = lambda costs: [single(a) for a in costs]
        info["execution"] = "serial_loop"
    elif parts[0] in ("policy", "workspace"):
        expected = 4 if parts[0] == "workspace" else 3
        if len(parts) != expected or parts[1] not in ("rust", "cpp"):
            raise UnsupportedMethod(f"Malformed controlled method: {method_id}")
        if parts[0] == "workspace" and not batch:
            raise UnsupportedMethod("Workspace contrasts require a batch case")
        if limit is not None and parts[2] in ("cutoff_off", "ties_cutoff_off"):
            raise UnsupportedMethod("Required-only cutoff has no optional treatment")
        if limit is None and shape[0] == shape[1] and parts[2] not in ("reference", "budget4", "w0", "w1", "w2"):
            raise UnsupportedMethod("Rectangular policy treatment is inapplicable to required square input")
        workers = int(parts[3]) if parts[0] == "workspace" else int(case.get("workers", 1))
        if workers not in (1, 4):
            raise UnsupportedMethod("Declared native worker counts are one and four")
        record = _build_record(build_index, parts[1], parts[2])
        try:
            module = importlib.import_module(".native_build", __package__)
            solver = module.load_solver(record)
        except (ImportError, OSError, ValueError) as error:
            raise UnsupportedMethod(f"Controlled build unavailable: {error}") from error
        fixed_limit = math.inf if limit is None else limit
        def controlled(costs):
            if not batch:
                a = as_cost(costs, shape)
                output = solver(masked_cost(a, gate), limit=fixed_limit, batch=False, workers=workers)
                return _decode(output, a, gate, limit)
            arrays, work = _batch_costs(costs, shape, gate)
            if len(arrays) == 0:
                return []
            if min(shape) == 0:
                return [np.empty((0, 2), dtype=np.int64) for _ in arrays]
            outputs = solver(work, limit=fixed_limit, batch=True, workers=workers)
            if len(outputs) != len(arrays):
                raise ValueError("Native batch output count differs from input count")
            return [_decode(p, a, gate, limit) for p, a in zip(outputs, arrays)]
        call = controlled
        info.update(build=record, workers=workers, execution="native_batch" if batch else "single",
                    pool_lifetime="cached" if batch else None)
    elif parts[0] == "batch":
        if not batch or len(parts) != 3 or parts[1] not in NATIVE_METHODS:
            raise UnsupportedMethod(f"Unsupported native batch method: {method_id}")
        workers = int(parts[2])
        if workers not in (1, 4):
            raise UnsupportedMethod("Declared native worker counts are one and four")
        solver = _native_solver(parts[1], True, workers)
        fixed_limit = math.inf if limit is None else limit
        def native_batch(costs):
            arrays, work = _batch_costs(costs, shape, gate)
            if len(arrays) == 0:
                return []
            if min(shape) == 0:
                return [np.empty((0, 2), dtype=np.int64) for _ in arrays]
            if parts[1] == "jvx":
                work = np.asarray(work, dtype=np.float64)
            outputs = solver(work, fixed_limit)
            if len(outputs) != len(arrays):
                raise ValueError("Native batch output count differs from input count")
            return [_decode(p, a, gate, limit) for p, a in zip(outputs, arrays)]
        call = native_batch
        info.update(workers=workers, execution="native_batch",
                    pool_lifetime="per_call" if parts[1] == "jvx" else "cached")
    elif batch:
        single = prepare_method(method_id, {**case, "kind": "single"}, build_index)
        call = lambda costs: [single(a) for a in costs]
        info["execution"] = "serial_loop"
    elif len(parts) == 1 and parts[0] in NATIVE_METHODS:
        solver = _native_solver(parts[0], False, 1)
        fixed_limit = math.inf if limit is None else limit
        def native(cost):
            a = as_cost(cost, shape)
            if min(shape) == 0:
                return np.empty((0, 2), dtype=np.int64)
            return _decode(solver(masked_cost(a, gate), fixed_limit), a, gate, limit)
        call = native
    elif parts[0] == "sparse":
        if len(parts) != 2 or parts[1] not in SPARSE_BACKENDS:
            raise UnsupportedMethod(f"Unknown sparse method: {method_id}")
        if limit is None:
            raise UnsupportedMethod("Sparse methods require an optional objective")
        solve = _sparse_solver(parts[1])
        info.update(backend=parts[1], family="sparse",
                    layout="diag_row_no_pad" if parts[1] == "csgraph" else "diag_split_sparse")
        def sparse_call(cost):
            a = as_cost(cost, shape)
            if min(shape) == 0:
                return np.empty((0, 2), dtype=np.int64)
            return _decode(solve(masked_cost(a, gate), limit), a, gate, limit)
        call = sparse_call
    else:
        if parts[0] in ("cost", "shape"):
            if len(parts) != 3 or parts[1] not in REQUIRED_BACKENDS:
                raise UnsupportedMethod(f"Unknown layout/backend method: {method_id}")
            family, backend, layout = parts
            if family == "cost" and (limit is None or layout not in COST_LAYOUTS):
                raise UnsupportedMethod("Cost layouts require an optional objective and known layout")
            if family == "shape" and (limit is not None or layout not in SHAPE_LAYOUTS):
                raise UnsupportedMethod("Shape layouts require a required objective and known layout")
        elif parts[0] == "scipy" and len(parts) <= 2:
            backend = "scipy"
            family = "canonical"
            layout = parts[1] if len(parts) == 2 else ("required" if limit is None else "capped")
            if layout not in ("required", "capped", "dummies", "diagonal_dummies", "square"):
                raise UnsupportedMethod(f"Unknown canonical SciPy layout: {layout}")
            if limit is None and layout != "required":
                raise UnsupportedMethod("Optional SciPy layouts require a finite penalty")
        elif parts[0] == "jvs" and len(parts) == 1:
            family, backend, layout = "canonical", parts[0], "required" if limit is None else "square"
        else:
            raise UnsupportedMethod(f"Unknown method: {method_id}")
        solve = _required_solver(backend)
        info.update(backend=backend, layout=layout, family=family)
        def represented(cost):
            a = as_cost(cost, shape)
            if min(shape) == 0:
                return np.empty((0, 2), dtype=np.int64)
            work = masked_cost(a, gate)
            transposed = False
            trim = False
            if family == "cost":
                work, _ = cost_layout(work, limit, layout)
                trim = True
            elif family == "shape":
                work, _ = shape_layout(work, layout)
                trim = True
            elif layout == "capped":
                work = np.minimum(work, limit)
            elif layout in ("dummies", "diagonal_dummies"):
                transposed = shape[0] > shape[1]
                base = work.T if transposed else work
                r, c = base.shape
                work = np.full((r, c+r), limit if layout == "dummies" else np.inf, dtype=np.float64)
                work[:, :c] = base
                if layout == "diagonal_dummies":
                    np.fill_diagonal(work[:, c:], limit)
                trim = True
            elif layout == "square":
                n, m = shape
                work2 = np.zeros((n+m, n+m), dtype=np.float64)
                work2[:n, :m] = work
                work2[:n, m:] = limit / 2
                work2[n:, :m] = limit / 2
                work = work2
                trim = True
            try:
                p = solve(work)
            except ValueError as error:
                if "infeasible" in str(error).lower() or "no feasible" in str(error).lower():
                    raise InfeasibleProblem(str(error)) from error
                raise
            if transposed:
                p = p[:, ::-1]
            return _decode(p, a, gate, limit, trim=trim)
        call = represented
    call.method_info = info
    return call


def method_status(method_id: str, case: dict, build_index=None) -> dict:
    """Availability, not a correctness certificate; performs no solver call."""
    try:
        call = prepare_method(method_id, case, build_index)
        return {"method": method_id, "status": "supported", "metadata": call.method_info}
    except (UnsupportedMethod, UnsupportedCostDomain) as error:
        return {"method": method_id, "status": "unsupported", "reason": str(error)}
