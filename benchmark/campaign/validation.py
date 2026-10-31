"""Independent, objective-based correctness gates; never timing observations.

Tiny cases use Fraction.from_float, i.e. the exact binary-rational values that
the float64 adapters actually receive. Optional comparisons deliberately do
not require the same pairs, cardinality, or selected-cost sum. Large-case
agreement is tolerance-based evidence, not an exactness proof.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from .adapters import (InfeasibleProblem, UnsupportedCostDomain, UnsupportedMethod,
                       _contract, admissible, as_cost, method_ids, prepare_method)


class ValidationFailure(ValueError):
    """A returned answer violates feasibility or the represented objective."""


class ExactObjectiveMismatch(ValidationFailure):
    def __init__(self, actual, expected):
        self.actual = actual
        self.expected = expected
        super().__init__(f"Exact represented objective {actual} != optimum {expected}")


def _fraction(value) -> Fraction:
    return Fraction.from_float(float(value))


def _checked_cost(cost, shape=None):
    a = as_cost(cost, shape)
    if np.isnan(a).any() or np.isneginf(a).any():
        raise UnsupportedCostDomain("NaN and negative infinity are outside the cost contract")
    return a


def validate_input(case, cost) -> dict:
    """Preflight the entire frozen input outside primary timing.

    Dtype conversion remains inside the complete call. Report both storage and
    represented arithmetic identities; +infinity denotes forbidden edges, while
    NaN/-infinity and penalties with an unrepresentable half are unsupported.
    """
    limit, gate = _contract(case)
    shape = (int(case["rows"]), int(case["cols"]))
    arrays = list(cost) if case.get("kind") == "batch" else [cost]
    inputs = []
    for value in arrays:
        a = _checked_cost(value, shape)
        inputs.append({"storage_dtype": str(np.asarray(value).dtype), "represented_dtype": "float64",
                       "shape": list(a.shape), "forbidden_entries": int((~admissible(a, gate)).sum())})
    return {"status": "passed", "inputs": inputs, "nominal_limit": case.get("limit"),
            "effective_limit": limit, "gate": gate,
            "contract": "represented_float64_costs_and_penalty"}


def check_feasible(cost, pairs, limit=None, gate=None) -> np.ndarray:
    a = _checked_cost(cost)
    p = np.asarray(pairs)
    if p.dtype != np.dtype(np.int64) or p.ndim != 2 or p.shape[1] != 2:
        raise ValidationFailure("Output must be an int64 (K,2) array, including empty outputs")
    if len(p) > min(a.shape):
        raise ValidationFailure("Too many selected pairs")
    for axis in (0, 1):
        if np.any(p[:, axis] < 0) or np.any(p[:, axis] >= a.shape[axis]):
            raise ValidationFailure("Selected endpoint is outside the original matrix")
        if len(np.unique(p[:, axis])) != len(p):
            raise ValidationFailure("An endpoint occurs more than once")
    if len(p) and not admissible(a, gate)[p[:, 0], p[:, 1]].all():
        raise ValidationFailure("Returned matching contains a forbidden edge")
    if limit is None and len(p) != min(a.shape):
        raise ValidationFailure("Required assignment lacks full smaller-side cardinality")
    return p


def objective(cost, pairs, limit=None) -> float:
    """Original-input objective, using L/2 for EACH unmatched endpoint."""
    a = np.asarray(cost, dtype=np.float64)
    p = np.asarray(pairs)
    values = [float(a[i, j]) for i, j in p]
    if limit is not None and math.isfinite(limit):
        values.append((float(limit) / 2) * (sum(a.shape) - 2 * len(p)))
    try:
        result = math.fsum(values)
    except OverflowError as error:
        raise UnsupportedCostDomain("Original-input objective overflows float64") from error
    if not math.isfinite(result):
        raise UnsupportedCostDomain("Original-input objective is nonfinite")
    return result


penalized_objective = objective


def exact_objective(cost, pairs, limit=None) -> Fraction:
    a = _checked_cost(cost)
    total = sum((_fraction(a[i, j]) for i, j in pairs), Fraction(0))
    if limit is not None and math.isfinite(limit):
        total += _fraction(limit) * (sum(a.shape) - 2 * len(pairs)) / 2
    return total


def exact_oracle(cost, limit=None, gate=None, objective_kind=None, max_dimension=9) -> dict:
    """Enumerate partial matchings, independently of every solver/reduction.

    The hard-gated maximum-cardinality control is lexicographic (-K, cost).
    It has no optional penalty and is never silently admitted to timed methods.
    Returned Fractions are for internal comparison; summary records stringify
    them explicitly. Both orientations and all optimal cardinalities are covered.
    """
    a = _checked_cost(cost)
    if max(a.shape, default=0) > max_dimension:
        raise ValueError("Exact oracle is restricted to tiny validation matrices")
    original_shape = a.shape
    a = a if a.shape[0] <= a.shape[1] else a.T
    r, c = a.shape
    finite = admissible(a, gate)
    values = [[_fraction(a[i, j]) if finite[i, j] else None for j in range(c)] for i in range(r)]
    lexicographic = objective_kind in ("maximum_cardinality", "hard_gated_maximum_cardinality", "hard_gated")
    required = limit is None and not lexicographic

    @lru_cache(None)
    def visit(i, used):
        if i == r:
            return ((0, Fraction(0)),)
        best_by_k = {}
        candidates = []
        if not required:
            candidates.extend(visit(i + 1, used))
        for j in range(c):
            if not used & (1 << j) and values[i][j] is not None:
                candidates.extend((k + 1, value + values[i][j])
                                  for k, value in visit(i + 1, used | (1 << j)))
        for k, value in candidates:
            if k not in best_by_k or value < best_by_k[k]:
                best_by_k[k] = value
        return tuple(sorted(best_by_k.items()))

    possibilities = visit(0, 0)
    if not possibilities:
        return {"feasible": False, "objective": None, "cardinalities": []}
    if lexicographic:
        k, value = max(possibilities, key=lambda item: item[0])
        return {"feasible": True, "objective": value, "cardinalities": [k],
                "maximum_cardinality": k, "objective_kind": "maximum_cardinality"}
    adjusted = [(k, value if required else value + _fraction(limit) * (sum(original_shape) - 2*k) / 2)
                for k, value in possibilities]
    best = min(value for _, value in adjusted)
    return {"feasible": True, "objective": best,
            "cardinalities": [k for k, value in adjusted if value == best]}


def _one_output(case, cost, output, reference=None, *, exact=False, rtol=1e-11, atol=1e-12):
    limit, gate = _contract(case)
    a = as_cost(cost, (int(case["rows"]), int(case["cols"])))
    p = check_feasible(a, output, limit, gate)
    actual = objective(a, p, limit)
    actual_exact = exact_objective(a, p, limit) if exact else None
    expected = None
    if isinstance(reference, dict):
        if reference.get("feasible") is False:
            raise ValidationFailure("An output was returned for an oracle-infeasible problem")
        expected = reference.get("objective")
    elif isinstance(reference, np.ndarray):
        check_feasible(a, reference, limit, gate)
        expected = exact_objective(a, reference, limit) if exact else objective(a, reference, limit)
    elif reference is not None:
        expected = reference
    if expected is not None:
        if exact:
            if actual_exact != expected:
                raise ExactObjectiveMismatch(actual_exact, expected)
        elif not math.isclose(actual, float(expected), rel_tol=rtol, abs_tol=atol):
            raise ValidationFailure(f"Objective {actual!r} != reference {float(expected)!r}")
    return {"status": "passed", "objective": actual, "cardinality": len(p),
            "exact_objective": str(actual_exact) if exact else None,
            "comparison": "exact_binary_rational" if exact else ("tolerance" if expected is not None else "feasibility_only"),
            "rtol": None if exact else rtol, "atol": None if exact else atol}


def validate_output(case, cost, output, reference=None, *, exact=False,
                    rtol=1e-11, atol=1e-12) -> dict:
    """Validate a runner/native-arm output, optionally against pairs or objective.

    A batch is one unit: all its outputs must pass; none may be silently omitted.
    ``reference`` can be a scalar objective, exact_oracle result, pairs array,
    or a corresponding list for batches. This function performs no solver call.
    """
    if case.get("kind") != "batch":
        return _one_output(case, cost, output, reference, exact=exact, rtol=rtol, atol=atol)
    arrays = list(cost)
    if not isinstance(output, (list, tuple)) or len(output) != len(arrays):
        raise ValidationFailure("Batch output count/type does not match the full input batch")
    refs = [None] * len(arrays) if reference is None else list(reference)
    if len(refs) != len(arrays):
        raise ValidationFailure("Reference count differs from input batch")
    items = [_one_output(case, a, p, ref, exact=exact, rtol=rtol, atol=atol)
             for a, p, ref in zip(arrays, output, refs)]
    return {"status": "passed", "items": items, "count": len(items)}


def fixtures() -> list[dict]:
    """Small deterministic contracts, including ties without an edge equal to L."""
    records = []
    def add(name, values, limit=None, gate=None, *, transpose=True, noncontiguous=False,
            numerical_boundary=False):
        a = np.asarray(values, dtype=np.float64)
        if noncontiguous:
            backing = np.zeros((a.shape[0], a.shape[1]*2), dtype=np.float64)
            backing[:, ::2] = a
            a = backing[:, ::2]
        case = {"id": name, "kind": "single", "objective": "required" if limit is None else "optional",
                "rows": a.shape[0], "cols": a.shape[1], "limit": limit, "gate": gate}
        records.append({"case": case, "cost": a, "numerical_boundary": numerical_boundary})
        if transpose and a.shape[0] != a.shape[1]:
            records.append({"case": {**case, "id": name+"_transpose", "rows": a.shape[1], "cols": a.shape[0]},
                            "cost": a.T, "numerical_boundary": numerical_boundary})
    add("required_signed", [[-2, 1, 0], [0, -1, 2]])
    add("required_ties", [[0, 0, 1], [0, 0, 1]])
    add("required_forbidden", [[0, np.inf, 1], [np.inf, 0, 1]])
    add("required_hall_infeasible", [[0, np.inf], [0, np.inf]])
    add("required_all_forbidden", [[np.inf, np.inf]])
    add("required_noncontiguous", [[1, 0, 2], [0, 1, 2]], noncontiguous=True)
    add("different_cardinality_optima", [[0, .5], [.5, np.inf]], 1.0)
    add("threshold_equality", [[1, 1.5, 1], [1.5, 1, 1.5]], 1.0)
    add("threshold_adjacent", [[np.nextafter(1.0, 0.0), np.nextafter(1.0, np.inf)]], 1.0,
        numerical_boundary=True)
    add("optional_signed", [[-2, 0, 3], [-1, -1, 2]], .5)
    add("optional_negative_penalty", [[-2, 0, -0.25], [-1, -1, 2]], -.25)
    add("optional_all_forbidden", [[np.inf, np.inf], [np.inf, np.inf]], 1.0)
    add("gate_forbidden_below_limit", [[.75, .8], [.8, 0]], 1.0, .25)
    add("gate_boundary", [[.5, .75, .25], [.25, .5, .75]], 1.0, .5)
    add("gate_redundant", [[0, .5, 1.5], [.75, 1, 0]], 1.0, 1.25)
    add("optional_noncontiguous", [[0, 2, 1], [1, 0, 2]], 1.0, noncontiguous=True)
    add("zero_penalty_signed", [[-1, 0, 1], [0, 0, 1]], 0.0)
    for shape in ((0, 0), (0, 3), (3, 0)):
        add(f"required_empty_{shape}", np.empty(shape), transpose=False)
        add(f"optional_empty_{shape}", np.empty(shape), .5, transpose=False)
    return records


def batch_fixtures(copies=4) -> list[dict]:
    """Same strict dyadic/feasibility contracts through native and loop batches.

    Adjacent-representable diagnostics remain separately identified single-input
    fixtures. Four copies exercise the declared four-worker API without changing
    the objective or adding workload timing observations.
    """
    return [{**item, "case": {**item["case"], "id": "batch_"+item["case"]["id"], "kind": "batch"},
             "cost": np.stack([item["cost"]] * copies)}
            for item in fixtures() if not item.get("numerical_boundary")]


def _unchanged(before, after) -> bool:
    return np.array_equal(before, after, equal_nan=True)


def boundary_tolerance(cost, limit) -> dict:
    """Declared diagnostic threshold on ORIGINAL cost scale, never on M.

    This is a classification rule for explicitly tagged precision-boundary
    fixtures, not a forward-error theorem. Ordinary exact dyadic tests do not
    receive this exception. Real campaign inputs still require their own
    feasibility and tolerance-based reference check before measurement.
    """
    a = as_cost(cost)
    finite = np.abs(a[np.isfinite(a)])
    r = min(a.shape)
    scale = max(1.0, r * (float(finite.max()) if finite.size else 0.0),
                (abs(float(limit)) / 2) * sum(a.shape) if limit is not None else 0.0)
    tolerance = 8 * np.finfo(np.float64).eps * (r + 1) * scale
    return {"policy": "tagged_boundary_8eps_rplus1_original_scale_v1",
            "scale": scale, "absolute_tolerance": tolerance,
            "interpretation": "diagnostic classification; not exact optimality or a solver error bound"}


def validate_method(method_id: str, records=None, build_index=None) -> dict:
    """Run exact tiny fixtures; report unsupported separately from failures.

    Controlled arms use this very same API. Unsupported objective/shape cells
    are retained, rather than passing merely because no tests ran.
    """
    records = fixtures() if records is None else records
    results = []
    for record in records:
        case, cost = record["case"], record["cost"]
        item = {"method": method_id, "case": case["id"]}
        try:
            validate_input(case, cost)
            call = prepare_method(method_id, case, build_index)
        except (UnsupportedMethod, UnsupportedCostDomain) as error:
            results.append({**item, "status": "unsupported", "reason": str(error)})
            continue
        arrays = list(cost) if case.get("kind") == "batch" else [cost]
        before = [np.array(a, copy=True) for a in arrays]
        limit, gate = _contract(case)
        oracles = [exact_oracle(a, limit, gate) for a in arrays]
        try:
            output = call(cost)
            checked = validate_output(case, cost, output,
                                      oracles if case.get("kind") == "batch" else oracles[0], exact=True)
            if any(not _unchanged(a, b) for a, b in zip(arrays, before)):
                raise ValidationFailure("Solver/adapter mutated the original input")
            results.append({**item, **checked})
        except (InfeasibleProblem, ValueError) as error:
            infeasible = any(not x["feasible"] for x in oracles)
            claims_infeasible = isinstance(error, InfeasibleProblem) or any(
                word in str(error).lower() for word in ("infeasible", "no feasible"))
            unchanged = all(_unchanged(a, b) for a, b in zip(arrays, before))
            if isinstance(error, ExactObjectiveMismatch) and record.get("numerical_boundary") and len(arrays) == 1 and unchanged:
                tolerance = boundary_tolerance(arrays[0], limit)
                gap = error.actual - error.expected
                if 0 < gap <= _fraction(tolerance["absolute_tolerance"]):
                    results.append({**item, "status": "numerical_boundary_limitation",
                                    "reason": str(error), "exact_actual": str(error.actual),
                                    "exact_optimum": str(error.expected), "exact_gap": str(gap),
                                    "diagnostic_tolerance": tolerance,
                                    "campaign_rule": "each input must separately pass reference/feasibility validation"})
                    continue
            if infeasible and claims_infeasible and unchanged:
                results.append({**item, "status": "passed", "outcome": "expected_infeasible"})
            elif isinstance(error, UnsupportedCostDomain):
                results.append({**item, "status": "unsupported", "reason": str(error)})
            else:
                results.append({**item, "status": "correctness_failed", "reason": str(error)})
        except Exception as error:
            results.append({**item, "status": "correctness_failed", "reason": f"{type(error).__name__}: {error}"})
    failures = [x for x in results if x["status"] == "correctness_failed"]
    unsupported = [x for x in results if x["status"] == "unsupported"]
    passed = [x for x in results if x["status"] == "passed"]
    numerical = [x for x in results if x["status"] == "numerical_boundary_limitation"]
    status = ("correctness_failed" if failures else "passed_with_numerical_limitations" if numerical
              else "passed" if passed else "unsupported")
    return {"method": method_id, "status": status, "passed": len(passed),
            "unsupported": len(unsupported), "failures": len(failures),
            "numerical_boundary_limitations": len(numerical), "results": results}


def validate_methods(methods=None, records=None, build_index=None) -> dict:
    results = [validate_method(method, records, build_index) for method in (method_ids() if methods is None else methods)]
    return {"scope": "tiny_correctness_validation_not_campaign_measurements", "methods": results,
            "passed": [x["method"] for x in results if x["status"] == "passed"],
            "passed_with_numerical_limitations": [x["method"] for x in results if x["status"] == "passed_with_numerical_limitations"],
            "unsupported": [x["method"] for x in results if x["status"] == "unsupported"],
            "correctness_failed": [x["method"] for x in results if x["status"] == "correctness_failed"]}
