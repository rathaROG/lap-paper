"""Tiny contract and oracle checks, never a performance campaign."""
from fractions import Fraction
import math

import numpy as np
import pytest

from . import adapters, validation
from .adapters import (COST_LAYOUTS, SHAPE_LAYOUTS, InfeasibleProblem,
                       UnsupportedCostDomain, UnsupportedMethod, cost_layout,
                       finite_sentinel, method_ids, method_status, prepare_method,
                       shape_layout)
from .validation import (ValidationFailure, check_feasible, exact_objective,
                         exact_oracle, fixtures, objective, validate_method,
                         validate_methods, validate_output, validate_input, batch_fixtures)


def case(cost, limit=None, gate=None, kind="single"):
    return {"id": "test", "kind": kind, "objective": "required" if limit is None else "optional",
            "rows": cost.shape[-2], "cols": cost.shape[-1], "limit": limit, "gate": gate}


def test_registry_has_all_eleven_cost_and_six_shape_layouts():
    assert len(COST_LAYOUTS) == 11
    assert len(SHAPE_LAYOUTS) == 6
    assert len(method_ids()) == 63 == len(set(method_ids()))


@pytest.mark.parametrize("method", ["sparse:csgraph", "sparse:lapmod"])
def test_sparse_methods_keep_zero_and_negative_weights(method):
    # SciPy drops stored zero weights and lapmod rejects negative ones; the
    # common shift must keep the zero-cost and negative-cost optimal edges.
    for a, limit in ((np.array([[0., .75], [.75, 0.]]), 1.), (np.array([[-1., .5], [.5, -1.]]), 0.),
                     (np.array([[0., 2.], [2., 2.], [2., 0.]]), 0.25)):
        expected = exact_oracle(a, limit)
        p = prepare_method(method, case(a, limit))(a)
        assert validate_output(case(a, limit), a, p, expected, exact=True)["status"] == "passed"
    with pytest.raises(UnsupportedMethod):
        prepare_method(method, case(np.ones((2, 2))))


def test_endpoint_penalty_empty_sides_and_original_costs():
    a = np.array([[2., 8., 9.], [6., 3., 7.]])
    p = np.array([[0, 0], [1, 1]], dtype=np.int64)
    assert objective(a, p, 4) == 7  # one unmatched endpoint costs L/2
    assert objective(a, p) == 5
    empty = np.empty((0, 2), dtype=np.int64)
    assert objective(np.empty((0, 3)), empty, 4) == 6
    assert objective(np.empty((3, 0)), empty, -4) == -6
    assert objective(np.empty((0, 0)), empty, 4) == 0
    assert objective(np.empty((0, 2)), empty, 1e308) == 1e308


def test_different_cardinalities_are_equally_optimal_without_edges_at_limit():
    a = np.array([[0., .5], [.5, np.inf]])
    one = np.array([[0, 0]], dtype=np.int64)
    two = np.array([[0, 1], [1, 0]], dtype=np.int64)
    oracle = exact_oracle(a, 1)
    assert oracle == {"feasible": True, "objective": Fraction(1), "cardinalities": [1, 2]}
    assert not np.any(a == 1)
    for p in (one, two):
        assert validate_output(case(a, 1), a, p, oracle, exact=True)["status"] == "passed"


def test_hard_gated_maximum_cardinality_is_not_optional_assignment():
    a = np.array([[0., 100.], [100., np.inf]])
    hard = exact_oracle(a, gate=100, objective_kind="maximum_cardinality")
    optional = exact_oracle(a, limit=1, gate=100)
    assert hard["maximum_cardinality"] == 2 and hard["objective"] == 200
    assert optional["cardinalities"] == [1] and optional["objective"] == 1
    with pytest.raises(UnsupportedMethod):
        prepare_method("scipy", {**case(a), "objective": "maximum_cardinality"})


@pytest.mark.parametrize("method", ["scipy:capped", "scipy:dummies", "scipy:diagonal_dummies", "scipy:square"])
def test_gated_decoder_rejects_forbidden_raw_cost_below_L(method):
    a = np.array([[.75, .8], [.8, 0.]])
    spec = case(a, 1, .25)
    p = prepare_method(method, spec)(a)
    assert p.tolist() == [[1, 1]]
    validate_output(spec, a, p, exact_oracle(a, 1, .25), exact=True)
    with pytest.raises(ValidationFailure, match="forbidden"):
        validate_output(spec, a, np.array([[0, 0], [1, 1]], dtype=np.int64))


@pytest.mark.parametrize("layout", COST_LAYOUTS)
@pytest.mark.parametrize("transposed", [False, True])
def test_cost_layout_exact_offset_and_decoding(layout, transposed):
    from scipy.optimize import linear_sum_assignment
    a = np.array([[-.5, .75, np.inf], [.5, 2., .25]])
    if transposed:
        a = a.T
    work, metadata = cost_layout(a, .5, layout)
    rows, cols = linear_sum_assignment(work)
    all_pairs = np.column_stack((rows, cols)).astype(np.int64)
    keep = (rows < a.shape[0]) & (cols < a.shape[1])
    p = all_pairs[keep]
    represented = exact_objective(work, all_pairs) + Fraction.from_float(metadata["objective_offset"])
    assert represented == exact_objective(a, p, .5) == exact_oracle(a, .5)["objective"]
    assert metadata["cost_buffer_bytes"] == (sum(a.shape))**2 * 8
    assert np.isfinite(work).all()


@pytest.mark.parametrize("layout", SHAPE_LAYOUTS)
@pytest.mark.parametrize("transposed", [False, True])
def test_shape_layout_exact_offset_and_decoding(layout, transposed):
    from scipy.optimize import linear_sum_assignment
    a = np.array([[-.5, .75, np.inf], [.5, 2., .25]])
    if transposed:
        a = a.T
    work, metadata = shape_layout(a, layout)
    rows, cols = linear_sum_assignment(work)
    all_pairs = np.column_stack((rows, cols)).astype(np.int64)
    p = all_pairs[(rows < a.shape[0]) & (cols < a.shape[1])]
    check_feasible(a, p)
    represented = exact_objective(work, all_pairs) + Fraction.from_float(metadata["objective_offset"])
    assert represented == exact_objective(a, p) == exact_oracle(a)["objective"]


def test_finite_sentinel_has_a_bound_and_rejects_unrepresentable_domain():
    a = np.array([[-3., 2., np.inf]])
    sentinel = finite_sentinel(a, -4, order=4)
    assert math.log2(sentinel).is_integer()
    assert sentinel > 2*4*4
    with pytest.raises(UnsupportedCostDomain, match="overflow|representable"):
        finite_sentinel(np.array([[np.finfo(float).max]]), 1)


@pytest.mark.parametrize("value", [np.nan, -np.inf])
def test_nonfinite_input_rejected_in_preflight_before_solver(value):
    a = np.array([[value]])
    with pytest.raises(UnsupportedCostDomain):
        validate_input(case(a), a)


def test_nonrepresentable_endpoint_penalty_is_explicitly_unsupported():
    a = np.array([[0.]])
    limit = np.nextafter(0., 1.)
    status = method_status("scipy:capped", case(a, limit))
    assert status["status"] == "unsupported" and "L/2" in status["reason"]


def test_noncontiguous_input_and_transpose_are_immutable():
    backing = np.arange(24, dtype=np.float64).reshape(4, 6)
    a = backing[:, ::2]
    before = backing.copy()
    for x in (a, a.T):
        spec = case(x, 3, 2)
        p = prepare_method("scipy:diagonal_dummies", spec)(x)
        validate_output(spec, x, p, exact_oracle(x, 3, 2), exact=True)
    np.testing.assert_array_equal(backing, before)


@pytest.mark.parametrize("method", ["loop:scipy", "loop:jvrust", "batch:jvrust:1", "batch:jvcpp:1",
                                   "batch:jvhybrid:1", "batch:jvx:1"])
def test_batch_output_contract_and_existing_3d_input(method):
    costs = np.array([[[0., 2., 1.], [1., 0., 2.]], [[2., 0., 1.], [0., 1., 2.]]])
    spec = case(costs, 1, .5, kind="batch")
    status = method_status(method, spec)
    if status["status"] == "unsupported":
        pytest.skip(status["reason"])
    call = prepare_method(method, spec)
    before = costs.copy()
    for data in (costs, list(costs)):
        out = call(data)
        result = validate_output(spec, data, out, [exact_oracle(a, 1, .5) for a in costs], exact=True)
        assert result["count"] == 2
    np.testing.assert_array_equal(costs, before)
    if method.startswith("batch:jvx"):
        assert call.method_info["pool_lifetime"] == "per_call"
    with pytest.raises(ValidationFailure, match="count"):
        validate_output(spec, costs, out[:1])


def test_frozen_input_domain_scan_is_outside_the_timed_adapter(monkeypatch):
    a = np.array([[1., 0.], [0., 1.]])
    spec = case(a, 1)
    validate_input(spec, a)
    call = prepare_method("scipy:capped", spec)
    def unexpected(*args, **kwargs):
        raise AssertionError("Generic domain scan was repeated inside complete call")
    monkeypatch.setattr(adapters.np, "isnan", unexpected)
    monkeypatch.setattr(adapters.np, "isneginf", unexpected)
    assert call(a).shape == (2, 2)


def test_batch_fixture_registry_and_float32_representation_identity():
    assert len(batch_fixtures()) == len(fixtures()) - 2
    assert all(x["cost"].ndim == 3 and len(x["cost"]) == 4 for x in batch_fixtures())
    a = np.array([[2**24 + 1, .25]], dtype=np.float32)
    report = validate_input(case(a), a)
    assert report["inputs"][0]["storage_dtype"] == "float32"
    assert report["inputs"][0]["represented_dtype"] == "float64"
    # Conversion cannot recover the integer already rounded in float32 storage.
    assert Fraction.from_float(float(a[0, 0])) == 2**24


def test_method_selection_and_imports_happen_before_complete_call(monkeypatch):
    a = np.array([[1., 0.], [0., 1.]])
    call = prepare_method("scipy:capped", case(a, 1))
    def unexpected(*args, **kwargs):
        raise AssertionError("Import/selection occurred inside measured callable")
    monkeypatch.setattr(adapters.importlib, "import_module", unexpected)
    assert call(a).shape == (2, 2)


def test_output_dtype_uniqueness_bounds_and_required_cardinality_are_strict():
    a = np.zeros((2, 2))
    invalid = [np.array([[0, 0]], dtype=np.int32), np.array([[0, 0], [0, 1]], dtype=np.int64),
               np.array([[0, 0], [1, 2]], dtype=np.int64), np.array([[0, 0]], dtype=np.int64)]
    for output in invalid:
        with pytest.raises(ValidationFailure):
            validate_output(case(a), a, output)


def test_precision_boundary_classification_is_tagged_and_keeps_exact_gap(monkeypatch):
    a = np.array([[np.nextafter(1., 0.), np.nextafter(1., np.inf)]])
    record = {"case": case(a, 1), "cost": a, "numerical_boundary": True}
    monkeypatch.setattr(validation, "prepare_method", lambda *args: lambda cost: np.empty((0, 2), dtype=np.int64))
    result = validate_method("synthetic", [record])
    assert result["status"] == "passed_with_numerical_limitations"
    assert result["results"][0]["exact_gap"] == "1/9007199254740992"
    result = validate_method("synthetic", [{**record, "numerical_boundary": False}])
    assert result["status"] == "correctness_failed"


def test_small_delta_does_not_excuse_forbidden_output(monkeypatch):
    a = np.array([[np.nextafter(1., 0.)]])
    record = {"case": case(a, 1, 0), "cost": a, "numerical_boundary": True}
    monkeypatch.setattr(validation, "prepare_method", lambda *args: lambda cost: np.array([[0, 0]], dtype=np.int64))
    assert validate_method("synthetic", [record])["status"] == "correctness_failed"


def test_available_methods_pass_strict_fixtures_and_report_boundary_limits():
    result = validate_methods()
    assert not result["correctness_failed"], [x for m in result["methods"] for x in m["results"]
                                              if x["status"] == "correctness_failed"]
    assert "scipy:capped" in result["passed"]
    assert len(result["methods"]) == 63
    for method in result["methods"]:
        for item in method["results"]:
            if item["status"] == "numerical_boundary_limitation":
                assert item["case"].startswith("threshold_adjacent")
                assert "exact_gap" in item


def test_missing_backend_and_invalid_combination_remain_unsupported(monkeypatch):
    a = np.zeros((2, 3))
    assert method_status("cost:scipy:fill_split", case(a))["status"] == "unsupported"
    assert method_status("shape:scipy:smallest_fill_zero", case(a, 1))["status"] == "unsupported"
    assert method_status("policy:rust:reference", case(a))["status"] == "unsupported"
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("deliberately missing backend")
    monkeypatch.setattr(adapters.importlib, "import_module", missing)
    assert method_status("jvx", case(a))["status"] == "unsupported"


def test_exact_random_dyadic_small_inputs_across_all_layouts():
    rng = np.random.default_rng(20261030)
    for shape in ((1, 3), (2, 3), (3, 2)):
        a = rng.integers(-4, 9, size=shape).astype(np.float64) / 4
        a[0, -1] = np.inf
        for limit in (-.25, .5, 1.0):
            oracle = exact_oracle(a, limit)
            for layout in COST_LAYOUTS:
                spec = case(a, limit)
                output = prepare_method("cost:scipy:"+layout, spec)(a)
                validate_output(spec, a, output, oracle, exact=True)
        oracle = exact_oracle(a)
        for layout in SHAPE_LAYOUTS:
            spec = case(a)
            output = prepare_method("shape:scipy:"+layout, spec)(a)
            validate_output(spec, a, output, oracle, exact=True)
