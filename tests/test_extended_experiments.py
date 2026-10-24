"""Check pairing, feasibility guarantees, and independent edge-case validation."""
import math
import numpy as np
import pytest

import run_extended_experiments as extension


@pytest.fixture(scope="module", autouse=True)
def runtime():
    extension.base.initialize()


def spec(**overrides):
    value = dict(n=8, m=12, input_key="paired", distribution="uniform", limit=.5)
    value.update(overrides)
    return value


def test_threshold_sweep_reuses_identical_inputs():
    a, _ = extension.make_input(spec(limit=.001), 42, 0)
    b, _ = extension.make_input(spec(limit=4.), 42, 0)
    np.testing.assert_array_equal(a, b)


def test_orientation_pair_is_exact_transpose():
    a, canonical_a = extension.make_input(spec(), 42, 0)
    b, canonical_b = extension.make_input(spec(n=12, m=8), 42, 0)
    np.testing.assert_array_equal(a.T, b)
    assert b.flags.c_contiguous and canonical_a == canonical_b


def test_forbidden_sweep_preserves_allowed_costs_and_required_feasibility():
    a, _ = extension.make_input(spec(forbidden=.5, planted_feasible=True), 42, 0)
    b, _ = extension.make_input(spec(forbidden=.99, planted_feasible=True), 42, 0)
    assert np.all(np.isinf(b)[np.isinf(a)])
    assert np.all(np.isfinite(np.diag(b)))
    np.testing.assert_array_equal(a[np.isfinite(b)], b[np.isfinite(b)])
    for method in extension.REQUIRED:
        pairs = extension.solve(method, b, math.inf)
        extension.base.check_pairs(b, pairs, math.inf)


def test_equal_objectives_accept_distinct_tied_matchings():
    cost = np.ones((3, 4))
    for pairs in (np.array([[0,0],[1,1],[2,2]], dtype=np.int64),
                  np.array([[0,1],[1,2],[2,3]], dtype=np.int64)):
        assert extension.checked_objective(cost, pairs, math.inf, 3.) == 3.


def test_wrong_objective_is_rejected():
    with pytest.raises(AssertionError):
        extension.checked_objective(np.array([[0., 1.], [1., 0.]]),
                                    np.array([[0,1],[1,0]], dtype=np.int64), math.inf, 0.)


def test_edge_cases_against_independent_enumeration():
    report = extension.validate_edges()
    assert report["status"] == "passed"
    assert {row["status"] for row in report["checks"]} == {"passed", "expected_infeasible"}
