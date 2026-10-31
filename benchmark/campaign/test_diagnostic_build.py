"""Separate diagnostic correctness/counter checks, never primary samples."""
import json
import os
from pathlib import Path

import pytest

from .diagnostic_build import COUNTERS, load_diagnostic
from .native_build import NativeBuildError, load_solver


def test_primary_loader_rejects_diagnostic_and_diagnostic_rejects_primary():
    with pytest.raises(NativeBuildError, match="uninstrumented"):
        load_solver({"profile_enabled": True, "diagnostic": True})
    with pytest.raises(NativeBuildError, match="instrumented"):
        load_diagnostic({"profile_enabled": False, "diagnostic": False})


@pytest.fixture(scope="module")
def diagnostics():
    path = os.environ.get("CAMPAIGN_DIAGNOSTIC_BUILD_INDEX")
    if not path:
        pytest.skip("Set CAMPAIGN_DIAGNOSTIC_BUILD_INDEX to exercise diagnostic artifacts")
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(record["language"], record["arm_id"]): load_diagnostic(record) for record in index["builds"]}


def test_all_diagnostics_preserve_objective_and_reset_counters(diagnostics):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(21812)
    for key, diagnostic in diagnostics.items():
        for shape in [(4, 7), (7, 4), (4, 4)]:
            cost = rng.integers(-3, 8, shape).astype(float) / 4
            for limit in [float("inf"), .5]:
                diagnostic.reset()
                assert set(diagnostic.snapshot()) == set(COUNTERS)
                assert all(x == 0 for x in diagnostic.snapshot().values())
                pairs = diagnostic.solve(cost, limit=limit)
                capped = cost if np.isinf(limit) else np.minimum(cost, limit)
                row, col = linear_sum_assignment(capped)
                expected = float(capped[row, col].sum())
                if np.isfinite(limit):
                    expected += (max(shape) - min(shape)) * limit / 2
                value = float(cost[pairs[:, 0], pairs[:, 1]].sum())
                if np.isfinite(limit):
                    value += (sum(shape) - 2 * len(pairs)) * limit / 2
                assert value == expected, (key, shape, limit)
                snapshot = diagnostic.snapshot()
                assert snapshot["problems"] == 1
                if key[1] == "no_reductions" and (shape[0] != shape[1] or np.isfinite(limit)):
                    assert snapshot["initialization_row_scans"] == 0
                    assert snapshot["initialization_chain_starts"] == 0
                    assert snapshot["augmentations"] == min(shape)


def test_initializer_and_no_reduction_counts_have_defined_meaning(diagnostics):
    import numpy as np
    for language in ("rust", "cpp"):
        reference = diagnostics[language, "reference"]
        reference.reset()
        reference.solve(np.ones((3, 5)), limit=.5)
        counts = reference.snapshot()
        assert counts["initialization_row_scans"] == 3
        assert counts["initialization_chain_starts"] == 3
        assert counts["rejection_endings"] == 3
        assert counts["augmentations"] == 0
        direct = diagnostics[language, "no_reductions"]
        direct.reset()
        direct.solve(np.ones((3, 5)), limit=.5)
        counts = direct.snapshot()
        assert counts["augmentations"] == 3
        assert counts["search_row_scans"] == 3
        assert counts["rejection_endings"] == 3


def test_workspace_arms_measure_actual_capacity_and_drop_lifetimes(diagnostics):
    import numpy as np
    for language in ("rust", "cpp"):
        for arm in ("reference", "fresh_state", "fresh_worker"):
            diagnostic = diagnostics[language, arm]
            diagnostic.reset()
            cost = np.zeros((16, 4))
            diagnostic.solve(cost, limit=.5)
            diagnostic.solve(cost, limit=.5)
            counts = diagnostic.snapshot()
            assert counts["problems"] == 2
            assert counts["orientation_capacity_after_solve_max_bytes"] >= cost.nbytes
            if arm == "fresh_worker":
                assert counts["orientation_capacity_before_solve_max_bytes"] == 0
                assert counts["orientation_capacity_after_cleanup_max_bytes"] == 0
                assert counts["orientation_releases"] == 2
                assert counts["fresh_worker_drops"] == 2
            else:
                assert counts["orientation_capacity_before_solve_max_bytes"] >= cost.nbytes
                assert counts["orientation_capacity_after_cleanup_max_bytes"] >= cost.nbytes
                assert counts["orientation_nonzero_before_solve_count"] >= 1
                assert counts["fresh_state_drops"] == (2 if arm == "fresh_state" else 0)
