"""Recipe invariants plus opt-in checks of the actual controlled extensions.

Set CAMPAIGN_NATIVE_BUILD_INDEX to a generated build-index.json to exercise all
available native arms. These are correctness checks, not benchmark measurements.
"""
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from . import native_build as native


def test_policy_catalog_excludes_only_duplicate_optional_cutoffs():
    assert len(native.eligible_arms(False)) == 8
    assert len(native.eligible_arms(True)) == 6
    assert set(native.eligible_arms(False)) - set(native.eligible_arms(True)) == {
        "cutoff_off", "ties_cutoff_off"}
    assert native.policy_for("no_reductions")["reduction_passes"] == 0
    assert native.policy_for("no_reductions")["budget"] == 4
    assert native.policy_for("budget8")["budget"] == 8
    assert native.policy_for("fresh_state")["workspace"] == 1
    assert native.policy_for("fresh_worker")["workspace"] == 2


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/escape", "file"),
                                       ("link", "symlink"), ("hard", "hardlink")])
def test_archive_rejects_escape_and_link_before_extraction(tmp_path, name, kind):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        ordinary = tarfile.TarInfo("ordinary")
        stream.addfile(ordinary, io.BytesIO())
        bad = tarfile.TarInfo(name)
        if kind != "file":
            bad.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
            bad.linkname = "../elsewhere"
        stream.addfile(bad, io.BytesIO())
    with pytest.raises(native.NativeBuildError, match="Unsafe"):
        native._safe_extract(archive.getvalue(), tmp_path)
    assert not (tmp_path / "ordinary").exists()


def test_build_resolution_aliases_and_uniqueness():
    index = {"builds": [{"language": "rust", "arm_id": "reference"}]}
    assert native.resolve_build(index, "rust", "w0")["arm_id"] == "reference"
    with pytest.raises(native.NativeBuildError):
        native.resolve_build(index, "cpp", "reference")
    index["builds"].append(index["builds"][0])
    with pytest.raises(native.NativeBuildError):
        native.resolve_build(index, "rust", "reference")


def test_loader_rejects_instrumented_build_before_import():
    with pytest.raises(native.NativeBuildError, match="uninstrumented"):
        native.load_solver({"profile_enabled": True})


@pytest.fixture(scope="module")
def native_solvers():
    path = os.environ.get("CAMPAIGN_NATIVE_BUILD_INDEX")
    if not path:
        pytest.skip("Set CAMPAIGN_NATIVE_BUILD_INDEX to validate built extensions")
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    # All variants intentionally load into this one process.
    return [(record, native.load_solver(record)) for record in index["builds"]]


def _objective(cost, pairs, limit):
    import numpy as np
    pairs = np.asarray(pairs)
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    assert np.issubdtype(pairs.dtype, np.integer)
    assert len(set(pairs[:, 0])) == len(pairs)
    assert len(set(pairs[:, 1])) == len(pairs)
    assert np.all(pairs >= 0)
    assert np.all(pairs[:, 0] < cost.shape[0])
    assert np.all(pairs[:, 1] < cost.shape[1])
    chosen = cost[pairs[:, 0], pairs[:, 1]]
    assert np.all(np.isfinite(chosen))
    if np.isinf(limit):
        assert len(pairs) == min(cost.shape)
        return float(chosen.sum())
    return float(chosen.sum() + (sum(cost.shape) - 2 * len(pairs)) * limit / 2)


def _small_oracle(cost, limit):
    """Exhaust all injective partial assignments; use exact dyadic test inputs."""
    import numpy as np
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
    rows, cols = cost.shape
    best = float("inf")

    def visit(row, occupied, value, count):
        nonlocal best
        if row == rows:
            if np.isinf(limit):
                if count == rows:
                    best = min(best, value)
            else:
                best = min(best, value + (rows + cols - 2 * count) * limit / 2)
            return
        if np.isfinite(limit):
            visit(row + 1, occupied, value, count)
        for col in range(cols):
            if not (occupied & (1 << col)) and np.isfinite(cost[row, col]):
                visit(row + 1, occupied | (1 << col), value + float(cost[row, col]), count + 1)

    visit(0, 0, 0.0, 0)
    return best


def test_native_variants_match_enumerated_objective(native_solvers):
    import numpy as np
    rng = np.random.default_rng(49219)
    cases = [np.array([[1.]]), np.array([[0., .5], [.5, np.inf]]),
             np.array([[0., 1.], [1., np.inf]])]
    for shape in [(2, 3), (3, 2), (4, 4), (4, 5), (5, 4)]:
        for _ in range(4):
            cost = rng.integers(-4, 13, size=shape).astype(float) / 4
            cost[rng.random(shape) < .2] = np.inf
            np.fill_diagonal(cost, 0.)  # Guarantee required feasibility.
            cases.append(cost)
    for cost in cases:
        for limit in [float("inf"), .5, 1., 2.]:
            expected = _small_oracle(cost, limit)
            for record, solve in native_solvers:
                pairs = solve(cost, limit=limit)
                assert _objective(cost, pairs, limit) == expected, (
                    record["language"], record["arm_id"], cost, limit, pairs, expected)


def test_native_batch_parallel_and_orientation_objectives(native_solvers):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(24612)
    # Exactly 65,536 values crosses both libraries' parallel-dispatch threshold.
    wide = rng.integers(0, 17, size=(32, 8, 256)).astype(float) / 16
    for costs in [wide, np.ascontiguousarray(wide.transpose(0, 2, 1))]:
        for limit in [float("inf"), .5]:
            expected = []
            for cost in costs:
                cap = cost if np.isinf(limit) else np.minimum(cost, limit)
                row, col = linear_sum_assignment(cap)
                pairs = np.column_stack((row, col))
                if np.isfinite(limit):
                    pairs = pairs[cost[row, col] < limit]
                expected.append(_objective(cost, pairs, limit))
            for record, solve in native_solvers:
                result = solve(costs, limit=limit, batch=True, workers=4)
                assert len(result) == len(costs)
                assert [_objective(cost, pairs, limit) for cost, pairs in zip(costs, result)] == expected, (
                    record["language"], record["arm_id"])
