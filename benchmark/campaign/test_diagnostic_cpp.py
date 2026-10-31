"""Guard and placement checks for the separately compiled C++ diagnostics.

The optional source test reads an existing controlled snapshot; it never edits
that snapshot, compiles code, or invokes a solver. Runtime diagnostic tests live
with the diagnostic builder and explicitly identified diagnostic artifacts.
"""
import json
import os
from pathlib import Path

import pytest

from . import diagnostic_cpp as diagnostic


def test_diagnostic_counter_schema_and_c_abi_are_fixed():
    assert len(diagnostic.FIELD_NAMES) == 17
    assert len(set(diagnostic.FIELD_NAMES)) == 17
    assert diagnostic.FIELD_NAMES[0] == "initialization_row_scans"
    assert diagnostic.FIELD_NAMES[6] == "orientation_capacity_before_solve_max_bytes"
    assert diagnostic.FIELD_NAMES[16] == "fresh_worker_drops"
    assert "void campaign_diag_reset(void);" in diagnostic.HEADER
    assert "void campaign_diag_snapshot(unsigned long long* out);" in diagnostic.HEADER
    assert "#ifdef __cplusplus" in diagnostic.HEADER


def test_primary_namespace_is_rejected_without_modification(tmp_path):
    (tmp_path / "setup.py").write_text("Extension('campaign_cpp_reference._lapjv', [])")
    (tmp_path / "campaign_cpp_reference").mkdir()
    untouched = tmp_path / "solver.cpp"
    untouched.write_text("primary source sentinel")
    with pytest.raises(ValueError, match="isolated campaign_diag_cpp"):
        diagnostic.patch_cpp(tmp_path)
    assert untouched.read_text() == "primary source sentinel"
    assert not (tmp_path / "src/_lapjv/campaign_diag.h").exists()


def test_unexpected_source_rejected_before_any_write(tmp_path):
    (tmp_path / "setup.py").write_text("Extension('campaign_diag_cpp_reference._lapjv', [])")
    (tmp_path / "campaign_diag_cpp_reference").mkdir()
    source = tmp_path / "src/_lapjv/solver.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("unrecognized source")
    with pytest.raises(ValueError, match="source guard"):
        diagnostic.patch_cpp(tmp_path)
    assert source.read_text() == "unrecognized source"
    assert not (source.parent / "campaign_diag.h").exists()


def test_pinned_snapshot_instrumentation_preserves_required_square_and_primary_source():
    index_path = os.environ.get("CAMPAIGN_NATIVE_BUILD_INDEX")
    if not index_path:
        pytest.skip("Set CAMPAIGN_NATIVE_BUILD_INDEX for a read-only pinned-source patch check")
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    record = next(x for x in index["builds"] if x["language"] == "cpp" and x["arm_id"] == "reference")
    path = Path(record["source_copy"]) / "src/_lapjv/solver.cpp"
    before = path.read_text(encoding="utf-8")
    after = diagnostic.instrument_text(before)
    begin = "struct Square {"
    end = "template<class T> struct Rectangular {"
    assert before[before.index(begin):before.index(end)] == after[after.index(begin):after.index(end)]
    assert after.count("campaign_diag::add(0);") == 1
    assert after.count("campaign_diag::add(1);") == 1
    assert after.count("campaign_diag::add(2);") == 1
    assert after.count("campaign_diag::add(3);") == 1
    assert after.count("campaign_diag::add(4);") == 1
    assert after.count("campaign_diag::add(5);") == 3
    assert "DiagnosticSolveSample<T> diagnostic_sample(fresh, true);" in after
    assert "retained.swap(fresh); campaign_diag::add(15);" in after
    assert "value.orientation.capacity()*sizeof(T)" in after
    assert path.read_text(encoding="utf-8") == before
    with pytest.raises(ValueError, match="already instrumented"):
        diagnostic.instrument_text(after)


def test_cpp_diagnostic_events_count_retries_chains_and_cached_root_once():
    index_path = os.environ.get("CAMPAIGN_DIAGNOSTIC_BUILD_INDEX")
    if not index_path:
        pytest.skip("Set CAMPAIGN_DIAGNOSTIC_BUILD_INDEX for tiny counter-semantic checks")
    import numpy as np
    from .diagnostic_build import load_diagnostic
    from .validation import exact_objective, exact_oracle
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    records = {r["arm_id"]: r for r in index["builds"] if r["language"] == "cpp"}
    solvers = {arm: load_diagnostic(records[arm]) for arm in ("reference", "no_reductions")}
    examples = [
        # A displaced row is retried immediately: 3 scans, but only 2 chains.
        ("reference", [[0, 1, 100], [0, 3, 100]], None, [3, 0, 1, 2, 0, 0]),
        # Cached root labels do not double-count the two completion relaxations.
        ("no_reductions", [[0, 2, 3], [3, 0, 2]], None, [0, 2, 0, 0, 2, 0]),
        # The rejection can occur during initialization or completion.
        ("reference", [[2, 2, 2], [2, 2, 2]], 1., [2, 0, 0, 2, 0, 2]),
        ("no_reductions", [[2, 2, 2], [2, 2, 2]], 1., [0, 2, 0, 0, 2, 2]),
        # Deferred displacement starts another chain in the next pass.
        ("reference", [[0, 0, 2, 2]]*3, None, [4, 3, 2, 4, 1, 0]),
        # The distinct required-square initializer is outside rect-counter scope.
        ("reference", [[0, 1], [1, 0]], None, [0, 0, 0, 0, 0, 0]),
    ]
    for arm, values, limit, expected in examples:
        solver = solvers[arm]
        cost = np.array(values, dtype=np.float64)
        solver.reset()
        pairs = solver.solve(cost, limit=float("inf") if limit is None else limit)
        counters = solver.snapshot()
        assert [counters[name] for name in diagnostic.FIELD_NAMES[:6]] == expected
        assert counters["problems"] == 1
        assert exact_objective(cost, pairs, limit) == exact_oracle(cost, limit)["objective"]
