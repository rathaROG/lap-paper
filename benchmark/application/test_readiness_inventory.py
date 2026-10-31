"""Readiness coverage checks with solver and measurement calls replaced."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import numpy as np
import pytest

from . import realbench_campaign, realbench_report, realbench_tracking, run


def mock_corpora(tmp_path, monkeypatch, *, matrices=7539, missing_corpus=False):
    names = [f"{sequence}_{source}_{tracker}"
             for sequence in run.SEQUENCES for source in run.SOURCES for tracker in run.TRACKERS]
    if missing_corpus:
        names.pop()
    directory = tmp_path / "collection" / "matrices"
    directory.mkdir(parents=True)
    base, remainder = divmod(matrices, len(names))
    counts = {name: base + (index < remainder) for index, name in enumerate(names)}
    for name in names:
        (directory / (name + ".npz")).touch()

    class Store:
        def __init__(self, path):
            self.files = list(range(counts[path.stem]))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            return np.array([[0., 1.]])

    monkeypatch.setattr(run, "collection_path", lambda: directory.parent)
    monkeypatch.setattr(np, "load", lambda path, **kwargs: Store(path))
    monkeypatch.setattr(realbench_tracking, "solve_pairs",
                        lambda *args: np.array([[0, 0]], dtype=np.int32))


def test_validation_accepts_current_solver_inventory_and_checks_the_alias(tmp_path, monkeypatch):
    mock_corpora(tmp_path, monkeypatch)
    run.validate_corpora(tmp_path)
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["passed"]
    assert report["matrices"] == 7539
    assert report["checks"] == 7539 * 2 * (len(run.SOLVERS) + 1)
    assert {row["solver"] for row in report["results"]} == set(run.SOLVERS) | {"lapjvr"}


@pytest.mark.parametrize("missing_corpus", [False, True])
def test_validation_still_rejects_incomplete_frozen_coverage(tmp_path, monkeypatch, missing_corpus):
    # A missing corpus cannot pass by redistributing all 7,539 matrices into
    # the remaining files; all eight corpora and the full total are required.
    mock_corpora(tmp_path, monkeypatch, matrices=7539 if missing_corpus else 7538,
                 missing_corpus=missing_corpus)
    with pytest.raises(AssertionError, match="Frozen corpus coverage changed"):
        run.validate_corpora(tmp_path)
    assert not (tmp_path / "validation.json").exists()


@pytest.mark.parametrize("missing_pipeline", [False, True])
def test_smoke_checks_current_inventory_without_measurements(tmp_path, monkeypatch, missing_pipeline):
    cases = [{"case": f"{sequence}_{source}_{tracker}", "sequence": sequence,
              "source": source, "tracker": tracker}
             for sequence in run.SEQUENCES for source in run.SOURCES for tracker in run.TRACKERS]
    if missing_pipeline:
        cases.pop()
    caches = {f"{sequence}_{source}": {}
              for sequence in run.SEQUENCES for source in run.SOURCES}
    fingerprint = realbench_campaign.prediction_hash([])

    def tracking(options, output):
        run.write_json(output / "summary.json", {"tracking": [
            {"case": case["case"], "solver": solver, "prediction_sha256": fingerprint}
            for case in cases for solver in options.solvers]})

    def replay(options, output):
        run.write_json(output / "summary.json", {"replay": [
            {"objective_mismatch_count": 0}
            for case in cases for precision in options.precisions for solver in options.solvers]})

    monkeypatch.setattr(run, "tracker_stage", tracking)
    monkeypatch.setattr(run, "replay_stage", replay)
    monkeypatch.setattr(run, "load_cases", lambda options: (caches, cases))
    monkeypatch.setattr(run, "cpu_session", lambda options: nullcontext())
    monkeypatch.setattr(realbench_tracking, "run_tracking", lambda *args, **kwargs: {"predictions": []})
    monkeypatch.setattr(realbench_report, "generate_report", lambda output: None)
    if missing_pipeline:
        with pytest.raises(AssertionError):
            run.smoke_stage(SimpleNamespace(), tmp_path)
        assert not (tmp_path / "summary.json").exists()
    else:
        run.smoke_stage(SimpleNamespace(), tmp_path)
        summary = json.loads((tmp_path / "summary.json").read_text())
        assert summary["status"] == "complete"
        assert summary["tracking_conditions"] == summary["profile_conditions"] == 64
        assert summary["replay_conditions"] == 128
        assert summary["captured_pipelines"] == 8
        assert summary["objective_mismatches"] == 0
