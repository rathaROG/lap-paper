"""Exercise the full collection flow on tiny fixtures, with measured kernels mocked."""
from types import SimpleNamespace
import json

import pytest

from . import realbench_campaign as campaign
from . import realbench_evaluate, realbench_report
from .test_realbench_campaign import make_inputs


@pytest.mark.parametrize("profile", ["full", "light"])
def test_collect_runs_primary_profile_capture_replay_and_evaluation(tmp_path, monkeypatch, profile):
    inputs, output = tmp_path / "inputs", tmp_path / "run"
    make_inputs(inputs, ["gt"])
    calls = []
    monkeypatch.setattr(campaign, "environment", lambda directory: {"fixture": True})

    def tracking(cache, solver, tracker, *, mode="timed", **kwargs):
        calls.append((solver, tracker, mode))
        return {"predictions": [{"frame": 1, "tracks": []}, {"frame": 2, "tracks": []}],
                "tracker_total_ns": 1000, "tracker_ms_per_frame": .0005,
                "solver_calls": None if mode == "timed" else 1, "solver_total_ns": 100}

    def replay(directory, case, capture, options, rng):
        return ([{**case, "solver": solver, "precision": precision, "median_ns_per_assignment": 1}
                 for solver in campaign.SOLVERS for precision in ("float32", "float64")], {"passed": True})

    monkeypatch.setattr(campaign, "run_tracking", tracking)
    monkeypatch.setattr(campaign, "replay_case", replay)
    monkeypatch.setattr(realbench_evaluate, "evaluate", lambda *args: {
        "idf1": 1, "mota": 1, "num_switches": 0, "num_false_positives": 0, "num_misses": 0})
    monkeypatch.setattr(realbench_report, "generate_report", lambda directory: (directory / "report.html").write_text("fixture"))
    campaign.collect(SimpleNamespace(output=output, inputs=inputs, sources=["gt"], frames=2,
                                     rounds=1, seed=1, block_seconds=.001, cpu=0, warmup_frames=1, profile=profile))
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["tracking"]) == 32
    assert summary["protocol"]["profile"] == profile
    assert len(summary["replay"]) == 64
    assert sum(mode == "timed" for _, _, mode in calls) == 32
    assert sum(mode == "profile" for _, _, mode in calls) == 32
    assert sum(mode == "capture" for _, _, mode in calls) == 4
    assert summary["reproducibility"]["all_instrumented_predictions_match"]
    assert (output / "report.html").is_file()
    inventory = json.loads((output / "file_manifest.json").read_text())
    for name, identity in inventory.items():
        assert campaign.sha256_file(output / name) == identity["sha256"]
