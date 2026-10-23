"""Regressions for batch correctness reporting; native solvers are not required."""

from types import SimpleNamespace

import numpy as np
import pytest

import check_artifact as audit
import run_experiments as experiment


@pytest.fixture
def batch_harness(monkeypatch):
    pairs = np.array([[0, 0], [1, 1]], dtype=np.int64)
    costs = np.array([[0.0, 10.0, 10.0], [10.0, 0.0, 10.0]])
    monkeypatch.setattr(experiment, "np", np)
    monkeypatch.setattr(experiment, "batch_specs", lambda: [
        {"case": "test_batch", "b": 2, "n": 2, "m": 3, "limit": float("inf")}
    ])
    monkeypatch.setattr(experiment, "make_cost", lambda spec, rng: costs.copy())
    monkeypatch.setattr(experiment, "solve", lambda *args: pairs.copy())
    monkeypatch.setattr(experiment, "batch_call", lambda name, costs, limit: [
        pairs.copy() for _ in costs
    ])
    return SimpleNamespace(seed=1, rounds=2, block_seconds=0.01), pairs


@pytest.mark.parametrize("validation_only", [False, True])
@pytest.mark.parametrize("output_count", [0, 1, 3])
def test_incomplete_or_extra_outputs_fail_before_timing(
    monkeypatch, batch_harness, validation_only, output_count
):
    options, pairs = batch_harness
    monkeypatch.setattr(experiment, "batch_call", lambda *args, **kwargs: [
        pairs.copy() for _ in range(output_count)
    ])
    monkeypatch.setattr(experiment, "timed_block", lambda *args: pytest.fail(
        "Invalid batches must fail before calibration or timing"
    ))
    with pytest.raises(AssertionError, match="batch length mismatch"):
        experiment.batch_experiment(options, 4, validation_only=validation_only)


def test_incomplete_reference_fails(batch_harness):
    _, pairs = batch_harness
    costs = np.zeros((2, 2, 3))
    with pytest.raises(AssertionError, match="references=1"):
        experiment.check_batch(costs, [pairs, pairs], [0.0], float("inf"))


def test_last_output_objective_is_checked(monkeypatch, batch_harness):
    options, pairs = batch_harness
    wrong = np.array([[0, 1], [1, 0]], dtype=np.int64)
    monkeypatch.setattr(experiment, "batch_call", lambda *args, **kwargs: [pairs, wrong])
    with pytest.raises(AssertionError):
        experiment.batch_experiment(options, 4, validation_only=True)


@pytest.mark.parametrize("validation_only", [False, True])
def test_valid_batches_count_every_checked_output(monkeypatch, batch_harness, validation_only):
    options, _ = batch_harness
    timing_calls = []

    def timed(*args):
        assert not validation_only, "Validation-only mode must never time or calibrate"
        timing_calls.append(args)
        return 1000.0

    monkeypatch.setattr(experiment, "timed_block", timed)
    rows, checks = experiment.batch_experiment(options, 4, validation_only=validation_only)
    assert len(rows) == (0 if validation_only else 20)
    assert len(timing_calls) == (0 if validation_only else 30)
    assert len(checks) == 1
    assert checks[0]["problems"] == 2
    assert checks[0]["solvers"] == 10
    assert checks[0]["objective_checks_passed"] is True
    assert checks[0]["checked_outputs"] == {
        name: 2 for name in experiment.batch_names(4)
    }


def test_measurement_provenance_requires_matching_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "PAPER", tmp_path)
    current = tmp_path / "run_experiments.py"
    current.write_bytes(b"original\r\n")
    original_hash = audit.digest(current)
    assert audit.measurement_script(current.name, original_hash) == current

    snapshot = tmp_path / "archive" / "measurement-scripts" / current.name
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(current.read_bytes())
    current.write_bytes(b"corrected\r\n")
    assert audit.measurement_script(current.name, original_hash) == snapshot
    assert audit.measurement_script(current.name, audit.digest(current)) == current

    snapshot.write_bytes(b"modified historical source\r\n")
    with pytest.raises(AssertionError, match="No preserved measurement script matches"):
        audit.measurement_script(current.name, original_hash)
