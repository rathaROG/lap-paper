"""Run guards, output ownership and placement checks without benchmark timing."""
from contextlib import nullcontext
import json

import pytest

from . import run


@pytest.mark.parametrize("mode", ["campaign", "tracker-only", "replay-only", "regenerate-inputs"])
def test_measurement_requires_opt_in_before_creating_output(tmp_path, mode):
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        run.main(["--" + mode, "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("options", [["--cases", "mot16_04_gt_centroid"],
                                    ["--precisions", "float32"], ["--limit-matrices", "2"]])
def test_full_campaign_rejects_ignored_subset_options(tmp_path, options):
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        run.main(["--campaign", "--allow-full-campaign", "--output", str(output), *options])
    assert not output.exists()


@pytest.mark.parametrize("profile,rounds", [("full", 9), ("light", 3)])
def test_campaign_profile_selects_rounds_and_records_identity(tmp_path, monkeypatch, profile, rounds):
    import platform
    from . import realbench_campaign, realbench_paths
    from benchmark.campaign import resources, runner

    monkeypatch.setattr(platform, "python_version", lambda: "3.11.18")
    monkeypatch.setattr(resources, "HostLock", nullcontext)
    monkeypatch.setattr(runner, "environment", lambda: {})
    monkeypatch.setattr(run, "verify_inputs", lambda: {})
    monkeypatch.setattr(run, "configure_pyppbox", lambda path: None)
    monkeypatch.setattr(realbench_paths, "configure_pyppbox", lambda path: None)
    monkeypatch.setattr(run, "cpu_session", lambda args: nullcontext({}))
    calls = []
    monkeypatch.setattr(realbench_campaign, "collect", lambda args: calls.append(args))
    output = tmp_path / "run"
    assert run.main(["--campaign", "--allow-full-campaign", "--profile", profile,
                     "--inputs", str(tmp_path / "inputs"), "--output", str(output)]) == 0
    assert calls[0].rounds == rounds and calls[0].profile == profile
    assert calls[0].frames == 999 and calls[0].precisions == ["float32", "float64"]
    record = json.loads((output / "run.json").read_text())
    assert record["arguments"]["profile"] == profile
    assert record["arguments"]["rounds"] == rounds


@pytest.mark.parametrize("options", [["--rounds", "9"], ["--cases", "mot16_04_gt_centroid"]])
def test_light_campaign_rejects_changes_to_frozen_protocol_before_output(tmp_path, options):
    output = tmp_path / "run"
    with pytest.raises(SystemExit):
        run.main(["--campaign", "--allow-full-campaign", "--profile", "light",
                  "--output", str(output), *options])
    assert not output.exists()


def test_nonempty_output_is_never_reused(tmp_path):
    sentinel = tmp_path / "owned.json"
    sentinel.write_bytes(b"existing result")
    with pytest.raises(FileExistsError, match="overwrite"):
        run.main(["--smoke", "--output", str(tmp_path)])
    assert sentinel.read_bytes() == b"existing result"
    assert list(tmp_path.iterdir()) == [sentinel]


def test_failed_smoke_retains_failure_and_never_claims_complete(tmp_path, monkeypatch):
    from benchmark.campaign import resources, runner
    monkeypatch.setattr(resources, "HostLock", nullcontext)
    monkeypatch.setattr(runner, "environment", lambda: {"python": "test"})
    monkeypatch.setattr(run, "verify_inputs", lambda: {"passed": True})

    def missing_backend(*args):
        raise ModuleNotFoundError("missing solver")

    monkeypatch.setattr(run, "smoke_stage", missing_backend)
    with pytest.raises(ModuleNotFoundError, match="missing solver"):
        run.main(["--smoke", "--output", str(tmp_path)])
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["scope"] == "smoke_not_publication_evidence"
    assert report["error_type"] == "ModuleNotFoundError"
    assert report["source_sha256"]


def test_final_file_inventory_hashes_completed_wrapper_records(tmp_path, monkeypatch):
    from benchmark.campaign import resources, runner
    monkeypatch.setattr(resources, "HostLock", nullcontext)
    monkeypatch.setattr(runner, "environment", lambda: {"python": "test"})
    monkeypatch.setattr(run, "verify_inputs", lambda: {"passed": True})

    def partial_inventory(args, output):
        run.write_json(output / "file_manifest.json", {"run.json": {"sha256": "running-status"}})
        run.write_json(output / "placement-checks.json", {"before": [0], "after": [0]})

    monkeypatch.setattr(run, "smoke_stage", partial_inventory)
    run.main(["--smoke", "--output", str(tmp_path)])
    inventory = json.loads((tmp_path / "file_manifest.json").read_text())
    assert "placement-checks.json" in inventory
    assert json.loads((tmp_path / "run.json").read_text())["status"] == "complete"
    for name, identity in inventory.items():
        assert run.sha256(tmp_path / name) == identity["sha256"]
        assert (tmp_path / name).stat().st_size == identity["bytes"]


def test_affinity_drift_fails_and_restores_original_mask(monkeypatch):
    import psutil
    from types import SimpleNamespace
    from . import realbench_tracking

    current = {"mask": [0, 1]}

    def affinity(mask=None):
        if mask is not None:
            current["mask"] = mask
        return current["mask"]

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(cpu_affinity=affinity))
    monkeypatch.setattr(realbench_tracking, "solve_pairs", lambda *args: None)
    with pytest.raises(RuntimeError, match="affinity changed"):
        with run.cpu_session(SimpleNamespace(cpu=0, solvers=["scipy"])):
            current["mask"] = [1]
    assert current["mask"] == [0, 1]
