"""Protocol regressions: pairing, independent repeats, isolation, safe entry."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from .catalog import INPUT_COUNT, ROUNDS, build_manifest
from .test_failures import supervisor
from .runner import attempt_activity, smoke_plan, split_placements
from .worker import calibrated_repetitions, paired_schedule


def test_schedule_covers_complete_pairs_and_shared_input_order():
    methods = ["a", "b", "c"]
    schedule = paired_schedule(methods, INPUT_COUNT, ROUNDS, 123)
    assert len(schedule) == INPUT_COUNT * ROUNDS * 3
    assert len(set(schedule)) == len(schedule)
    for start in range(0, len(schedule), 3):
        block = schedule[start:start+3]
        assert len({(r, i) for r, i, m in block}) == 1
        assert {m for r, i, m in block} == set(methods)
    assert schedule == paired_schedule(methods, INPUT_COUNT, ROUNDS, 123)
    assert schedule != paired_schedule(methods, INPUT_COUNT, ROUNDS, 124)
    fewer = paired_schedule(methods[:2], INPUT_COUNT, ROUNDS, 123)
    assert [(r, i) for r, i, m in schedule[::3]] == [(r, i) for r, i, m in fewer[::2]]


def test_commands_without_output_use_the_minute_run_folder(tmp_path, monkeypatch, capsys):
    import argparse
    import datetime as dt
    import platform
    from .. import paths
    from . import __main__ as cli
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    folder = paths.run_folder(dt.datetime(2026, 10, 30, 15, 42))
    assert folder == tmp_path / f"{platform.system().lower()}-py{platform.python_version()}-20261030-1542"
    assert cli.main(["manifest"]) == 0
    written = list(tmp_path.glob("*/campaign-plan.json"))
    assert len(written) == 1 and f"Output: {written[0]}" in capsys.readouterr().out
    smoke = tmp_path / "earlier" / "campaign-smoke"
    estimate = cli.default_output(argparse.Namespace(command="estimate", smoke_root=smoke))
    assert estimate == smoke.resolve().parent / "campaign-estimate"


def test_calibration_preserves_slow_single_calls_and_bounds_growth():
    assert calibrated_repetitions(.2, 1, .03) == 1
    assert calibrated_repetitions(.000001, 1, .03) == 8
    assert calibrated_repetitions(.015, 100, .03) == 200
    assert calibrated_repetitions(0, 1_000_000, .03) == 1_000_000


def test_placement_groups_do_not_run_serial_on_four_cores():
    methods = ["loop:scipy", "workspace:rust:w0:1", "workspace:rust:w0:4", "batch:jvx:4", "scipy"]
    groups = dict(split_placements(methods))
    assert groups[1] == ["loop:scipy", "workspace:rust:w0:1", "scipy"]
    assert groups[4] == ["workspace:rust:w0:4", "batch:jvx:4"]


def test_smoke_is_small_prespecified_subset_not_campaign():
    manifest = build_manifest()
    smoke = smoke_plan(manifest)
    assert len(smoke) == 13
    assert {"sparse:csgraph", "sparse:lapmod"} <= {m for c in smoke for m in c["methods"]}
    originals = {c["id"]: c for c in manifest["cases"]}
    assert len(smoke) < len(originals) // 10
    assert all(set(c["methods"]) <= set(originals[c["id"]]["methods"]) for c in smoke)
    assert any(c["gate"] is not None and c["gate"] < c["limit"] for c in smoke)
    assert {c["metadata"].get("orientation_payload_bytes") for c in smoke} >= {8*1024**2, 16*1024**2}


def test_failed_attempt_activity_is_carried_without_double_counting(tmp_path):
    rows = [{"record_type": "phase", "method": "a", "elapsed_seconds": .4},
            {"record_type": "timing", "method": "a", "elapsed_seconds": .4},
            {"record_type": "phase", "method": "b", "elapsed_seconds": .2}]
    (tmp_path / "measurements.jsonl").write_text("\n".join(json.dumps(x) for x in rows))
    assert attempt_activity(tmp_path, {"status": "exited"}) == {"a": .4, "b": .2}
    result = attempt_activity(tmp_path, {"status": "terminated", "last_state": {
        "method": "b", "phase": "primary", "monotonic": time.monotonic() - .1}})
    assert result["a"] == .4 and result["b"] >= .3 - 1e-9


def test_atomic_progress_write_retries_windows_reader_sharing(tmp_path, monkeypatch):
    from . import common
    original = common.os.replace
    calls = []
    def briefly_locked(source, target):
        calls.append(target)
        if len(calls) < 3:
            raise PermissionError("Windows reader has not yet released delete sharing")
        return original(source, target)
    monkeypatch.setattr(common.os, "replace", briefly_locked)
    monkeypatch.setattr(common.time, "sleep", lambda seconds: None)
    common.write_json(tmp_path / "state.json", {"phase": "primary"})
    assert common.read_json(tmp_path / "state.json") == {"phase": "primary"}
    assert len(calls) == 3


def test_full_run_guard_rejects_before_preparing_any_directory(tmp_path):
    target = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, "-m", "benchmark.campaign", "run",
                             "--output", str(target), "--build-index", "nonexistent.json"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "--allow-full-campaign" in result.stderr
    assert not target.exists()


@pytest.mark.parametrize("profile,rounds", [("full", 5), ("light", 3)])
def test_primary_worker_requests_follow_selected_manifest(supervisor, monkeypatch, profile, rounds):
    from . import runner
    from .test_failures import completed
    environment = runner.environment()
    monkeypatch.setattr(runner, "environment", lambda: {**environment, "python": "3.11.18"})
    monkeypatch.setattr(runner, "admission", lambda case, methods, input_count: {
        "host_status": "ready", "methods": {m: {"status": "admitted"} for m in methods}})
    requests = []

    def launch(request, module):
        requests.append(request)
        result = completed(methods=request["methods"])
        return result["result"], result["watchdog"]

    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "attempt_activity", lambda *args: {})
    manifest = build_manifest(profile=profile)
    case = manifest["cases"][0]
    summary = runner.run_session(manifest, "virtual-primary", smoke=False, selected_cases=[case])
    assert summary["status"] == "finished"
    assert requests[0]["rounds"] == rounds and requests[0]["inputs"] == 5
    assert requests[0]["memory_trials"] == 5 and requests[0]["profile"] == profile
    assert requests[0]["target_seconds"] == .02
    assert len(paired_schedule(case["methods"], 5, rounds, 123)) == len(case["methods"]) * 5 * rounds


def test_lapx_distribution_version_and_binaries_enter_environment_identity(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from . import runner
    from .common import digest_json
    import psutil

    extension = tmp_path / "lap" / "_lapjv.pyd"
    extension.parent.mkdir()
    extension.write_bytes(b"first solver build")
    distribution = SimpleNamespace(files=[Path("lap/_lapjv.pyd")],
                                   locate_file=lambda file: tmp_path / file)
    installed = {"version": "0.13.0"}

    def version(name):
        if name == "lapx":
            return installed["version"]
        raise runner.importlib.metadata.PackageNotFoundError(name)

    def lookup(name):
        if name == "lapx":
            return distribution
        raise runner.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runner.importlib.metadata, "version", version)
    monkeypatch.setattr(runner.importlib.metadata, "distribution", lookup)
    monkeypatch.setattr(psutil, "cpu_percent", lambda **kwargs: 0.)
    first = runner.environment()
    assert first["packages"]["lapx"] == "0.13.0"
    assert set(first["native_artifact_hashes"]["lapx"]) == {str(Path("lap/_lapjv.pyd"))}
    identity = lambda env: digest_json({key: env[key] for key in ("packages", "native_artifact_hashes")})
    extension.write_bytes(b"different solver build")
    second = runner.environment()
    assert identity(first) != identity(second)
    installed["version"] = "0.13.1"
    third = runner.environment()
    assert identity(second) != identity(third)
