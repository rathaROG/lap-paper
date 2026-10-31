import json
from pathlib import Path
import subprocess
import sys

import pytest

from .rectangular_campaign import ROOT, read_manifest, smoke_manifest, parse_args
from . import controlled_comparison as common
from .summarize_rectangular import check_reports


def test_restored_manifest_preserves_frozen_input_definition():
    document = read_manifest(ROOT / "data/rectangular-grid-v1.json", None)
    assert document["manifest_sha256"] == "875745fcdef90f08f1d57414e3c0e1bea35b5e417972ddb15f3de8ececfa4a9b"
    assert document["manifest"]["case_count"] == 67
    smoke = smoke_manifest(document["manifest"])
    assert len(smoke["cases"]) == 10
    assert {c["block"] for c in smoke["cases"]} == set("ABCDEF")
    assert smoke["inputs_per_case"] == 1 and smoke["rounds"] == 2


def test_full_run_requires_opt_in_before_output_creation(tmp_path):
    output = tmp_path / "never-created"
    result = subprocess.run([sys.executable, "-B", "-m", "benchmark.lsape", "run",
                             "--build-dir", str(tmp_path / "build"), "--output", str(output),
                             "--session-index", "1"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--allow-full-campaign" in result.stderr
    assert not output.exists()


def test_builds_require_explicit_location():
    with pytest.raises(SystemExit):
        parse_args(["--session-index", "1", "--allow-full-campaign"])


def test_runtime_rejects_different_python_patch_before_loading(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"python_version": "0.0.0"}))
    with pytest.raises(RuntimeError, match="same Python patch version"):
        common.load_runtime(tmp_path)


def test_summary_rejects_smoke():
    reports = [{"session_index": i, "scope": "smoke_not_publication_evidence"} for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="Smoke"):
        check_reports([], reports)


def test_upstream_source_is_unchanged():
    assert common.check_upstream_archive()["all_members_identical"]


def complete_reports():
    """Minimal complete three-session fixture with independently checked shapes."""
    from copy import deepcopy
    manifest = read_manifest(ROOT / "data/rectangular-grid-v1.json", None)["manifest"]
    methods = list(manifest["methods"])
    template = {
        "scope": "lsape_five_arm_primary", "status": "complete", "validate_only": False,
        "manifest": manifest, "manifest_sha256": "frozen", "driver_sha256": "driver",
        "shared_harness_sha256": "harness", "provenance": {"loaded_bridge_sha256": "binary"},
        "python": "3.11.18 build identity", "platform": "Windows test platform",
        "processor": "CPU model", "versions": {"numpy": "2.4.6", "scipy": "1.17.1"},
        "affinity": {"applied": True, "requested_logical_cpu": 0, "original_process_mask": 255,
                     "system_mask": 255, "error": None},
        "thread_environment": {"requested": 1, "previous": {}},
        "cases": [{"case": spec["case"], "status": "complete", "input_unchanged_after_timing": True,
                   "methods": {m: {"round_average_us": [1.] * manifest["rounds"]} for m in methods},
                   "round_order": [methods[:] for _ in range(manifest["rounds"])],
                   "diagnostics": [{"methods": {m: {"certificate": {"passed": True},
                        "input_unchanged": True, "bitwise_state_equals_one_sided_upstream": True}
                        for m in methods}} for _ in range(manifest["inputs_per_case"])]}
                  for spec in manifest["cases"]],
    }
    return [{**deepcopy(template), "session_index": index, "pid": 100 + index} for index in (1, 2, 3)]


def test_summary_accepts_matching_environments():
    reports = complete_reports()
    manifest, identities = check_reports([Path(str(i)) for i in range(3)], reports)
    assert len(manifest["cases"]) == 67
    assert len(identities["environment_sha256"]) == 64


@pytest.mark.parametrize("field, replacement", [
    ("python", "3.14.10 build identity"),
    ("versions", {"numpy": "2.4.6", "scipy": "different"}),
    ("platform", "Windows different build"),
    ("processor", "Different CPU"),
    ("affinity", {"applied": True, "requested_logical_cpu": 1, "original_process_mask": 255,
                  "system_mask": 255, "error": None}),
])
def test_summary_rejects_mixed_environments(field, replacement):
    reports = complete_reports()
    reports[1][field] = replacement
    with pytest.raises(ValueError, match="differs across sessions"):
        check_reports([Path(str(i)) for i in range(3)], reports)


@pytest.mark.parametrize("field", ["python", "versions", "platform", "processor", "affinity", "thread_environment"])
def test_summary_rejects_missing_environment_identity(field):
    reports = complete_reports()
    del reports[1][field]
    with pytest.raises(ValueError, match="[Ee]nvironment"):
        check_reports([Path(str(i)) for i in range(3)], reports)
