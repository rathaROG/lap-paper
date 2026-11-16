"""Evidence packaging regressions using synthetic files; never run benchmarks."""
import json
from pathlib import Path
import zipfile

import pytest

from . import paper_evidence as evidence


@pytest.fixture
def package_root(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "DESTINATION", tmp_path / "benchmark/evidence")
    monkeypatch.setattr(evidence, "MANIFEST", evidence.DESTINATION / "manifest.json")
    return tmp_path


def write_inventory(root, paths, format="records"):
    inventory = root / "generator_input_files.json"
    source = paths if format == "list" else {"files": [{"path": path} for path in paths]}
    inventory.write_text(json.dumps(source), encoding="utf-8")
    return inventory


@pytest.mark.parametrize("format", ["list", "records"])
def test_mixed_inventory_packs_campaigns_and_restores_exact_bytes(package_root, format):
    primary = "benchmark/runs/primary"
    sensitivity = "benchmark/runs/linux-py3.11.18-20261030-1434"
    raw_suffix = "/paper-run-light/campaign-session/attempts/attempt-1/measurements.jsonl"
    campaign = {primary + "/summary.json", sensitivity + "/summary.json"}
    microscopy = {"benchmark/cell_tracking/observations.json", "benchmark/cell_tracking/recalculate.py"}
    paths = campaign | microscopy | {primary + raw_suffix, sensitivity + raw_suffix}
    payloads = {path: (path + "\r\n").encode("utf-8") for path in paths}
    for relative, payload in payloads.items():
        target = package_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    inventory = write_inventory(package_root, sorted(campaign | microscopy), format)
    original_inventory = inventory.read_bytes()

    evidence.pack(inventory)

    assert inventory.read_bytes() == original_inventory
    manifest = json.loads(evidence.MANIFEST.read_text(encoding="utf-8"))
    expected = campaign | {primary + raw_suffix}
    assert {row["path"] for row in manifest["files"]} == expected
    assert {row["path"] for row in manifest["files"] if row["role"] == "generator_input"} == campaign
    assert len(manifest["archives"]) == 2
    assert "microscopy evidence is versioned separately" in manifest["scope"]
    for archive in manifest["archives"]:
        with zipfile.ZipFile(package_root / archive["path"]) as packed:
            assert packed.namelist() == sorted(packed.namelist())
            assert all(packed.read(name) == payloads[name] for name in packed.namelist())
    for relative in expected:
        (package_root / relative).unlink()
    evidence.restore(extract=True)
    assert all((package_root / path).read_bytes() == payload for path, payload in payloads.items())
    evidence.restore(extract=False)


@pytest.mark.parametrize("unexpected", [
    "latex/main.tex", "benchmark/runs/run/../outside.json",
    "benchmark/cell_tracking/../outside.json", "benchmark/runs/run\\outside.json",
    "/benchmark/runs/run/file.json", "benchmark/runs/run/C:outside.json",
    "benchmark/runs/run", "benchmark/runs//run/file.json",
])
def test_invalid_inventory_is_rejected_before_packaging(package_root, unexpected):
    inventory = write_inventory(package_root, ["benchmark/runs/primary/summary.json", unexpected])
    with pytest.raises(ValueError, match="(Unsafe|Unexpected) generator input path"):
        evidence.pack(inventory)
    assert not evidence.DESTINATION.exists()


@pytest.mark.parametrize("paths", [[], ["benchmark/cell_tracking/observations.json"]])
def test_inventory_without_campaigns_is_rejected(package_root, paths):
    inventory = write_inventory(package_root, paths)
    with pytest.raises(ValueError, match="No campaign inputs"):
        evidence.pack(inventory)
    assert not evidence.DESTINATION.exists()


def test_current_generator_inventory_selects_all_and_only_campaign_inputs():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root / "benchmark/paper-artifacts/generator_input_files.json").read_text(encoding="utf-8"))
    all_paths = {row["path"] for row in source["files"]}
    campaign = evidence.campaign_inputs(source)
    assert campaign == {path for path in all_paths if path.startswith("benchmark/runs/")}
    assert all_paths - campaign
    assert all(path.startswith("benchmark/cell_tracking/") for path in all_paths - campaign)
