"""Archived build metadata must remain tied to the recorded source and binary."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import controlled_comparison as common
from .build_provenance import verified_builder


def prepared_build(tmp_path, monkeypatch):
    root, build = tmp_path / "source", tmp_path / "build"
    inputs = {
        "lsape_bridge.cpp": "lsape_bridge.cpp",
        "build_bridge.py": "build_bridge.py",
        "lsape-v4.zip": "src/lsape-v4.zip",
        "hungarian-lsape.hh": "src/upstream-v4/lsape/include/hungarian-lsape.hh",
        "src/specialized/hungarian-lsape-uniform.hh": "src/specialized/hungarian-lsape-uniform.hh",
    }
    for name, relative in inputs.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    snapshot = build / "snapshot"
    archived = snapshot / "sources/build_bridge.py"
    archived.parent.mkdir(parents=True)
    archived.write_text("recorded builder; never execute", encoding="utf-8")
    binary = build / "_lsape_native.pyd"
    binary.write_bytes(b"recorded native binary")
    (snapshot / binary.name).write_bytes(binary.read_bytes())
    record = {"returncode": 0, "output_sha256": common.digest(binary),
              "compiler_flags": ["/O2", "/fp:precise"],
              "input_sha256": {name: common.digest(root / relative) for name, relative in inputs.items()}}
    record["input_sha256"]["build_bridge.py"] = common.digest(archived)
    for path in (build / "provenance.json", snapshot / "provenance.json"):
        path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(common, "ROOT", root)
    return root, build, record, SimpleNamespace(__file__=str(binary))


def test_snapshot_keeps_actual_builder_and_current_sources(tmp_path, monkeypatch):
    root, build, record, bridge = prepared_build(tmp_path, monkeypatch)
    output = tmp_path / "run"
    result = common.snapshot_provenance(output, bridge)
    builder = next(item for item in result["verified_build_inputs"] if item["recorded_name"] == "build_bridge.py")
    assert Path(builder["path"]) == build / "snapshot/sources/build_bridge.py"
    assert builder["sha256"] == record["input_sha256"]["build_bridge.py"]
    assert (output / "provenance/build_bridge.py").read_bytes() == (root / "build_bridge.py").read_bytes()
    for relative in ("sources/build_bridge.py", "provenance.json", "_lsape_native.pyd"):
        assert (output / "provenance/build/snapshot" / relative).read_bytes() == (build / "snapshot" / relative).read_bytes()


@pytest.mark.parametrize("target", ["snapshot/sources/build_bridge.py", "snapshot/provenance.json",
                                    "snapshot/_lsape_native.pyd", "_lsape_native.pyd"])
def test_archive_fallback_rejects_changed_evidence(tmp_path, monkeypatch, target):
    root, build, record, _ = prepared_build(tmp_path, monkeypatch)
    (build / target).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="snapshot"):
        verified_builder(build, root, record)


def test_archive_fallback_requires_builder_bytes(tmp_path, monkeypatch):
    root, build, record, _ = prepared_build(tmp_path, monkeypatch)
    (build / "snapshot/sources/build_bridge.py").unlink()
    with pytest.raises(RuntimeError, match="snapshot"):
        verified_builder(build, root, record)


@pytest.mark.parametrize("relative", ["lsape_bridge.cpp", "src/lsape-v4.zip",
                                     "src/upstream-v4/lsape/include/hungarian-lsape.hh",
                                     "src/specialized/hungarian-lsape-uniform.hh"])
def test_archived_builder_keeps_current_compiled_input_guards(tmp_path, monkeypatch, relative):
    root, _, _, bridge = prepared_build(tmp_path, monkeypatch)
    (root / relative).write_bytes(b"changed compiled input")
    with pytest.raises(RuntimeError, match="Source changed"):
        common.snapshot_provenance(tmp_path / "run", bridge)


def test_current_builder_needs_no_archived_fallback(tmp_path, monkeypatch):
    root, build, record, _ = prepared_build(tmp_path, monkeypatch)
    record["input_sha256"]["build_bridge.py"] = common.digest(root / "build_bridge.py")
    assert verified_builder(build, root, record) == (root / "build_bridge.py", ())


def test_failed_build_record_is_rejected(tmp_path, monkeypatch):
    root, build, record, _ = prepared_build(tmp_path, monkeypatch)
    record["returncode"] = 1
    with pytest.raises(RuntimeError, match="successful build"):
        verified_builder(build, root, record)


def test_record_cannot_omit_a_required_compiled_input(tmp_path, monkeypatch):
    _, build, record, bridge = prepared_build(tmp_path, monkeypatch)
    del record["input_sha256"]["lsape_bridge.cpp"]
    for path in (build / "provenance.json", build / "snapshot/provenance.json"):
        path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Required build input"):
        common.snapshot_provenance(tmp_path / "run", bridge)
