"""Automatic artifact preparation without network, compilers or native loads."""
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

import pytest

from . import preparation
from .campaign import native_build


def campaign_index(directory, *, diagnostic=False):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for language, commit in native_build.SOURCE_COMMITS.items():
        for arm in native_build.ALL_ARMS:
            namespace = f"campaign_{'diag_' if diagnostic else ''}{language}_{arm}"
            package_root = directory / language / arm
            package = package_root / namespace
            package.mkdir(parents=True)
            artifact = package / ("_lapjv" + sysconfig.get_config_var("EXT_SUFFIX"))
            artifact.write_bytes(f"{language}/{arm}".encode())
            records.append(dict(
                language=language, arm_id=arm, source_commit=commit,
                policy=native_build.policy_for(arm), release=True,
                profile_enabled=diagnostic, runtime_policy_overrides=False,
                diagnostic=diagnostic, eligible_for_primary_timing=not diagnostic,
                package_name=namespace, package_root=str(package_root),
                artifact_path=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                python=dict(version=platform.python_version(), cache_tag=sys.implementation.cache_tag,
                            platform=sysconfig.get_platform(), soabi=sysconfig.get_config_var("SOABI"))))
    name = "diagnostic-build-index.json" if diagnostic else "build-index.json"
    path = directory / name
    path.write_text(json.dumps(dict(schema_version=1, diagnostic=diagnostic,
                                   eligible_for_primary_timing=not diagnostic, builds=records)), encoding="utf-8")
    return path


@pytest.fixture
def isolated_loader(monkeypatch):
    commands = []

    def run(command, **kwargs):
        # Preparation may load the controlled APIs in a child interpreter;
        # release package installation and build commands are forbidden here.
        assert command[:3] == [sys.executable, "-B", "-c"]
        assert "pip" not in command and "setup.py" not in command
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(preparation, "_run", run)
    return commands


@pytest.mark.parametrize("diagnostic", [False, True])
def test_reuses_current_environment_artifacts_without_source_or_build_work(
        tmp_path, monkeypatch, isolated_loader, diagnostic):
    environment = tmp_path / "prepared-environment"
    index = campaign_index(environment / ("diagnostic-builds" if diagnostic else "builds"),
                           diagnostic=diagnostic)
    monkeypatch.setattr(sys, "prefix", str(environment / "env"))
    monkeypatch.setattr(preparation, "RUNS", tmp_path / "runs")
    monkeypatch.delenv("CAMPAIGN_NATIVE_BUILD_INDEX", raising=False)
    monkeypatch.delenv("CAMPAIGN_DIAGNOSTIC_BUILD_INDEX", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("A compatible environment index must not need sources, compilers or builds")

    for name in ("_sources", "_build_campaign", "_check_toolchains"):
        monkeypatch.setattr(preparation, name, forbidden)
    result = preparation.prepare_campaign(tmp_path / "runs" / "preparation", diagnostic=diagnostic)
    assert result == index.resolve()
    assert isolated_loader


@pytest.mark.parametrize("diagnostic", [False, True])
def test_discovers_indexes_saved_with_previous_runs(tmp_path, monkeypatch, diagnostic):
    runs = tmp_path / "runs"
    index = campaign_index(runs / "previous-session" / "campaign-smoke", diagnostic=diagnostic)
    if not diagnostic:
        index = index.rename(index.with_name("native-build-index.json"))
    monkeypatch.setattr(preparation, "RUNS", runs)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unprepared-environment" / "env"))
    monkeypatch.delenv("CAMPAIGN_NATIVE_BUILD_INDEX", raising=False)
    monkeypatch.delenv("CAMPAIGN_DIAGNOSTIC_BUILD_INDEX", raising=False)
    assert index.resolve() in preparation._campaign_candidates(tmp_path / "new-cache", diagnostic)


@pytest.mark.parametrize("defect", ["python", "abi", "platform", "commit", "policy", "instrumentation", "artifact"])
def test_incompatible_campaign_metadata_is_rejected_before_native_loading(
        tmp_path, isolated_loader, defect):
    index = campaign_index(tmp_path / "builds")
    payload = json.loads(index.read_text())
    record = payload["builds"][0]
    if defect == "python":
        record["python"]["version"] = "0.0.0"
    elif defect == "abi":
        record["python"]["cache_tag"] = "cpython-other"
    elif defect == "platform":
        record["python"]["platform"] = "other-platform"
    elif defect == "commit":
        record["source_commit"] = "0" * 40
    elif defect == "policy":
        record["policy"]["budget"] = 99
    elif defect == "instrumentation":
        record["profile_enabled"] = True
    else:
        Path(record["artifact_path"]).write_bytes(b"changed binary")
    index.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(preparation.PreparationError):
        preparation._validate_campaign(index, diagnostic=False)
    assert not isolated_loader


@pytest.mark.parametrize("diagnostic", [False, True])
def test_primary_and_diagnostic_indexes_cannot_be_substituted(tmp_path, isolated_loader, diagnostic):
    index = campaign_index(tmp_path / "builds", diagnostic=not diagnostic)
    with pytest.raises(preparation.PreparationError):
        preparation._validate_campaign(index, diagnostic=diagnostic)
    assert not isolated_loader


@pytest.mark.parametrize("diagnostic", [False, True])
def test_missing_compatible_cache_prepares_only_controlled_extensions(
        tmp_path, monkeypatch, isolated_loader, diagnostic):
    events = []
    sources = {language: tmp_path / language for language in native_build.SOURCE_COMMITS}
    monkeypatch.setattr(preparation, "_campaign_candidates", lambda *args: [])
    monkeypatch.setattr(preparation, "_check_toolchains",
                        lambda **kwargs: events.append(("toolchains", kwargs)))

    def acquire(cache):
        events.append(("sources", cache))
        return sources

    def build(selected_sources, output, selected_diagnostic):
        assert selected_sources == sources and selected_diagnostic == diagnostic
        events.append(("build", output))
        return campaign_index(output, diagnostic=selected_diagnostic)

    monkeypatch.setattr(preparation, "_sources", acquire)
    monkeypatch.setattr(preparation, "_build_campaign", build)
    cache = tmp_path / "runs" / "preparation"
    result = preparation.prepare_campaign(cache, diagnostic=diagnostic)
    assert result.is_file() and result.is_relative_to(cache)
    assert [event[0] for event in events] == ["toolchains", "sources", "build"]
    assert events[0][1] == {"campaign": True}


def test_missing_toolchain_fails_before_acquiring_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_campaign_candidates", lambda *args: [])

    def unavailable(**kwargs):
        raise preparation.PreparationError("Compiler unavailable")

    monkeypatch.setattr(preparation, "_check_toolchains", unavailable)
    monkeypatch.setattr(preparation, "_sources", lambda *args: pytest.fail("Sources acquired before toolchain check"))
    with pytest.raises(preparation.PreparationError, match="Compiler unavailable"):
        preparation.prepare_campaign(tmp_path / "runs" / "preparation")


@pytest.mark.parametrize("cached", [False, True])
def test_pinned_sources_are_fetched_only_when_missing(tmp_path, monkeypatch, cached):
    cache = tmp_path / "preparation"
    if cached:
        for language, commit in native_build.SOURCE_COMMITS.items():
            name = "laprust" if language == "rust" else "lapcpp"
            (cache / "sources" / f"{name}-{commit[:12]}" / ".git").mkdir(parents=True)
    commands = []

    def command(parts, **kwargs):
        assert parts[0] == "git"
        commands.append(parts)
        if parts[1] == "init":
            (Path(parts[2]) / ".git").mkdir()
        output = parts[-1].split("^")[0] if "rev-parse" in parts else ""
        return subprocess.CompletedProcess(parts, 0, stdout=output, stderr="")

    def exists(parts, **kwargs):
        assert parts[0] == "git" and parts[3:5] == ["cat-file", "-e"]
        return subprocess.CompletedProcess(parts, 0 if cached else 1)

    monkeypatch.setattr(preparation, "_run", command)
    monkeypatch.setattr(preparation.subprocess, "run", exists)
    sources = preparation._sources(cache)
    assert set(sources) == set(native_build.SOURCE_COMMITS)
    fetches = [parts for parts in commands if "fetch" in parts]
    assert len(fetches) == (0 if cached else 2)
    for language, commit in native_build.SOURCE_COMMITS.items():
        assert sources[language].name.endswith(commit[:12])
        if not cached:
            assert any(parts[-2:] == [preparation.SOURCE_URLS[language], commit] for parts in fetches)


def lsape_build(directory):
    from .lsape import controlled_comparison as common
    directory.mkdir(parents=True)
    binary = directory / "_lsape_native.pyd"
    binary.write_bytes(b"test-only bridge")
    inputs = {
        "lsape_bridge.cpp": common.ROOT / "lsape_bridge.cpp",
        "build_bridge.py": common.ROOT / "build_bridge.py",
        "lsape-v4.zip": common.ROOT / "src/lsape-v4.zip",
        "hungarian-lsape.hh": common.ROOT / "src/upstream-v4/lsape/include/hungarian-lsape.hh",
        "src/specialized/hungarian-lsape-uniform.hh": common.ROOT / "src/specialized/hungarian-lsape-uniform.hh",
    }
    record = dict(returncode=0, python_version=platform.python_version(),
                  output_sha256=common.digest(binary),
                  input_sha256={name: common.digest(path) for name, path in inputs.items()})
    (directory / "provenance.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def test_reuses_environment_lsape_bridge_without_toolchains(tmp_path, monkeypatch, isolated_loader):
    environment = tmp_path / "prepared-environment"
    existing = lsape_build(environment / "lsape-readiness" / "build")
    monkeypatch.setattr(sys, "prefix", str(environment / "env"))
    monkeypatch.setattr(preparation, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(preparation.platform, "system", lambda: "Windows")
    monkeypatch.delenv("LSAPE_BUILD_DIR", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing compatible LSAPE build must not need a compiler or build")

    monkeypatch.setattr(preparation, "_check_toolchains", forbidden)
    monkeypatch.setattr(preparation, "_build_lsape", forbidden)
    cache = tmp_path / "runs" / "preparation"
    result = preparation.prepare_lsape(cache)
    assert result == (cache / "lsape-build").resolve()
    assert (result / "provenance.json").read_bytes() == (existing / "provenance.json").read_bytes()
    assert isolated_loader


def archived_lsape_build(directory):
    """Keep an older actual builder bound to its source record and binary."""
    from .lsape import controlled_comparison as common
    build = lsape_build(directory)
    builder = build / "snapshot/sources/build_bridge.py"
    builder.parent.mkdir(parents=True)
    builder.write_bytes(b"archived actual builder; never executed")
    record = json.loads((build / "provenance.json").read_text(encoding="utf-8"))
    record["input_sha256"]["build_bridge.py"] = common.digest(builder)
    for provenance in (build / "provenance.json", build / "snapshot/provenance.json"):
        provenance.write_text(json.dumps(record), encoding="utf-8")
    shutil.copy2(build / "_lsape_native.pyd", build / "snapshot/_lsape_native.pyd")
    (build / "build.log").write_bytes(b"recorded compiler output")
    (build / "compile_bridge.cmd").write_bytes(b"recorded compiler invocation")
    return build


def build_bytes(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_lsape_cache_retains_bound_build_after_original_and_report_removal(
        tmp_path, monkeypatch, isolated_loader):
    from .lsape.build_provenance import verified_builder
    from .lsape import controlled_comparison as common
    environment = tmp_path / "environment"
    original = archived_lsape_build(environment / "lsape-readiness/build")
    before = build_bytes(original)
    runs = tmp_path / "runs"
    report = runs / "dated/paper-smoke/lsape-smoke/provenance/build"
    shutil.copytree(original, report)
    os.utime(report / "provenance.json", (4_000_000_000, 4_000_000_000))
    cache = runs / ".cache/windows-py-test"
    monkeypatch.setattr(sys, "prefix", str(environment / "env"))
    monkeypatch.setattr(preparation, "RUNS", runs)
    monkeypatch.setattr(preparation.platform, "system", lambda: "Windows")
    monkeypatch.delenv("LSAPE_BUILD_DIR", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Compatible cached LSAPE must not need a compiler or a rebuild")

    monkeypatch.setattr(preparation, "_check_toolchains", forbidden)
    monkeypatch.setattr(preparation, "_build_lsape", forbidden)
    candidates = preparation._lsape_candidates(cache)
    assert candidates[0] == original.resolve()
    assert report.resolve() not in candidates
    result = preparation.prepare_lsape(cache)
    assert result == (cache / "lsape-build").resolve()
    assert build_bytes(result) == before == build_bytes(original)
    record = json.loads((result / "provenance.json").read_text(encoding="utf-8"))
    builder, evidence = verified_builder(result, common.ROOT, record)
    assert builder == result / "snapshot/sources/build_bridge.py"
    assert set(evidence) == {result / "snapshot/provenance.json", result / "snapshot/_lsape_native.pyd"}
    for directory in (environment, runs / "dated"):
        assert directory.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(directory)
    monkeypatch.setattr(preparation.shutil, "copytree", forbidden)
    assert preparation.prepare_lsape(cache) == result
    assert build_bytes(result) == before


def test_lsape_candidate_priority_keeps_canonical_cache_before_newer_originals(tmp_path, monkeypatch):
    cache, environment, runs = tmp_path / "cache", tmp_path / "environment", tmp_path / "runs"
    canonical = lsape_build(cache / "lsape-build")
    newer_cache = lsape_build(cache / "lsape-build-later")
    original = lsape_build(environment / "lsape-readiness/build")
    dated = lsape_build(runs / "dated/lsape-build")
    report = lsape_build(runs / "dated/report/provenance/build")
    snapshot = lsape_build(runs / "dated/verification/build-snapshot")
    for path in (newer_cache, original, dated, report, snapshot):
        os.utime(path / "provenance.json", (4_000_000_000, 4_000_000_000))
    monkeypatch.setattr(sys, "prefix", str(environment / "env"))
    monkeypatch.setattr(preparation, "RUNS", runs)
    monkeypatch.delenv("LSAPE_BUILD_DIR", raising=False)
    assert preparation._lsape_candidates(cache) == [canonical, newer_cache, original, dated]
    monkeypatch.setenv("LSAPE_BUILD_DIR", str(original))
    assert preparation._lsape_candidates(cache) == [original, canonical, newer_cache, dated]


def test_failed_lsape_cache_validation_stops_without_rebuilding(tmp_path, monkeypatch, isolated_loader):
    original = archived_lsape_build(tmp_path / "environment/lsape-readiness/build")
    cache = tmp_path / "cache"
    monkeypatch.setattr(preparation.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preparation, "_lsape_candidates", lambda root: [original])
    validate = preparation._validate_lsape

    def reject_copy(directory):
        if Path(directory).resolve() != original.resolve():
            raise preparation.PreparationError("copied build failed validation")
        validate(directory)

    def forbidden(*args, **kwargs):
        pytest.fail("A failed copy must stop, rather than silently rebuilding")

    monkeypatch.setattr(preparation, "_validate_lsape", reject_copy)
    monkeypatch.setattr(preparation, "_check_toolchains", forbidden)
    monkeypatch.setattr(preparation, "_build_lsape", forbidden)
    with pytest.raises(preparation.PreparationError, match="copied build failed validation"):
        preparation.prepare_lsape(cache)
    assert (cache / "lsape-build/provenance.json").is_file()


def test_lsape_rejects_interpreter_mismatch_before_native_loading(tmp_path, isolated_loader):
    existing = lsape_build(tmp_path / "build")
    provenance = existing / "provenance.json"
    record = json.loads(provenance.read_text())
    record["python_version"] = "0.0.0"
    provenance.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(preparation.PreparationError):
        preparation._validate_lsape(existing)
    assert not isolated_loader


@pytest.mark.parametrize("system, pointer_bytes", [("Linux", 8), ("Windows", 4)])
def test_lsape_platform_guard_runs_before_discovery_or_build(tmp_path, monkeypatch, system, pointer_bytes):
    monkeypatch.setattr(preparation.platform, "system", lambda: system)
    monkeypatch.setattr(preparation.struct, "calcsize", lambda fmt: pointer_bytes)
    monkeypatch.setattr(preparation, "_lsape_candidates", lambda *args: pytest.fail("Unsupported LSAPE platform"))
    with pytest.raises(preparation.PreparationError, match="64-bit Windows"):
        preparation.prepare_lsape(tmp_path / "preparation")


def test_missing_lsape_bridge_builds_bundled_sources_only(tmp_path, monkeypatch, isolated_loader):
    events = []
    monkeypatch.setattr(preparation.platform, "system", lambda: "Windows")
    monkeypatch.setattr(preparation, "_lsape_candidates", lambda *args: [])
    monkeypatch.setattr(preparation, "_check_toolchains", lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(preparation, "_sources", lambda *args: pytest.fail("LSAPE sources are already bundled"))
    monkeypatch.setattr(preparation, "_build_lsape", lsape_build)
    cache = tmp_path / "runs" / "preparation"
    result = preparation.prepare_lsape(cache)
    assert result.is_relative_to(cache) and (result / "_lsape_native.pyd").is_file()
    assert events == [{"campaign": False}]
    assert isolated_loader
