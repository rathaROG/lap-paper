"""Find compatible custom builds, preparing only missing experiment variants."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import sysconfig
import uuid


RUNS = Path(__file__).resolve().parent / "runs"
SOURCE_URLS = {"rust": "https://github.com/rathaROG/laprust.git",
               "cpp": "https://github.com/rathaROG/lapcpp.git"}


class PreparationError(RuntimeError):
    """Preparation cannot continue with the available artifacts or tools."""


def _run(command, *, capture=False, cwd=None):
    try:
        result = subprocess.run([str(part) for part in command], cwd=cwd,
                                capture_output=capture, text=True, encoding="utf-8",
                                errors="replace", check=False)
    except OSError as error:
        raise PreparationError(f"Cannot start {command[0]}: {error}") from error
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise PreparationError(f"{Path(command[0]).name} failed" + (f": {detail}" if detail else ""))
    return result


def _recent(paths):
    found = {}
    for path in paths:
        try:
            path = Path(path).expanduser().resolve()
            found[path] = path.stat().st_mtime if path.is_file() else -1
        except OSError:
            continue
    return sorted((path for path in found if found[path] >= 0), key=lambda path: found[path], reverse=True)


def _campaign_candidates(cache_root, diagnostic):
    names = ("diagnostic-build-index.json",) if diagnostic else ("build-index.json", "native-build-index.json")
    folder = "diagnostic-builds" if diagnostic else "builds"
    paths = []
    for root in (cache_root, Path(sys.prefix), Path(sys.prefix).parent):
        for name in names:
            paths.extend(root.glob(f"{folder}*/{name}"))
    for name in names:
        paths.extend(RUNS.rglob(name))
    candidates = _recent(paths)
    configured = os.environ.get("CAMPAIGN_DIAGNOSTIC_BUILD_INDEX" if diagnostic else "CAMPAIGN_NATIVE_BUILD_INDEX")
    if configured:
        selected = Path(configured).expanduser().resolve()
        candidates = [selected] + [path for path in candidates if path != selected]
    return candidates


def _validate_campaign(index, diagnostic):
    from .campaign import native_build
    try:
        records = json.loads(Path(index).read_text(encoding="utf-8"))
        for language, commit in native_build.SOURCE_COMMITS.items():
            for arm in native_build.ALL_ARMS:
                record = native_build.resolve_build(records, language, arm)
                identity = record["python"]
                if (identity.get("version", "").split()[0] != platform.python_version()
                        or identity.get("cache_tag") != sys.implementation.cache_tag
                        or identity.get("platform") != sysconfig.get_platform()):
                    raise PreparationError("Custom build belongs to a different Python or platform")
                if record.get("source_commit") != commit or record.get("policy") != native_build.policy_for(arm):
                    raise PreparationError("Custom build does not match the pinned experiment")
                if diagnostic:
                    valid = record.get("diagnostic") is True and record.get("eligible_for_primary_timing") is False
                else:
                    valid = record.get("profile_enabled") is False and record.get("runtime_policy_overrides") is False
                if not valid:
                    raise PreparationError("Custom build has the wrong instrumentation settings")
                if native_build.sha256_file(Path(record["artifact_path"])) != record["artifact_sha256"]:
                    raise PreparationError("Custom build artifact hash changed")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        raise PreparationError(f"Invalid custom build index: {index}") from error
    code = """import json, sys
from benchmark.campaign import native_build, diagnostic_build
index = json.load(open(sys.argv[1], encoding='utf-8'))
loader = diagnostic_build.load_diagnostic if sys.argv[2] == 'diagnostic' else native_build.load_solver
for language in native_build.SOURCE_COMMITS:
    for arm in native_build.ALL_ARMS:
        loader(native_build.resolve_build(index, language, arm))
"""
    _run([sys.executable, "-B", "-c", code, str(index), "diagnostic" if diagnostic else "primary"], capture=True)


def _check_toolchains(*, campaign):
    if platform.system() not in ("Windows", "Linux"):
        raise PreparationError("Custom experiment builds support Windows and Linux")
    if campaign:
        missing = [name for name in ("git", "cargo", "rustc") if shutil.which(name) is None]
        if missing:
            raise PreparationError("Custom comparisons need " + ", ".join(missing) + " available on this computer")
        for command in (("cargo", "--version"), ("rustc", "--version")):
            _run(command, capture=True)
        modules = ("numpy", "setuptools", "Cython", "pybind11", "wheel")
    else:
        if platform.system() != "Windows" or struct.calcsize("P") != 8:
            raise PreparationError("The LSAPE experiment requires 64-bit Windows and MSVC")
        modules = ("numpy",)
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise PreparationError("Custom builds need the benchmark dependencies: " + ", ".join(missing))
    if not (Path(sysconfig.get_path("include")) / "Python.h").is_file():
        raise PreparationError("Custom builds need the running Python's development headers")
    if os.name == "nt":
        from .lsape.build_bridge import find_vcvars
        try:
            available = find_vcvars().is_file()
        except SystemExit as error:
            raise PreparationError("Custom comparisons need the MSVC x64 C++ build tools") from error
        if not available:
            raise PreparationError("Custom comparisons need the MSVC x64 C++ build tools")
    elif not any(shutil.which(name) for name in ("c++", "g++", "clang++")):
        raise PreparationError("Custom comparisons need a C++ compiler")


def _fresh_directory(cache_root, name):
    output = cache_root / name
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        output = cache_root / f"{name}-{uuid.uuid4().hex[:12]}"
    return output


def _sources(cache_root):
    from .campaign.native_build import SOURCE_COMMITS
    sources = {}
    for language, commit in SOURCE_COMMITS.items():
        name = "laprust" if language == "rust" else "lapcpp"
        repository = cache_root / "sources" / f"{name}-{commit[:12]}"
        if repository.exists() and not (repository / ".git").is_dir():
            repository = repository.with_name(repository.name + "-" + uuid.uuid4().hex[:12])
        repository.mkdir(parents=True, exist_ok=True)
        if not (repository / ".git").is_dir():
            _run(["git", "init", str(repository)], capture=True)
        try:
            result = subprocess.run(["git", "-C", str(repository), "cat-file", "-e", commit + "^{commit}"],
                                    capture_output=True, check=False)
        except OSError as error:
            raise PreparationError(f"Cannot inspect the pinned {name} source: {error}") from error
        if result.returncode:
            _run(["git", "-C", str(repository), "fetch", "--depth=1", "--no-tags", SOURCE_URLS[language], commit], capture=True)
        resolved = _run(["git", "-C", str(repository), "rev-parse", commit + "^{commit}"], capture=True).stdout.strip()
        if resolved != commit:
            raise PreparationError(f"Cannot verify the pinned {name} source commit")
        sources[language] = repository
    return sources


def _build_campaign(sources, output, diagnostic):
    module = "benchmark.campaign.diagnostic_build" if diagnostic else "benchmark.campaign.native_build"
    _run([sys.executable, "-B", "-m", module, "--rust-source", sources["rust"],
          "--cpp-source", sources["cpp"], "--output", output])
    return output / ("diagnostic-build-index.json" if diagnostic else "build-index.json")


def prepare_campaign(cache_root: Path, *, diagnostic: bool = False) -> Path:
    """Reuse a compatible complete index, or build the pinned custom variants."""
    cache_root = Path(cache_root).expanduser().resolve()
    for index in _campaign_candidates(cache_root, diagnostic):
        try:
            _validate_campaign(index, diagnostic)
        except PreparationError:
            continue
        print(f"Using {'diagnostic' if diagnostic else 'controlled'} builds: {index}", flush=True)
        return index.resolve()
    _check_toolchains(campaign=True)
    sources = _sources(cache_root)
    output = _fresh_directory(cache_root, "diagnostic-builds" if diagnostic else "builds")
    index = _build_campaign(sources, output, diagnostic)
    _validate_campaign(index, diagnostic)
    return index.resolve()


def _lsape_candidates(cache_root):
    canonical = cache_root / "lsape-build" / "provenance.json"
    cached = _recent([canonical]) + _recent(cache_root.glob("lsape-build*/provenance.json"))
    originals = []
    for root in (Path(sys.prefix), Path(sys.prefix).parent):
        originals.extend(root.glob("lsape-build*/provenance.json"))
        originals.extend(root.glob("*/build/provenance.json"))
    dated = []
    for binary in RUNS.rglob("_lsape_native.pyd"):
        folders = {part.lower() for part in binary.parent.relative_to(RUNS).parts}
        if not folders.intersection(("provenance", "snapshot", "build-snapshot")):
            dated.append(binary.parent / "provenance.json")
    candidates = list(dict.fromkeys(path.parent for path in cached + _recent(originals) + _recent(dated)))
    configured = os.environ.get("LSAPE_BUILD_DIR")
    if configured:
        selected = Path(configured).expanduser().resolve()
        candidates = [selected] + [path for path in candidates if path != selected]
    return candidates


def _cache_lsape(build_dir, cache_root):
    """Retain the entire validated build, including its bound builder snapshot."""
    build_dir = build_dir.resolve()
    if build_dir.is_relative_to(cache_root):
        return build_dir
    output = _fresh_directory(cache_root, "lsape-build")
    try:
        shutil.copytree(build_dir, output, dirs_exist_ok=True)
        _validate_lsape(output)
    except (OSError, PreparationError) as error:
        raise PreparationError(f"Cannot retain the LSAPE build in {output}: {error}") from error
    return output.resolve()


def _validate_lsape(build_dir):
    from .lsape import controlled_comparison as common
    from .lsape.build_provenance import verified_builder
    build_dir = Path(build_dir)
    try:
        record = json.loads((build_dir / "provenance.json").read_text(encoding="utf-8"))
        if record.get("python_version", "").split()[0] != platform.python_version():
            raise PreparationError("LSAPE build belongs to a different Python patch version")
        if common.digest(build_dir / "_lsape_native.pyd") != record.get("output_sha256"):
            raise PreparationError("LSAPE native artifact hash changed")
        builder, _ = verified_builder(build_dir, common.ROOT, record)
        candidates = common.source_files()
        declared = record.get("input_sha256", {})
        required = {"lsape_bridge.cpp", "build_bridge.py", "lsape-v4.zip", "hungarian-lsape.hh",
                    "src/specialized/hungarian-lsape-uniform.hh"}
        if not required <= declared.keys():
            raise PreparationError("LSAPE build omits required source provenance")
        for name, expected in declared.items():
            source = builder if name == "build_bridge.py" else common.resolve_build_input(name, candidates)
            if common.digest(source) != expected:
                raise PreparationError(f"LSAPE compiled input changed: {name}")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, RuntimeError) as error:
        raise PreparationError(f"Incompatible LSAPE build: {build_dir}: {error}") from error
    code = """import sys
from benchmark.lsape.controlled_comparison import load_runtime
from benchmark.lsape.rectangular_campaign import METHODS
bridge, *_ = load_runtime(sys.argv[1])
assert all(callable(getattr(bridge, method[0])) for method in METHODS.values())
assert callable(bridge.diagnose)
"""
    _run([sys.executable, "-B", "-c", code, str(build_dir)], capture=True)


def _build_lsape(output):
    _run([sys.executable, "-B", "-m", "benchmark.lsape", "build", "--output-dir", output])
    return output


def prepare_lsape(cache_root: Path) -> Path:
    """Find or prepare the separate Windows/MSVC LSAPE bridge."""
    if platform.system() != "Windows" or struct.calcsize("P") != 8:
        raise PreparationError("The LSAPE experiment requires 64-bit Windows and MSVC")
    cache_root = Path(cache_root).expanduser().resolve()
    for build_dir in _lsape_candidates(cache_root):
        try:
            _validate_lsape(build_dir)
        except PreparationError:
            continue
        cached = _cache_lsape(build_dir, cache_root)
        print(f"Using LSAPE build: {cached}", flush=True)
        return cached
    _check_toolchains(campaign=False)
    output = _build_lsape(_fresh_directory(cache_root, "lsape-build"))
    _validate_lsape(output)
    return output.resolve()
