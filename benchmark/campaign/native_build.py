"""Reproducible, isolated release builds for the controlled native experiments.

This module never modifies or installs into a source repository.  Each build
starts with ``git archive`` of a pinned commit and uses an independent Python
package namespace, so a case worker can interleave all variants in one process.
The copied sources, exact patch, compiler log, and artifact hashes are retained.
Only compile-time policies are changed; the profile feature is prohibited.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import importlib.machinery
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import time
from typing import Any


SOURCE_COMMITS = {
    "rust": "a99cebd468b35b0555716d21f87d9c44f84261f9",
    "cpp": "60d4e8ea9aeddf24d242c835919bb3febd013d8a",
}
POLICY_ARMS = (
    "reference", "budget1", "budget2", "budget8", "ties_off",
    "cutoff_off", "ties_cutoff_off", "no_reductions",
)
WORKSPACE_ARMS = ("fresh_state", "fresh_worker")
ALL_ARMS = POLICY_ARMS + WORKSPACE_ARMS
WORKSPACE_ALIASES = {"w0": "reference", "w1": "fresh_state", "w2": "fresh_worker"}
SCHEMA_VERSION = 1


class NativeBuildError(ValueError):
    """A source, build, or loader invariant did not hold."""


def policy_for(arm: str) -> dict[str, Any]:
    if arm not in ALL_ARMS:
        raise NativeBuildError(f"Unknown native arm: {arm}")
    return {
        "budget": int(arm.removeprefix("budget")) if arm.startswith("budget") else 4,
        "free_ties": arm not in ("ties_off", "ties_cutoff_off"),
        "required_cutoff": arm not in ("cutoff_off", "ties_cutoff_off"),
        "reduction_passes": 0 if arm == "no_reductions" else 2,
        "workspace": {"fresh_state": 1, "fresh_worker": 2}.get(arm, 0),
    }


def eligible_arms(optional: bool) -> tuple[str, ...]:
    return tuple(arm for arm in POLICY_ARMS if not optional or arm not in (
        "cutoff_off", "ties_cutoff_off"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise NativeBuildError(f"Pinned source patch must match once: {old[:110]!r}")
    return text.replace(old, new, 1)


def _safe_extract(archive: bytes, destination: Path) -> None:
    """Extract regular files only; reject escapes and links before any writes."""
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        members = stream.getmembers()
        for member in members:
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or not (member.isfile() or member.isdir()):
                raise NativeBuildError(f"Unsafe archive entry: {member.name}")
        for member in members:
            target = root / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise NativeBuildError(f"Cannot read archived file: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o777)


def _rust_wrapper() -> str:
    return '''// Campaign scaffold: every arm, including the reference, uses this wrapper.
// Constant matching removes unused workspace branches from release builds.
fn solve_one<T: Cost>(
    worker: &mut Worker<T>, input: &[T], original: &Original<'_>, base: usize,
    p: &Problem, output: &mut [i64],
) -> Result<(usize, f64), Error> {
    match super::campaign_policy::WORKSPACE {
        1 => {
            let mut fresh = Worker {
                orientation: std::mem::take(&mut worker.orientation),
                ..Worker::default()
            };
            // Restore the orientation even when solve_one_inner reports an error.
            let result = solve_one_inner(&mut fresh, input, original, base, p, output);
            worker.orientation = std::mem::take(&mut fresh.orientation);
            result
        }
        2 => {
            let mut fresh = Worker::default();
            solve_one_inner(&mut fresh, input, original, base, p, output)
        }
        _ => solve_one_inner(worker, input, original, base, p, output),
    }
}

'''


def _cpp_wrapper() -> str:
    return '''// Campaign scaffold: the reference uses the same compile-time wrapper.
template<class T> void solve_one(Worker<T>& retained, const T* costs, const void* original,
    bool original32, size_t rows, size_t cols, double limit, bool pairs, bool return_cost,
    int64_t* output, size_t& count, double& total) {
    if constexpr (campaign_policy::workspace == 1) {
        Worker<T> fresh;
        fresh.orientation.swap(retained.orientation);
        struct RestoreOrientation {
            std::vector<T>& retained;
            std::vector<T>& fresh;
            ~RestoreOrientation() { retained.swap(fresh); }
        } restore{retained.orientation, fresh.orientation};
        solve_one_inner(fresh, costs, original, original32, rows, cols, limit,
                       pairs, return_cost, output, count, total);
    } else if constexpr (campaign_policy::workspace == 2) {
        Worker<T> fresh;
        solve_one_inner(fresh, costs, original, original32, rows, cols, limit,
                       pairs, return_cost, output, count, total);
    } else {
        solve_one_inner(retained, costs, original, original32, rows, cols, limit,
                       pairs, return_cost, output, count, total);
    }
}

'''


def patch_source(root: Path, language: str, arm: str, package_name: str) -> dict[str, str]:
    """Patch only the archived copy. Exact-match guards reject unexpected sources."""
    policy = policy_for(arm)
    package = "lap" + language
    changes: dict[str, str] = {}

    def edit(relative: str, transform) -> None:
        path = root / relative
        before = path.read_text(encoding="utf-8")
        after = transform(before)
        path.write_text(after, encoding="utf-8", newline="\n")
        changes[relative] = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile="original/" + relative, tofile="campaign/" + relative))

    def setup_patch(text: str) -> str:
        # Rust's extension selector and its declaration must both be renamed.
        expected = 4 if language == "rust" else 2
        if text.count(package + "._lapjv") != expected:
            raise NativeBuildError("Unexpected native extension declarations")
        text = text.replace(package + "._lapjv", package_name + "._lapjv")
        text = _replace_once(text, f'name="{package}", version=version,',
                             f'name="{package_name}", version=version,')
        text = _replace_once(text,
            f'packages=["{package}"], package_data={{"{package}":',
            f'packages=["{package_name}"], package_data={{"{package_name}":')
        if language == "rust":
            text = _replace_once(text,
                "target_dir = Path(self.build_temp).resolve() / 'laprust-core'",
                "target_dir = Path(os.environ['CAMPAIGN_CARGO_TARGET']).resolve()")
        return text

    edit("setup.py", setup_patch)
    shutil.copytree(root / package, root / package_name)
    if language == "rust":
        generated = "// Generated controlled, uninstrumented release policy.\n" + "\n".join([
            f"pub const BUDGET: usize = {policy['budget']};",
            f"pub const FREE_TIES: bool = {str(policy['free_ties']).lower()};",
            f"pub const REQUIRED_CUTOFF: bool = {str(policy['required_cutoff']).lower()};",
            f"pub const REDUCTION_PASSES: usize = {policy['reduction_passes']};",
            f"pub const WORKSPACE: usize = {policy['workspace']};", ""])
        generated_path = "src/_lapjv/src/campaign_policy.rs"
        (root / generated_path).write_text(generated, encoding="utf-8")
        changes[generated_path] = generated
        edit("src/_lapjv/src/lib.rs", lambda text: _replace_once(text, "mod ffi;",
            'mod campaign_policy;\n#[cfg(feature = "profile")]\n'
            'compile_error!("Campaign primary builds must disable profiling");\nmod ffi;'))
        def rectangular(text: str) -> str:
            start = text.index('        #[cfg(feature = "profile")]\n        let reduction_limit')
            end = text.index("        // Stop lowering after", start)
            text = text[:start] + (
                "        let reduction_limit = cols.min(super::campaign_policy::BUDGET);\n"
                "        let prefer_free_ties = super::campaign_policy::FREE_TIES;\n") + text[end:]
            text = _replace_once(text,
                "let window = if OPTIONAL { usize::MAX } else { rows.max(8) };",
                "let window = if OPTIONAL || !super::campaign_policy::REQUIRED_CUTOFF "
                "{ usize::MAX } else { rows.max(8) };")
            return _replace_once(text, "        for _ in 0..2 {",
                "        for _ in 0..super::campaign_policy::REDUCTION_PASSES {")
        edit("src/_lapjv/src/rectangular.rs", rectangular)
        edit("src/_lapjv/src/ffi.rs", lambda text: _replace_once(text,
            "fn solve_one<T: Cost>(", _rust_wrapper() + "fn solve_one_inner<T: Cost>("))
    elif language == "cpp":
        generated = (
            "// Generated controlled, uninstrumented release policy.\n#pragma once\n"
            "#ifdef LAPCPP_TESTING\n#error Campaign builds must disable test instrumentation\n#endif\n"
            "namespace campaign_policy {\n" + "\n".join([
                f"constexpr unsigned budget = {policy['budget']};",
                f"constexpr bool free_ties = {str(policy['free_ties']).lower()};",
                f"constexpr bool required_cutoff = {str(policy['required_cutoff']).lower()};",
                f"constexpr int reduction_passes = {policy['reduction_passes']};",
                f"constexpr int workspace = {policy['workspace']};", "}\n"]))
        generated_path = "src/_lapjv/campaign_policy.h"
        (root / generated_path).write_text(generated, encoding="utf-8")
        changes[generated_path] = generated
        def cpp_patch(text: str) -> str:
            for before, after in (
                ('#include "lapjv.h"', '#include "lapjv.h"\n#include "campaign_policy.h"'),
                ("const size_t reduction_limit = std::min(cols,size_t(4));",
                 "const size_t reduction_limit = std::min(cols,size_t(campaign_policy::budget));"),
                ("const size_t window = Optional ? FREE : std::max(rows,size_t(8));",
                 "const size_t window = (Optional || !campaign_policy::required_cutoff) "
                 "? FREE : std::max(rows,size_t(8));"),
                ("bool reduction_active = true;\n        for (int pass = 0; pass < 2; ++pass) {",
                 "bool reduction_active = true;\n        for (int pass = 0; pass < campaign_policy::reduction_passes; ++pass) {"),
                ("if (m.first == m.second && yy[j1] != FREE) {",
                 "if (campaign_policy::free_ties && m.first == m.second && yy[j1] != FREE) {"),
                ("template<class T> void solve_one(Worker<T>& w,",
                 "template<class T> void solve_one_inner(Worker<T>& w,"),
                ("template<class T> int run(const T* costs,", _cpp_wrapper() +
                 "template<class T> int run(const T* costs,"),
            ):
                text = _replace_once(text, before, after)
            return text
        edit("src/_lapjv/solver.cpp", cpp_patch)
    else:
        raise NativeBuildError(f"Unknown language: {language}")
    return changes


def controlled_environment(output: Path) -> tuple[dict[str, str], dict[str, Any]]:
    env = os.environ.copy()
    removed = []
    exact = {"CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "CL", "_CL_", "LINK", "_LINK_",
             "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_BUILD_TARGET"}
    for name in list(env):
        if name.upper() in exact or name.upper().startswith((
                "CARGO_PROFILE_", "LAPRUST_", "LAPCPP_", "CARGO_TARGET_")):
            removed.append(name)
            del env[name]
    fixed = {"LAPRUST_BASEOPTS": "1", "LAPRUST_NATIVE": "0", "LAPRUST_LTO": "1",
             "LAPCPP_BASEOPTS": "1", "LAPCPP_NATIVE": "0", "LAPCPP_LTO": "1",
             "CAMPAIGN_CARGO_TARGET": str(output / "cargo-target"),
             "PYTHONHASHSEED": "0"}
    env.update(fixed)
    return env, {"fixed": fixed, "removed_variable_names": sorted(removed)}


def _capture(command: list[str], **kwargs) -> str:
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise NativeBuildError(f"Command failed: {command[0]}: {result.stderr[-2000:]}")
    return result.stdout.strip()


def build_one(language: str, arm: str, repository: Path, output: Path,
              python: str = sys.executable) -> dict[str, Any]:
    """Build one arm. Existing successful artifacts are verified and reused."""
    policy = policy_for(arm)
    if language not in SOURCE_COMMITS:
        raise NativeBuildError(f"Unknown language: {language}")
    repository, output = repository.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = json.loads(_capture([python, "-c", "import sys,sysconfig,platform,json; "
        "print(json.dumps({'executable':sys.executable,'version':sys.version,"
        "'cache_tag':sys.implementation.cache_tag,'platform':sysconfig.get_platform(),"
        "'soabi':sysconfig.get_config_var('SOABI'),'machine':platform.machine()}))"]))
    tag = identity["cache_tag"] + "-" + identity["platform"]
    build_root = output / tag / f"{language}-{arm}"
    metadata = build_root / "build-record.json"
    if metadata.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("builder_sha256") != sha256_file(Path(__file__)):
            raise NativeBuildError(f"Builder changed; use a fresh output directory: {build_root}")
        if sha256_file(Path(record["artifact_path"])) != record["artifact_sha256"]:
            raise NativeBuildError(f"Existing artifact hash changed: {build_root}")
        return record
    if build_root.exists():
        raise NativeBuildError(f"Incomplete build exists; choose a new output directory: {build_root}")
    commit = SOURCE_COMMITS[language]
    resolved = _capture(["git", "-C", str(repository), "rev-parse", commit + "^{commit}"])
    if resolved != commit:
        raise NativeBuildError(f"Pinned source commit missing: {commit}")
    archive = subprocess.run(["git", "-C", str(repository), "archive", "--format=tar", commit],
                             capture_output=True, check=True).stdout
    source = build_root / "source"
    _safe_extract(archive, source)
    namespace = f"campaign_{language}_{arm}"
    changes = patch_source(source, language, arm, namespace)
    patch = build_root / "campaign.patch"
    patch.write_text("\n".join(changes.values()), encoding="utf-8")
    env, env_record = controlled_environment(output)
    build_log = build_root / "build.log"
    command = [python, "setup.py", "build_ext", "--inplace"]
    started = time.perf_counter()
    with build_log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=source, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        tail = build_log.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise NativeBuildError(f"Build failed ({language}/{arm}), see {build_log}\n{tail}")
    artifacts = [path for path in (source / namespace).iterdir()
                 if path.name.startswith("_lapjv.") and path.suffix in (".pyd", ".so")]
    if len(artifacts) != 1:
        raise NativeBuildError(f"Expected one native extension, got {artifacts}")
    artifact = artifacts[0]
    record = {
        "schema_version": SCHEMA_VERSION, "language": language, "arm_id": arm,
        "policy": policy, "workspace": ("stock", "fresh_state", "fresh_worker")[policy["workspace"]],
        "source_commit": commit, "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "source_repository": str(repository), "source_copy": str(source),
        "builder_sha256": sha256_file(Path(__file__)), "patch_sha256": sha256_file(patch),
        "patch_path": str(patch), "profile_enabled": False, "test_instrumentation": False,
        "runtime_policy_overrides": False, "release": True, "package_name": namespace,
        "package_root": str(source), "artifact_path": str(artifact),
        "artifact_sha256": sha256_file(artifact), "python": identity,
        "build_command": command, "build_environment": env_record,
        "build_log": str(build_log), "build_log_sha256": sha256_file(build_log),
        "build_seconds": time.perf_counter() - started,
        "changed_source_sha256": {name: sha256_file(source / name) for name in changes},
        "compiler_flags_evidence": "Exact compiler/linker invocations are retained in build_log",
    }
    if language == "rust":
        record["rustc"] = _capture(["rustc", "-vV"])
        record["cargo"] = _capture(["cargo", "--version"])
    metadata.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def resolve_build(index: dict[str, Any] | str | Path, language: str, arm: str) -> dict[str, Any]:
    if isinstance(index, (str, Path)):
        index = json.loads(Path(index).read_text(encoding="utf-8"))
    arm = WORKSPACE_ALIASES.get(arm, arm)
    matches = [record for record in index.get("builds", [])
               if record.get("language") == language and record.get("arm_id") == arm]
    if len(matches) != 1:
        raise NativeBuildError(f"Need exactly one build for {language}/{arm}; got {len(matches)}")
    return matches[0]


def load_solver(record: dict[str, Any]):
    """Return ``solve(cost, *, limit=inf, batch=False, workers=1)``.

    Loading and integrity checks occur before timing; each call invokes precisely
    the public native API with float64, pairs output, and total calculation off.
    Wrapper conversion/output buffers remain in all measured workspace arms.
    """
    if record.get("profile_enabled") is not False or record.get("runtime_policy_overrides") is not False:
        raise NativeBuildError("Primary timing requires an uninstrumented compile-time build")
    identity = record.get("python", {})
    if identity.get("cache_tag") != sys.implementation.cache_tag or identity.get("platform") != sysconfig.get_platform():
        raise NativeBuildError("Native build Python ABI/platform does not match this interpreter")
    artifact = Path(record["artifact_path"]).resolve()
    if not artifact.is_file() or sha256_file(artifact) != record["artifact_sha256"]:
        raise NativeBuildError(f"Native artifact missing or hash mismatch: {artifact}")
    namespace = record["package_name"]
    expected = f"campaign_{record['language']}_{record['arm_id']}"
    if namespace != expected or not namespace.isidentifier():
        raise NativeBuildError("Unexpected native package namespace")
    root = Path(record["package_root"]).resolve()
    if artifact.parent != root / namespace:
        raise NativeBuildError("Artifact does not belong to the indexed package")
    existing = sys.modules.get(namespace)
    if existing is not None and Path(existing.__file__).resolve().parent != artifact.parent:
        raise NativeBuildError(f"Different build already loaded under {namespace}; use a fresh process")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(namespace)
    finally:
        sys.path.pop(0)
    loaded = sys.modules[namespace + "._lapjv"]
    if Path(loaded.__file__).resolve() != artifact:
        raise NativeBuildError("Loaded extension differs from indexed artifact")

    def solve(cost, *, limit=float("inf"), batch=False, workers=1):
        options = dict(extend_cost=True, cost_limit=limit, return_cost=False,
                       precision="float64", output="pairs")
        if batch:
            return module.lapjv_batch(cost, n_threads=workers, **options)
        return module.lapjv(cost, **options)

    solve.build_record = record
    return solve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-source", type=Path)
    parser.add_argument("--cpp-source", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-builds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--languages", nargs="+", choices=tuple(SOURCE_COMMITS), default=list(SOURCE_COMMITS))
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    args = parser.parse_args(argv)
    if args.output is None:
        from ..paths import default_output
        args.output = default_output("campaign-builds")
        print(f"Output: {args.output}")
    records = []
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output.resolve() / "build-index.json"
    if index_path.exists():
        records = json.loads(index_path.read_text(encoding="utf-8")).get("builds", [])
    for language in args.languages:
        repository = getattr(args, language + "_source")
        if repository is None:
            parser.error(f"--{language}-source is required for selected language")
        for arm in args.arms:
            print(f"Building {language}/{arm}", flush=True)
            record = build_one(language, arm, repository, args.output, args.python)
            records = [item for item in records if (item["language"], item["arm_id"]) != (language, arm)]
            records.append(record)
            index_path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "builds": records}, indent=2) + "\n", encoding="utf-8")
            print(f"Ready {language}/{arm}: {record['build_seconds']:.2f} s", flush=True)
    print(str(index_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
