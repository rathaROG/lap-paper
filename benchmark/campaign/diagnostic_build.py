"""Separate instrumented native builds; never eligible for primary timings.

The counters are integer event counts/maxima, not elapsed-time measurements.
Every artifact starts from the same pinned source and compile-time policy as its
primary counterpart, then adds this explicitly separate diagnostic patch.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import time

from . import native_build as primary


COUNTERS = (
    "initialization_row_scans", "search_row_scans", "initialization_displacements",
    "initialization_chain_starts", "augmentations", "rejection_endings",
    "orientation_capacity_before_solve_max_bytes", "orientation_capacity_after_solve_max_bytes",
    "orientation_capacity_before_cleanup_max_bytes", "orientation_capacity_after_cleanup_max_bytes",
    "orientation_cleanup_calls", "orientation_releases", "problems",
    "orientation_nonzero_before_solve_count", "orientation_nonzero_after_cleanup_count",
    "fresh_state_drops", "fresh_worker_drops",
)
COUNTER_DEFINITIONS = {
    "scope": "Rectangular/optional policy event counters exclude the separate required-square algorithm; problem and workspace counters cover both cores.",
    "initialization_row_scans": "One per bounded row-minimum scan, including immediate displacement retries.",
    "search_row_scans": "One per completion-search row-label relaxation; root labels reuse the initial dual-tightening scan, so that scan is not counted twice.",
    "initialization_displacements": "Number of occupied rows actually displaced by the final initializer choice.",
    "initialization_chain_starts": "Initializer iterations other than an immediate retry of the just-displaced row; continuation resets at each pass.",
    "augmentations": "Completion searches started for remaining free rows, including searches ending in rejection.",
    "rejection_endings": "Initializer dummy selections plus completion searches ending at the implicit rejection terminal.",
    "capacity": "Actual allocated vector capacity in bytes, aggregated as maxima across worker threads and calls; maxima are not totals or per-worker traces.",
    "cleanup": "Stock release_large_buffers calls and W2 local-worker destruction, counted separately from solver calls. W1 retains orientation at stock cleanup boundaries.",
    "reset": "Reset counters only while no solve is active. Reset does not clear cached worker buffers/pools; use a fresh process for a cold diagnostic.",
    "timing": "Instrumentation uses relaxed atomic counters; diagnostic elapsed times are not primary timing observations.",
}
HEADER = '''#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void campaign_diag_reset(void);
void campaign_diag_snapshot(unsigned long long* output);
#ifdef __cplusplus
}
#endif
'''


def patch_rust(root: Path):
    changes = {}
    def edit(relative, transform):
        path = root / relative
        before = path.read_text(encoding="utf-8")
        after = transform(before)
        path.write_text(after, encoding="utf-8", newline="\n")
        changes[relative] = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                    fromfile="controlled/" + relative, tofile="diagnostic/" + relative))
    replace = primary._replace_once
    module = '''use std::sync::atomic::{AtomicU64, Ordering};
static COUNTERS: [AtomicU64; 17] = [
''' + ",\n".join("    AtomicU64::new(0)" for _ in COUNTERS) + '''
];
pub fn add(index: usize, value: u64) { COUNTERS[index].fetch_add(value, Ordering::Relaxed); }
pub fn maximum(index: usize, value: usize) {
    COUNTERS[index].fetch_max(value as u64, Ordering::Relaxed);
}
pub fn before_solve(capacity: usize) {
    add(12, 1); maximum(6, capacity); if capacity > 0 { add(13, 1); }
}
pub fn after_solve(capacity: usize) { maximum(7, capacity); }
pub fn before_cleanup(capacity: usize) { add(10, 1); maximum(8, capacity); }
pub fn after_cleanup(before: usize, after: usize) {
    maximum(9, after);
    if before > 0 && after == 0 { add(11, 1); }
    if after > 0 { add(14, 1); }
}
#[no_mangle]
pub extern "C" fn campaign_diag_reset() {
    for value in &COUNTERS { value.store(0, Ordering::Relaxed); }
}
#[no_mangle]
pub unsafe extern "C" fn campaign_diag_snapshot(output: *mut u64) {
    for (i, value) in COUNTERS.iter().enumerate() { *output.add(i) = value.load(Ordering::Relaxed); }
}
'''
    module_path = "src/_lapjv/src/campaign_diag.rs"
    (root / module_path).write_text(module, encoding="utf-8")
    changes[module_path] = module
    header_path = "src/_lapjv/campaign_diag.h"
    (root / header_path).write_text(HEADER, encoding="utf-8")
    changes[header_path] = HEADER
    def lib(text):
        text = replace(text, "mod campaign_policy;", "mod campaign_policy;\nmod campaign_diag;")
        text = replace(text, "    fn release_large_buffers(&mut self) {",
            "    fn release_large_buffers(&mut self) {\n"
            "        let diag_before = self.orientation.capacity() * std::mem::size_of::<T>();\n"
            "        campaign_diag::before_cleanup(diag_before);")
        text = replace(text,
            "        if self.rectangular.y.capacity() > 65536 {\n            self.rectangular = rectangular::Rectangular::default();\n        }",
            "        if self.rectangular.y.capacity() > 65536 {\n            self.rectangular = rectangular::Rectangular::default();\n        }\n"
            "        campaign_diag::after_cleanup(diag_before, self.orientation.capacity() * std::mem::size_of::<T>());")
        return text
    edit("src/_lapjv/src/lib.rs", lib)
    def rectangular(text):
        replacements = (
            ("            while k < previous {", "            let mut diag_continuation = false;\n            while k < previous {\n"
             "                super::campaign_diag::add(0, 1);\n"
             "                if !diag_continuation { super::campaign_diag::add(3, 1); }\n"
             "                diag_continuation = false;"),
            ("\n                if OPTIONAL && j1 == FREE {", "\n                if OPTIONAL && j1 == FREE {\n"
             "                    super::campaign_diag::add(5, 1);"),
            ("\n                    if OPTIONAL && j1 == FREE {", "\n                    if OPTIONAL && j1 == FREE {\n"
             "                        super::campaign_diag::add(5, 1);"),
            ("                if displaced != FREE {\n                    if lowers {", "                if displaced != FREE {\n"
             "                    super::campaign_diag::add(2, 1);\n                    if lowers {\n"
             "                        diag_continuation = true;"),
            ("        for &root in free.iter().take(num_free) {", "        for &root in free.iter().take(num_free) {\n"
             "            super::campaign_diag::add(4, 1);"),
            ("            let sink = loop {", "            let sink = loop {\n                super::campaign_diag::add(1, 1);"),
            ("                if OPTIONAL && dummy_distance.finite() && dummy_distance <= best {",
             "                if OPTIONAL && dummy_distance.finite() && dummy_distance <= best {\n"
             "                    super::campaign_diag::add(5, 1);"),
        )
        for old, new in replacements:
            text = replace(text, old, new)
        return text
    edit("src/_lapjv/src/rectangular.rs", rectangular)
    def ffi(text):
        text = replace(text,
            "            worker.orientation = std::mem::take(&mut fresh.orientation);\n            result",
            "            worker.orientation = std::mem::take(&mut fresh.orientation);\n"
            "            super::campaign_diag::add(15, 1);\n            result")
        text = replace(text,
            "            let mut fresh = Worker::default();\n            solve_one_inner(&mut fresh, input, original, base, p, output)",
            "            let mut fresh = Worker::default();\n"
            "            let result = solve_one_inner(&mut fresh, input, original, base, p, output);\n"
            "            let capacity = fresh.orientation.capacity() * std::mem::size_of::<T>();\n"
            "            super::campaign_diag::before_cleanup(capacity);\n            drop(fresh);\n"
            "            super::campaign_diag::after_cleanup(capacity, 0);\n"
            "            super::campaign_diag::add(16, 1);\n            result")
        text = replace(text, "    #[cfg(test)]\n    test_faults::hit(test_faults::Site::Panic)?;",
            "    super::campaign_diag::before_solve(worker.orientation.capacity() * std::mem::size_of::<T>());\n"
            "    #[cfg(test)]\n    test_faults::hit(test_faults::Site::Panic)?;")
        text = replace(text, "    Ok((count, total))\n}",
            "    super::campaign_diag::after_solve(orientation.capacity() * std::mem::size_of::<T>());\n"
            "    Ok((count, total))\n}")
        return text
    edit("src/_lapjv/src/ffi.rs", ffi)
    return changes


def _patch_wrapper(root):
    path = root / "src/_lapjv/_lapjv.pyx"
    extra = '''

# Instrumented diagnostics only: unavailable in all primary modules.
cdef extern from "campaign_diag.h" nogil:
    void campaign_diag_reset()
    void campaign_diag_snapshot(unsigned long long*)

def campaign_diagnostic_reset():
    campaign_diag_reset()

def campaign_diagnostic_snapshot():
    cdef unsigned long long values[17]
    campaign_diag_snapshot(values)
    return tuple(values[i] for i in range(17))
'''
    with path.open("a", encoding="utf-8") as stream:
        stream.write(extra)
    return {"src/_lapjv/_lapjv.pyx": extra}


def build_one(language, arm, repository, output, python=sys.executable):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    identity = json.loads(primary._capture([python, "-c", "import sys,sysconfig,json;print(json.dumps({"
        "'executable':sys.executable,'version':sys.version,'cache_tag':sys.implementation.cache_tag,"
        "'platform':sysconfig.get_platform(),'soabi':sysconfig.get_config_var('SOABI')}))"]))
    tag = identity["cache_tag"] + "-" + identity["platform"]
    build_root = output / tag / f"{language}-{arm}"
    record_path = build_root / "build-record.json"
    builder_hash = primary.sha256_file(Path(__file__))
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["builder_sha256"] != builder_hash or primary.sha256_file(Path(record["artifact_path"])) != record["artifact_sha256"]:
            raise primary.NativeBuildError("Diagnostic builder/artifact changed; use a new output directory")
        return record
    if build_root.exists():
        raise primary.NativeBuildError(f"Incomplete diagnostic build: choose a new output directory: {build_root}")
    commit = primary.SOURCE_COMMITS[language]
    if primary._capture(["git", "-C", str(repository), "rev-parse", commit + "^{commit}"]) != commit:
        raise primary.NativeBuildError("Missing pinned diagnostic source commit")
    archive = subprocess.check_output(["git", "-C", str(repository), "archive", "--format=tar", commit])
    source = build_root / "source"
    primary._safe_extract(archive, source)
    namespace = f"campaign_diag_{language}_{arm}"
    changes = primary.patch_source(source, language, arm, namespace)
    primary_patch = build_root / "controlled.patch"
    primary_patch.write_text("\n".join(changes.values()), encoding="utf-8")
    if language == "rust":
        diagnostic_changes = patch_rust(source)
    else:
        from .diagnostic_cpp import patch_cpp
        diagnostic_changes = patch_cpp(source)
    diagnostic_changes.update(_patch_wrapper(source))
    patch = build_root / "diagnostic.patch"
    patch.write_text("\n".join(diagnostic_changes.values()), encoding="utf-8")
    env, env_record = primary.controlled_environment(output)
    log = build_root / "build.log"
    command = [python, "setup.py", "build_ext", "--inplace"]
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        status = subprocess.run(command, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if status.returncode:
        raise primary.NativeBuildError(f"Diagnostic build failed: {log}\n" + log.read_text(encoding="utf-8", errors="replace")[-3000:])
    artifacts = [p for p in (source / namespace).iterdir() if p.name.startswith("_lapjv.") and p.suffix in (".pyd", ".so")]
    if len(artifacts) != 1:
        raise primary.NativeBuildError("Expected exactly one diagnostic native extension")
    artifact = artifacts[0]
    recipes = build_root / "recipe-source"
    recipes.mkdir()
    recipe_files = [Path(__file__), Path(primary.__file__)]
    if language == "cpp":
        recipe_files.append(Path(__file__).with_name("diagnostic_cpp.py"))
    for path in recipe_files:
        shutil.copy2(path, recipes / path.name)
    record = {"schema_version": 1, "language": language, "arm_id": arm,
              "diagnostic": True, "profile_enabled": True, "rust_profile_feature": False,
              "runtime_policy_overrides": False, "release": True, "eligible_for_primary_timing": False,
              "policy": primary.policy_for(arm), "source_commit": commit,
              "source_archive_sha256": hashlib.sha256(archive).hexdigest(), "source_copy": str(source),
              "package_name": namespace, "package_root": str(source), "artifact_path": str(artifact),
              "artifact_sha256": primary.sha256_file(artifact), "python": identity,
              "builder_sha256": builder_hash, "controlled_builder_sha256": primary.sha256_file(Path(primary.__file__)),
              "controlled_patch_path": str(primary_patch), "controlled_patch_sha256": primary.sha256_file(primary_patch),
              "diagnostic_patch_path": str(patch), "diagnostic_patch_sha256": primary.sha256_file(patch),
              "build_command": command, "build_environment": env_record,
              "build_log": str(log), "build_log_sha256": primary.sha256_file(log),
              "build_seconds": time.perf_counter() - started, "counter_names": COUNTERS,
              "counter_definitions": COUNTER_DEFINITIONS,
              "changed_source_sha256": {name: primary.sha256_file(source / name) for name in set(changes) | set(diagnostic_changes)}}
    if language == "rust":
        record["rustc"] = primary._capture(["rustc", "-vV"])
        record["cargo"] = primary._capture(["cargo", "--version"])
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


class DiagnosticSolver:
    def __init__(self, module, extension, record):
        self.module, self.extension, self.build_record = module, extension, record

    def solve(self, cost, *, limit=float("inf"), batch=False, workers=1):
        kwargs = dict(extend_cost=True, cost_limit=limit, return_cost=False,
                      precision="float64", output="pairs")
        return (self.module.lapjv_batch(cost, n_threads=workers, **kwargs) if batch else
                self.module.lapjv(cost, **kwargs))

    def reset(self):
        self.extension.campaign_diagnostic_reset()

    def snapshot(self):
        return dict(zip(COUNTERS, self.extension.campaign_diagnostic_snapshot()))


def load_diagnostic(record):
    if record.get("diagnostic") is not True or record.get("eligible_for_primary_timing") is not False:
        raise primary.NativeBuildError("Diagnostic loader requires an explicitly instrumented build")
    identity = record["python"]
    if identity["cache_tag"] != sys.implementation.cache_tag or identity["platform"] != sysconfig.get_platform():
        raise primary.NativeBuildError("Diagnostic extension ABI/platform mismatch")
    name = record["package_name"]
    if name != f"campaign_diag_{record['language']}_{record['arm_id']}":
        raise primary.NativeBuildError("Unexpected diagnostic namespace")
    artifact = Path(record["artifact_path"]).resolve()
    root = Path(record["package_root"]).resolve()
    if artifact.parent != root / name or primary.sha256_file(artifact) != record["artifact_sha256"]:
        raise primary.NativeBuildError("Diagnostic artifact identity mismatch")
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve().parent != artifact.parent:
        raise primary.NativeBuildError("Another diagnostic build is loaded under this namespace; start a new process")
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(name)
        extension = importlib.import_module(name + "._lapjv")
    finally:
        sys.path.pop(0)
    if Path(extension.__file__).resolve() != artifact:
        raise primary.NativeBuildError("Loaded diagnostic differs from indexed extension")
    return DiagnosticSolver(module, extension, record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-source", type=Path)
    parser.add_argument("--cpp-source", type=Path)
    parser.add_argument("--output", type=Path,
                        help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-diagnostic-builds")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--languages", nargs="+", choices=("rust", "cpp"), default=["rust", "cpp"])
    parser.add_argument("--arms", nargs="+", choices=primary.ALL_ARMS, default=list(primary.ALL_ARMS))
    args = parser.parse_args(argv)
    if args.output is None:
        from ..paths import default_output
        args.output = default_output("campaign-diagnostic-builds")
        print(f"Output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "diagnostic-build-index.json"
    records = json.loads(index_path.read_text(encoding="utf-8"))["builds"] if index_path.exists() else []
    for language in args.languages:
        repository = getattr(args, language + "_source")
        if repository is None:
            parser.error(f"--{language}-source is required")
        for arm in args.arms:
            print(f"Building diagnostic {language}/{arm}", flush=True)
            record = build_one(language, arm, repository, args.output, args.python)
            records = [r for r in records if (r["language"], r["arm_id"]) != (language, arm)] + [record]
            index_path.write_text(json.dumps({"schema_version": 1, "diagnostic": True,
                                             "eligible_for_primary_timing": False, "builds": records}, indent=2) + "\n", encoding="utf-8")
            print(f"Ready diagnostic {language}/{arm}: {record['build_seconds']:.2f} s", flush=True)
    print(index_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
