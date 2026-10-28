"""Collect a sequential, reproducible real-sequence tracking comparison.

Run input preparation first, then this program with a fresh campaign directory.
Saved tracker predictions and captured application matrices are evaluated outside
all performance measurements. Existing paper collections are never touched.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from realbench_data import SEQUENCES, SOURCES, file_identity, sha256_file
from realbench_tracking import run_tracking, solve_pairs, validate_pairs

HERE = Path(__file__).resolve().parent
SOLVERS = ("lapjv", "lapjvx", "lapjvxa", "lapjvh", "laprust", "lapcpp",
           "scipy", "lapjvs", "lapjvc")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    from realbench_paths import redact
    temporary.write_text(json.dumps(redact(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prediction_hash(predictions):
    return hashlib.sha256(json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def without_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def git_identity(path):
    def command(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
    try:
        return {"path": str(path), "commit": command("rev-parse", "HEAD"),
                "status": command("status", "--porcelain").splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"path": str(path), "commit": None, "status": [], "kind": "installed package; tracker file hashes verified"}


def environment(output):
    packages = {}
    for name in ("numpy", "scipy", "lapx", "laprust", "lapcpp", "pyppbox", "rfdetr",
                 "torch", "torchvision", "motmetrics", "pandas", "psutil", "threadpoolctl"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    binaries = {}
    for name in ("lap", "laprust", "lapcpp"):
        module = importlib.import_module(name)
        directory = Path(module.__file__).parent
        binaries[name] = {"path": str(directory), "module_version": module.__version__,
                          "native_files": [file_identity(p) for p in sorted(directory.iterdir())
                                           if p.suffix in (".pyd", ".so", ".dylib")]}
    scripts = {}
    for path in sorted(HERE.glob("realbench_*.py")):
        scripts[path.name] = file_identity(path)
        target = output / "provenance" / "sources" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    from realbench_paths import pyppbox_module
    package = Path(pyppbox_module().__file__).resolve().parent
    clone = package.parent
    clone_files = {}
    for name in ("modules/trackers/centroid/__init__.py", "modules/trackers/sort/__init__.py",
                 "modules/trackers/sort/origin/sort.py", "utils/persontools.py", "utils/gttools.py",
                 "modules/detectors/rfdetr/__init__.py", "config/myconfig.py"):
        source = package / name
        clone_files[name] = file_identity(source)
        target = output / "provenance" / "pyppbox" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(),
            "cpu": platform.processor(), "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "packages": packages,
            "solver_binaries": binaries, "measurement_sources": scripts,
            "pyppbox": git_identity(clone), "pyppbox_sources": clone_files,
            "affinity": psutil.Process().cpu_affinity(), "threadpools": threadpool_info()}


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return tuple(float(x) for x in (np.median(values), np.percentile(values, 25), np.percentile(values, 75)))


def replay_block(solver, matrices, precision, cycles):
    with without_gc():
        started = time.perf_counter_ns()
        for _ in range(cycles):
            for matrix in matrices:
                solve_pairs(solver, matrix, precision)
        elapsed = time.perf_counter_ns() - started
    return elapsed / (cycles * len(matrices))


def replay_case(output, case, capture, options, order_rng):
    archive = Path(capture["capture_output"])
    with np.load(archive, allow_pickle=False) as store:
        original = [store[row["key"]] for row in capture["matrix_metadata"]]
    if not original:
        write_json(output / "replay" / (case["case"] + "_empty.json"),
                   {"reason": "Stock tracker never called assignment", "matrices": 0})
        return [], {"matrices": 0}
    shape_counts = {}
    for matrix in original:
        key = f"{matrix.shape[0]}x{matrix.shape[1]}"
        shape_counts[key] = shape_counts.get(key, 0) + 1
    summaries, validation = [], {}
    for precision in ("float32", "float64"):
        matrices = [np.ascontiguousarray(a, dtype=precision) for a in original]
        references = []
        for a in matrices:
            pairs = solve_pairs("scipy", a, precision)
            references.append(float(a[pairs[:, 0], pairs[:, 1]].sum(dtype=np.float64)))
        checks, cycles = {}, {}
        for solver in (*SOLVERS, "lapjvr"):
            gaps, mismatches, changed_pairs = [], [], 0
            for index, (a, expected) in enumerate(zip(matrices, references)):
                pairs = solve_pairs(solver, a, precision)
                validate_pairs(pairs, a.shape)
                actual = float(a[pairs[:, 0], pairs[:, 1]].sum(dtype=np.float64))
                gap = actual - expected
                gaps.append(gap)
                tolerance = (1e-7 * abs(expected) + 1e-4 if precision == "float32"
                             else 1e-10 * abs(expected) + 1e-9)
                if abs(gap) > tolerance:
                    mismatches.append({"matrix": index, "frame": capture["matrix_metadata"][index]["frame"],
                                       "shape": list(a.shape), "reference": expected, "actual": actual,
                                       "gap": gap, "tolerance": tolerance})
                if solver == "lapjvr":
                    other = solve_pairs("laprust", a, precision)
                    # Compare objective, not arbitrary selected pairs in tied optima.
                    alias_value = float(a[other[:, 0], other[:, 1]].sum(dtype=np.float64))
                    if actual != alias_value:
                        changed_pairs += 1
            checks[solver] = {"matrices": len(matrices), "objective_max_abs_gap": max(map(abs, gaps)),
                              "objective_max_gap": max(gaps), "objective_min_gap": min(gaps),
                              "objective_nonzero_gap_count": sum(x != 0 for x in gaps),
                              "objective_mismatch_count": len(mismatches), "mismatches": mismatches,
                              "laprust_alias_objective_differences": changed_pairs if solver == "lapjvr" else None}
            if solver == "lapjvr" and changed_pairs:
                raise AssertionError(f"lapjvr/laprust alias objectives differ: {case['case']}, {precision}")
            if precision == "float64" and mismatches:
                raise AssertionError(f"Float64 replay objective mismatch: {case['case']}, {solver}: {mismatches[:1]}")
            if solver in SOLVERS:
                probe = replay_block(solver, matrices, precision, 1)
                cycles[solver] = max(1, min(10000, math.ceil(options.block_seconds * 1e9 /
                                                          (probe * len(matrices)))))
        rows = []
        for round_index in range(options.rounds):
            order = list(SOLVERS)
            order_rng.shuffle(order)
            for position, solver in enumerate(order):
                elapsed = replay_block(solver, matrices, precision, cycles[solver])
                rows.append(dict(case=case["case"], precision=precision, solver=solver,
                                 round=round_index, position=position, matrices=len(matrices),
                                 cycles=cycles[solver], nanoseconds=elapsed))
        write_json(output / "replay" / f"{case['case']}_{precision}.json",
                   {"case": case["case"], "precision": precision, "samples": rows,
                    "validation": checks, "matrix_shapes": shape_counts,
                    "precision_note": "float64 widens the same captured float32 costs; cost construction is unchanged"})
        for solver in SOLVERS:
            median, q25, q75 = statistics([row["nanoseconds"] for row in rows if row["solver"] == solver])
            summaries.append({**case, "precision": precision, "solver": solver,
                              "matrices": len(matrices), "rounds": options.rounds,
                              "median_ns_per_assignment": median, "q25_ns_per_assignment": q25,
                              "q75_ns_per_assignment": q75,
                              "objective_max_abs_gap": checks[solver]["objective_max_abs_gap"],
                              "objective_mismatch_count": checks[solver]["objective_mismatch_count"]})
        validation[precision] = checks
        print(f"Replay complete: {case['case']} {precision} ({len(matrices)} matrices)", flush=True)
    for precision in ("float32", "float64"):
        baseline = next(r["median_ns_per_assignment"] for r in summaries
                        if r["precision"] == precision and r["solver"] == "scipy")
        for row in summaries:
            if row["precision"] == precision:
                row["scipy_speedup"] = baseline / row["median_ns_per_assignment"]
    return summaries, validation


def load_inputs(directory, frames, sources=SOURCES):
    """Validate selected caches before creating a measurement campaign.

    An input directory may retain unselected detection sources. Selected sources
    must each contain exactly the two expected sequences, with canonical filenames
    and matching metadata. Reference annotations and video identities must agree
    when two sources describe the same sequence.
    """
    if isinstance(sources, str):
        sources = (sources,)
    sources = tuple(sources)
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError(f"sources must be unique selections from {SOURCES}")
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError("frames must be a positive integer")
    selected = set(sources)
    caches, paths, seen = {}, [], {}
    for path in sorted(Path(directory).resolve().glob("*.json")):
        cache = json.loads(path.read_text(encoding="utf-8"))
        meta = cache.get("metadata", {})
        source, sequence = meta.get("source"), meta.get("sequence")
        if source not in SOURCES or sequence not in SEQUENCES:
            raise ValueError(f"Unexpected sequence/source metadata: {path}")
        if source not in selected:
            continue
        identity = (sequence, source)
        if identity in seen:
            raise ValueError(f"Duplicate sequence/source cache: {identity}: {seen[identity]}, {path}")
        if path.stem != f"{sequence}_{source}":
            raise ValueError(f"Cache filename disagrees with sequence/source metadata: {path}")
        if not isinstance(cache.get("frames"), list) or len(cache["frames"]) != frames or meta.get("frames") != frames:
            raise ValueError(f"Unexpected frame coverage: {path}")
        video = meta.get("video", {})
        if video.get("decoded_frames") != frames:
            raise ValueError(f"Decoded frame metadata disagrees with cache: {path}")
        expected = list(range(1, frames + 1))
        if [r["frame"] for r in cache["frames"]] != expected:
            raise ValueError(f"Frame mapping mismatch: {path}")
        for frame in cache["frames"]:
            if not isinstance(frame.get("detections"), list) or not isinstance(frame.get("gt"), list):
                raise ValueError(f"Frame is missing detections/reference lists: {path}")
            if any("id" in det or "identity" in det for det in frame["detections"]):
                raise ValueError("Detection cache leaks reference identity")
        for other in caches.values():
            other_meta = other["metadata"]
            if other_meta["sequence"] == sequence:
                for input_name in ("video", "gt", "gt_map"):
                    left = meta.get("inputs", {}).get(input_name, {}).get("sha256")
                    right = other_meta.get("inputs", {}).get(input_name, {}).get("sha256")
                    if not left or left != right:
                        raise ValueError(f"Sources disagree on {sequence} {input_name} identity")
                for key in ("frame_mapping", "reference"):
                    if meta.get(key) != other_meta.get(key):
                        raise ValueError(f"Sources disagree on {sequence} {key} metadata")
                if any(a["gt"] != b["gt"] for a, b in zip(cache["frames"], other["frames"])):
                    raise ValueError(f"Sources disagree on {sequence} reference boxes")
        seen[identity] = path
        caches[path.stem] = cache
        paths.append(path)
    expected_pairs = {(sequence, source) for sequence in SEQUENCES for source in sources}
    if set(seen) != expected_pairs:
        raise ValueError(f"Expected {len(expected_pairs)} selected sequence/source caches; "
                         f"missing={sorted(expected_pairs-set(seen))}, extra={sorted(set(seen)-expected_pairs)}")
    # One detector configuration per selected source keeps the two sequences in
    # the declared condition; per-clip initialization/timing records may differ.
    for source in sources:
        entries = [cache["metadata"] for cache in caches.values() if cache["metadata"]["source"] == source]
        if source == "rfdetr_nano":
            configs = [(entry.get("detector", {}).get("settings"),
                        entry.get("detector", {}).get("checkpoint", {}).get("sha256")) for entry in entries]
            if not configs[0][0] or not configs[0][1] or any(config != configs[0] for config in configs[1:]):
                raise ValueError("RF-DETR caches must share recorded settings and checkpoint identity")
    cases = []
    for name, cache in caches.items():
        meta = cache["metadata"]
        for tracker in ("centroid", "sort"):
            cases.append({"case": f"{name}_{tracker}", "sequence": meta["sequence"],
                          "source": meta["source"], "tracker": tracker})
    return paths, caches, cases


def collect(options):
    output = options.output.resolve()
    if (output / "protocol.json").exists():
        raise FileExistsError("A campaign already exists here; choose a fresh directory")
    source_option = getattr(options, "sources", SOURCES)
    sources = (source_option,) if isinstance(source_option, str) else tuple(source_option)
    inputs, caches, cases = load_inputs(options.inputs, options.frames, sources)
    sequences = sorted({case["sequence"] for case in cases})
    protocol = {"name": "Real sequences with " + " and ".join(sources) + " detections",
                "frames_per_clip": options.frames, "clips": len(sequences), "sources": list(sources),
                "sequences": sequences, "pipeline_configurations": len(cases),
                "trackers": ["centroid", "sort"], "solvers": SOLVERS,
                "alias_validation_only": "lap.lapjvr versus laprust.lapjv",
                "rounds": options.rounds, "seed": options.seed, "minimum_replay_block_seconds": options.block_seconds,
                "cpu": options.cpu, "library_threads": 1, "tracker_input_precision": "float32",
                "replay_precisions": ["float32", "float64"], "warmup_frames": options.warmup_frames,
                "timing_scope": "sum of sequential tracker.update calls; Person construction, serialization, evaluation, decode and detector inference excluded",
                "adapter_scope": "contiguous dtype conversion, finite check, solver public API, int32 pair normalization, cardinality check",
                "replay_scope": "same full captured corpus for every solver; conversion to corpus dtype outside timing; adapter work included",
                "capture_solver": "scipy", "capture_precision": "float32",
                "statistics": "median and IQR across repeated whole-sequence/corpus mean times; descriptive, not significance or generalization intervals",
                "stock_tracker_semantics": "Centroid distance gate and SORT IoU post-filter/unique-feasible bypass preserved; no finite-limit objective substitution",
                "evaluation": "IoU>=0.5 motmetrics supplied-box metrics, no official class/ignore policy; outside timing",
                "objective_tolerance_float32": "abs(gap) <= 1e-7*abs(reference) + 1e-4",
                "objective_tolerance_float64": "abs(gap) <= 1e-10*abs(reference) + 1e-9",
                "objective_validation_note": "exact nonzero-gap counts and maximum gaps retained; large gated costs can inflate relative tolerances; tolerance-pass is not exact equality",
                "reinitialization": "fresh tracker after each warmup; SORT global ID counter reset; module hooks restored",
                "parallel_measurements": False}
    write_json(output / "protocol.json", protocol)
    (output / "provenance").mkdir(parents=True, exist_ok=True)
    for path in inputs:
        write_json(output / "provenance" / (path.stem + "_input_identity.json"), file_identity(path))
    env = environment(output)
    write_json(output / "environment.json", env)
    rng = random.Random(options.seed)
    settings = [(case, solver) for case in cases for solver in SOLVERS]
    samples, first, hashes = [], {}, {}
    started_utc = datetime.now(timezone.utc).isoformat()
    for round_index in range(options.rounds):
        order = list(settings)
        rng.shuffle(order)
        for position, (case, solver) in enumerate(order):
            key = case["case"] + "__" + solver
            cache_key = case["case"].removesuffix("_" + case["tracker"])
            with without_gc():
                result = run_tracking(caches[cache_key], solver, case["tracker"],
                                      warmup_frames=options.warmup_frames)
            predictions = result.pop("predictions")
            fingerprint = prediction_hash(predictions)
            if key in hashes and hashes[key] != fingerprint:
                raise AssertionError(f"Non-deterministic trajectories: {key}, round {round_index}")
            hashes[key] = fingerprint
            if key not in first:
                first[key] = output / "predictions" / (key + ".json")
                write_json(first[key], {"predictions": predictions})
            result.update(case=case["case"], sequence=case["sequence"], source=case["source"],
                          round=round_index, position=position, prediction_sha256=fingerprint)
            write_json(output / "timing" / f"round_{round_index:02d}" / (key + ".json"), result)
            samples.append({**case, "solver": solver, "round": round_index, "position": position,
                            "nanoseconds": result["tracker_total_ns"] / options.frames})
            print(f"Tracking round {round_index+1}/{options.rounds} setting {position+1}/{len(order)}: "
                  f"{key} {result['tracker_ms_per_frame']:.4f} ms/frame", flush=True)
        write_json(output / "tracking_samples.json", samples)
        write_json(output / "progress.json", {"phase": "tracking", "completed_rounds": round_index + 1})
    profiles, captures = {}, {}
    for case in cases:
        cache_key = case["case"].removesuffix("_" + case["tracker"])
        for solver in SOLVERS:
            key = case["case"] + "__" + solver
            mode = "capture" if solver == "scipy" else "profile"
            with without_gc():
                result = run_tracking(caches[cache_key], solver, case["tracker"], mode=mode,
                                      warmup_frames=options.warmup_frames,
                                      capture_output=output / "matrices" / (case["case"] + ".npz") if mode == "capture" else None)
            predictions = result.pop("predictions")
            if prediction_hash(predictions) != hashes[key]:
                raise AssertionError(f"Instrumentation changes trajectories: {key}")
            profiles[key] = result
            write_json(output / "profiles" / (key + ".json"), result)
            if mode == "capture":
                captures[case["case"]] = result
                # Capture time includes compressed file I/O; use a separate profile for the share.
                with without_gc():
                    profile = run_tracking(caches[cache_key], solver, case["tracker"], mode="profile",
                                           warmup_frames=options.warmup_frames)
                if prediction_hash(profile.pop("predictions")) != hashes[key]:
                    raise AssertionError(f"Separate SciPy profile changes trajectories: {key}")
                profiles[key] = profile
                write_json(output / "profiles" / (key + "_without_capture.json"), profile)
            print(f"Profile complete: {key}, {result['solver_calls']} solver calls", flush=True)
    replay, validation = [], {}
    for case in cases:
        rows, checks = replay_case(output, case, captures[case["case"]], options, rng)
        replay.extend(rows)
        validation[case["case"]] = checks
        write_json(output / "replay_summary_partial.json", replay)
    # Evaluation begins only after performance measurements have finished.
    from realbench_evaluate import evaluate
    metric_cache, tracking = {}, []
    for case, solver in settings:
        key = case["case"] + "__" + solver
        cache_key = case["case"].removesuffix("_" + case["tracker"])
        metric_key = (case["case"], hashes[key])
        if metric_key not in metric_cache:
            saved_predictions = json.loads(first[key].read_text(encoding="utf-8"))
            metric_cache[metric_key] = evaluate(caches[cache_key], saved_predictions)
            del saved_predictions
        metrics = metric_cache[metric_key]
        write_json(output / "metrics" / (key + ".json"), metrics)
        selected = [s["nanoseconds"] for s in samples if s["case"] == case["case"] and s["solver"] == solver]
        median, q25, q75 = statistics(selected)
        profile = profiles[key]
        calls = profile["solver_calls"]
        tracking.append({**case, "solver": solver, "precision": "float32", "frames": options.frames,
                         "rounds": options.rounds, "median_tracker_ns_per_frame": median,
                         "q25_tracker_ns_per_frame": q25, "q75_tracker_ns_per_frame": q75,
                         "tracker_fps": 1e9 / median, **{k: metrics[k] for k in
                         ("idf1", "mota", "num_switches", "num_false_positives", "num_misses")},
                         "solver_calls": calls,
                         "mean_solver_ns": profile["solver_total_ns"] / calls if calls else None,
                         "solver_fraction_instrumented": profile["solver_total_ns"] / profile["tracker_total_ns"],
                         "prediction_sha256": hashes[key]})
        print(f"Evaluation complete: {key} IDF1={metrics['idf1']:.4f} MOTA={metrics['mota']:.4f}", flush=True)
    for case in cases:
        baseline = next(r["median_tracker_ns_per_frame"] for r in tracking
                        if r["case"] == case["case"] and r["solver"] == "scipy")
        for row in tracking:
            if row["case"] == case["case"]:
                row["scipy_speedup"] = baseline / row["median_tracker_ns_per_frame"]
    env["final_threadpools"] = threadpool_info()
    write_json(output / "environment.json", env)
    summary = {"protocol": protocol, "environment": env,
               "started_utc": started_utc, "completed_utc": datetime.now(timezone.utc).isoformat(),
               "inputs": [caches[k]["metadata"] for k in sorted(caches)],
               "tracking": tracking, "replay": replay, "validation": validation,
               "reproducibility": {"all_repeated_prediction_hashes_identical": True,
                                   "all_instrumented_predictions_match": True,
                                   "unique_evaluated_trajectories": len(metric_cache)}}
    write_json(output / "summary.json", summary)
    from realbench_report import generate_report
    generate_report(output)
    write_json(output / "progress.json", {"phase": "complete", "completed_rounds": options.rounds})
    identities = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "file_manifest.json":
            identities[str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    write_json(output / "file_manifest.json", identities)
    from realbench_paths import redact
    print(f"Campaign complete: {redact(str(output))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pyppbox-root", type=Path)
    parser.add_argument("--sources", choices=SOURCES, nargs="+", default=list(SOURCES),
                        help="Detection sources to measure; defaults to both. Each selected source requires both sequences.")
    parser.add_argument("--frames", type=int, default=999)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--block-seconds", type=float, default=0.03)
    parser.add_argument("--warmup-frames", type=int, default=32)
    options = parser.parse_args()
    from realbench_paths import configure_pyppbox
    configure_pyppbox(options.pyppbox_root)
    if options.rounds < 1 or options.frames < 1 or options.block_seconds <= 0:
        parser.error("rounds, frames and block-seconds must be positive")
    os.environ["PYPPBOX_DISABLE_FILE_LOG"] = "1"
    process = psutil.Process()
    initial_affinity = process.cpu_affinity()
    try:
        process.cpu_affinity([options.cpu])
        # Import lazy native libraries before applying threadpool controls.
        for name in SOLVERS:
            solve_pairs(name, np.array([[2., 1., 3.], [1., 4., 2.]], dtype=np.float32))
        with threadpool_limits(limits=1):
            collect(options)
    finally:
        process.cpu_affinity(initial_affinity)


if __name__ == "__main__":
    main()
