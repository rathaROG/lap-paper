"""Portable entry point for frozen-input tracking, assignment replay and reports.

Verification uses only Python's standard library. Cached replay does not import
pyppbox, OpenCV, RF-DETR or Torch. Published measurements remain in the immutable
collection; every new measurement/report needs a fresh --output directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from .realbench_paths import (PACKAGE, collection_path, configure_pyppbox,
                             data_manifest, materialize_inputs, redact, sha256)

SEQUENCES = ("mot16_04", "mot20_03")
SOURCES = ("gt", "rfdetr_nano")
TRACKERS = ("centroid", "sort")
SOLVERS = ("lapjv", "lapjvx", "lapjvxa", "lapjvh", "laprust", "lapcpp",
           "scipy", "lapjvs")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_inputs():
    """Verify shipped bytes/schema/reference consistency without external assets."""
    manifest = data_manifest()
    cases, references, input_hashes = [], {}, {}
    for name, identity in manifest["files"].items():
        path = PACKAGE / "data" / name
        if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
            raise ValueError(f"Packaged data checksum mismatch: {name}")
        if not name.startswith("inputs/"):
            continue
        raw = gzip.decompress(path.read_bytes())
        if len(raw) != identity["uncompressed_bytes"] or hashlib.sha256(raw).hexdigest() != identity["uncompressed_sha256"]:
            raise ValueError(f"Uncompressed input checksum mismatch: {name}")
        cache = json.loads(raw)
        meta, frames = cache["metadata"], cache["frames"]
        sequence, source = meta["sequence"], meta["source"]
        if sequence not in SEQUENCES or source not in SOURCES or name != f"inputs/{sequence}_{source}.json.gz":
            raise ValueError(f"Unexpected input metadata: {name}")
        if len(frames) != 999 or meta["frames"] != 999 or [f["frame"] for f in frames] != list(range(1, 1000)):
            raise ValueError(f"Expected the complete frozen 999-frame excerpt: {name}")
        for frame in frames:
            detections = frame["detections"]
            if any(set(d) != {"box", "repspoint", "score"} for d in detections):
                raise ValueError(f"Invalid detection fields/reference identity leakage: {name}")
            if detections != sorted(detections, key=lambda d: (tuple(d["box"]), tuple(d["repspoint"]), -d["score"])):
                raise ValueError(f"Detection geometric order differs: {name}")
            for detection in detections:
                b, point, score = detection["box"], detection["repspoint"], detection["score"]
                if (len(b) != 4 or not all(type(x) is int for x in b) or b[2] <= b[0] or b[3] <= b[1]
                        or len(point) != 2 or not all(type(x) is int for x in point)
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError(f"Invalid detection geometry/score: {name}")
        ref = [f["gt"] for f in frames]
        if sequence in references and references[sequence] != ref:
            raise ValueError(f"Sources have different reference identities/boxes: {sequence}")
        references[sequence] = ref
        if source == "gt":
            for frame in frames:
                if sorted(tuple(d["box"]) for d in frame["detections"]) != sorted(tuple(g["box"]) for g in frame["gt"]):
                    raise ValueError(f"GT detection/reference box mismatch: {name}")
        else:
            settings, resolution = meta["detector"]["settings"], meta["detector"]["resolution"]
            if settings["imgsz"] != 768 or settings["conf"] != .2 or settings["max_det"] != 300 or settings["min_width_filter"] != 15:
                raise ValueError("Packaged RF condition differs from selected 768/.2 condition")
            if resolution["observed_input_tensor_shapes"] != [[1, 3, 768, 768]] * 3:
                raise ValueError("Missing actual network tensor resolution proof")
        input_hashes[(sequence, source)] = meta["inputs"]
        cases.append({"sequence": sequence, "source": source, "frames": len(frames),
                      "detections": sum(len(f["detections"]) for f in frames),
                      "original_cache_sha256": identity["uncompressed_sha256"]})
    if {(c["sequence"], c["source"]) for c in cases} != {(s, d) for s in SEQUENCES for d in SOURCES}:
        raise ValueError("Four packaged sequence/source caches are required")
    for sequence in SEQUENCES:
        for key in ("video", "gt", "gt_map"):
            if input_hashes[(sequence, "gt")][key]["sha256"] != input_hashes[(sequence, "rfdetr_nano")][key]["sha256"]:
                raise ValueError(f"Source identity differs between caches: {sequence}/{key}")
    return {"passed": True, "cases": cases, "videos_or_models_required": False,
            "note": "Metadata locations are role placeholders such as <dataset-root>; each recorded file keeps its SHA-256."}


def fresh_output(path, mode):
    if path is None:
        from ..paths import default_output
        path = default_output("application-" + mode)
        print(f"Output: {path}")
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_cases(args):
    directory = Path(args.inputs).resolve() if args.inputs else materialize_inputs(Path(args.output) / "inputs")
    caches, cases = {}, []
    for sequence in args.sequences:
        for source in args.sources:
            path = directory / f"{sequence}_{source}.json"
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache["metadata"]["sequence"] != sequence or cache["metadata"]["source"] != source:
                raise ValueError(f"Unexpected cache metadata: {path}")
            if not 1 <= args.frames <= len(cache["frames"]):
                raise ValueError(f"--frames exceeds available cache coverage: {path}")
            if [f["frame"] for f in cache["frames"][:args.frames]] != list(range(1, args.frames + 1)):
                raise ValueError(f"Noncontiguous selected frames: {path}")
            original_count = len(cache["frames"])
            cache["frames"] = cache["frames"][:args.frames]
            if args.frames != original_count:
                meta = dict(cache["metadata"])
                meta["original_cache_frames"], meta["frames"] = original_count, args.frames
                meta["video"] = dict(meta["video"], decoded_frames=args.frames, fully_decoded=False)
                cache["metadata"] = meta
            caches[f"{sequence}_{source}"] = cache
            for tracker in args.trackers:
                case = {"case": f"{sequence}_{source}_{tracker}", "sequence": sequence,
                        "source": source, "tracker": tracker}
                if not args.cases or case["case"] in args.cases:
                    cases.append(case)
    if not cases:
        raise ValueError("No cases selected")
    if args.cases and set(args.cases) != {c["case"] for c in cases}:
        raise ValueError("--cases contains an unavailable or excluded case")
    return caches, cases


def cpu_session(args):
    from contextlib import contextmanager

    @contextmanager
    def session():
        import numpy as np
        import psutil
        from threadpoolctl import threadpool_limits
        from .realbench_tracking import solve_pairs
        process = psutil.Process()
        original = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
        placement = {"expected": [args.cpu] if args.cpu >= 0 else original}
        try:
            if args.cpu >= 0:
                if original is None:
                    raise RuntimeError("CPU affinity is unavailable; use --cpu -1 and record that difference")
                process.cpu_affinity([args.cpu])
                if process.cpu_affinity() != [args.cpu]:
                    raise RuntimeError("Calling-thread affinity differs before measurement")
            for solver in args.solvers:
                solve_pairs(solver, np.array([[2., 1., 3.], [1., 4., 2.]], np.float32))
            with threadpool_limits(limits=1):
                placement["before"] = process.cpu_affinity() if original is not None else None
                yield {"affinity": placement["before"], "placement_checks": placement,
                       "library_threads": 1, "python": sys.version, "executable": sys.executable}
                placement["after"] = process.cpu_affinity() if original is not None else None
                if placement["after"] != placement["expected"]:
                    raise RuntimeError("Calling-thread affinity changed during measurement")
        finally:
            if original is not None:
                process.cpu_affinity(original)
    return session()


def tracker_stage(args, output):
    configure_pyppbox(args.pyppbox_root)
    from .realbench_campaign import prediction_hash, statistics, without_gc
    from .realbench_tracking import run_tracking
    caches, cases = load_cases(args)
    samples, first, fingerprints = [], {}, {}
    rng = random.Random(args.seed)
    with cpu_session(args) as environment:
        for round_index in range(args.rounds):
            settings = [(case, solver) for case in cases for solver in args.solvers]
            rng.shuffle(settings)
            for position, (case, solver) in enumerate(settings):
                name = case["case"] + "__" + solver
                cache = caches[f"{case['sequence']}_{case['source']}"]
                with without_gc():
                    result = run_tracking(cache, solver, case["tracker"], warmup_frames=args.warmup_frames)
                predictions = result.pop("predictions")
                fingerprint = prediction_hash(predictions)
                if name in fingerprints and fingerprints[name] != fingerprint:
                    raise AssertionError(f"Nondeterministic tracker trajectory: {name}")
                fingerprints[name] = fingerprint
                if name not in first:
                    first[name] = predictions
                    write_json(output / "predictions" / (name + ".json"), {"predictions": predictions})
                write_json(output / "timing" / f"round_{round_index:02d}" / (name + ".json"), result)
                samples.append({**case, "solver": solver, "round": round_index, "position": position,
                                "nanoseconds": result["tracker_total_ns"] / args.frames})
        # Evaluate only after all primary measurements finish.
        from .realbench_evaluate import evaluate
        rows = []
        for case in cases:
            cache = caches[f"{case['sequence']}_{case['source']}"]
            for solver in args.solvers:
                name = case["case"] + "__" + solver
                metrics = evaluate(cache, {"predictions": first[name]})
                write_json(output / "metrics" / (name + ".json"), metrics)
                med, q25, q75 = statistics([s["nanoseconds"] for s in samples if s["case"] == case["case"] and s["solver"] == solver])
                rows.append({**case, "solver": solver, "precision": "float32", "frames": args.frames,
                             "rounds": args.rounds, "median_tracker_ns_per_frame": med,
                             "q25_tracker_ns_per_frame": q25, "q75_tracker_ns_per_frame": q75,
                             "tracker_fps": 1e9 / med, "prediction_sha256": fingerprints[name],
                             "solver_calls": None, "mean_solver_ns": None, "solver_fraction_instrumented": None,
                             **{k: metrics[k] for k in ("idf1", "mota", "num_switches", "num_false_positives", "num_misses")}})
        for case in cases:
            baseline = next(x["median_tracker_ns_per_frame"] for x in rows if x["case"] == case["case"] and x["solver"] == "scipy")
            for row in rows:
                if row["case"] == case["case"]:
                    row["scipy_speedup"] = baseline / row["median_tracker_ns_per_frame"]
    protocol = {"stage": "cached tracker updates only", "frames_per_clip": args.frames,
                "rounds": args.rounds, "seed": args.seed, "warmup_frames": args.warmup_frames,
                "solvers": args.solvers, "timing_scope": "unchanged run_tracking tracker.update boundary; Persons/decode/inference/evaluation excluded",
                "profiling": "not performed by tracker-only; use --campaign for separate profile/capture passes",
                "partial_smoke": args.frames != 999 or args.rounds != 9}
    write_json(output / "tracking_samples.json", samples)
    write_json(output / "summary.json", {"protocol": protocol, "environment": environment,
               "inputs": [c["metadata"] for c in caches.values()], "tracking": rows, "replay": []})


def replay_stage(args, output):
    import numpy as np
    from .realbench_campaign import replay_block, statistics
    from .realbench_tracking import solve_pairs, validate_pairs
    collection = collection_path(args.collection)
    # Selecting cases needs only metadata/names, not pyppbox or detector imports.
    cases = [{"case": f"{seq}_{source}_{tracker}", "sequence": seq, "source": source, "tracker": tracker}
             for seq in args.sequences for source in args.sources for tracker in args.trackers]
    if args.cases:
        cases = [c for c in cases if c["case"] in args.cases]
        if {c["case"] for c in cases} != set(args.cases):
            raise ValueError("Unknown/excluded --cases")
    rows, all_checks, corpora = [], {}, []
    rng = random.Random(args.seed)
    with cpu_session(args) as environment:
        for case in cases:
            path = collection / "matrices" / (case["case"] + ".npz")
            with np.load(path, allow_pickle=False) as store:
                keys = sorted(store.files)
                if args.limit_matrices:
                    keys = keys[:args.limit_matrices]
                original = [store[key] for key in keys]
            if not original:
                raise ValueError(f"Empty assignment corpus: {path}")
            corpora.append({"case": case["case"], "file": str(path), "sha256": sha256(path), "selected_matrices": len(original)})
            for precision in args.precisions:
                matrices = [np.ascontiguousarray(a, dtype=precision) for a in original]
                reference = []
                for matrix in matrices:
                    pairs = solve_pairs("scipy", matrix, precision)
                    reference.append(float(matrix[pairs[:, 0], pairs[:, 1]].sum(dtype=np.float64)))
                checks, cycles = {}, {}
                for solver in args.solvers:
                    gaps, mismatch_count = [], 0
                    for matrix, expected in zip(matrices, reference):
                        pairs = solve_pairs(solver, matrix, precision)
                        validate_pairs(pairs, matrix.shape)
                        gap = float(matrix[pairs[:, 0], pairs[:, 1]].sum(dtype=np.float64)) - expected
                        gaps.append(gap)
                        tolerance = 1e-7 * abs(expected) + 1e-4 if precision == "float32" else 1e-10 * abs(expected) + 1e-9
                        mismatch_count += abs(gap) > tolerance
                    checks[solver] = {"matrices": len(matrices), "objective_max_abs_gap": max(map(abs, gaps)),
                                      "objective_nonzero_gap_count": sum(x != 0 for x in gaps),
                                      "objective_mismatch_count": int(mismatch_count)}
                    if mismatch_count:
                        raise AssertionError(f"{precision} replay objective mismatch: {case['case']}/{solver}")
                    probe = replay_block(solver, matrices, precision, 1)
                    cycles[solver] = max(1, min(10000, math.ceil(args.block_seconds * 1e9 / (probe * len(matrices)))))
                samples = []
                for round_index in range(args.rounds):
                    order = list(args.solvers)
                    rng.shuffle(order)
                    for position, solver in enumerate(order):
                        elapsed = replay_block(solver, matrices, precision, cycles[solver])
                        samples.append({"round": round_index, "position": position, "solver": solver,
                                        "cycles": cycles[solver], "matrices": len(matrices), "nanoseconds": elapsed})
                for solver in args.solvers:
                    med, q25, q75 = statistics([s["nanoseconds"] for s in samples if s["solver"] == solver])
                    rows.append({**case, "precision": precision, "solver": solver, "matrices": len(matrices),
                                 "rounds": args.rounds, "median_ns_per_assignment": med,
                                 "q25_ns_per_assignment": q25, "q75_ns_per_assignment": q75,
                                 **{k: checks[solver][k] for k in ("objective_max_abs_gap", "objective_mismatch_count")}})
                all_checks[f"{case['case']}/{precision}"] = checks
                write_json(output / "replay" / f"{case['case']}_{precision}.json", {"samples": samples, "validation": checks})
    for case in cases:
        for precision in args.precisions:
            baseline = next(x["median_ns_per_assignment"] for x in rows if x["case"] == case["case"] and x["precision"] == precision and x["solver"] == "scipy")
            for row in rows:
                if row["case"] == case["case"] and row["precision"] == precision:
                    row["scipy_speedup"] = baseline / row["median_ns_per_assignment"]
    protocol = {"stage": "cached common-matrix replay only", "rounds": args.rounds, "seed": args.seed,
                "minimum_replay_block_seconds": args.block_seconds, "precisions": args.precisions,
                "solvers": args.solvers, "timing_scope": "unchanged replay_block/solve_pairs; conversion/finite check/solve/pair adaptation included; matrix loading and validation excluded",
                "float64_scope": "same captured float32 costs widened outside timing, not recomputed in float64",
                "partial_smoke": args.limit_matrices is not None or args.rounds != 9 or args.block_seconds != .03}
    write_json(output / "summary.json", {"protocol": protocol, "environment": environment,
               "tracking": [], "replay": rows, "validation": all_checks, "corpora": corpora})
    with (output / "replay_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_snapshot(output):
    """Keep executable source identities and copies with each new run."""
    sources = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "tests" in path.parts or path.name.startswith("test_"):
            continue
        relative = path.relative_to(PACKAGE)
        target = output / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[relative.as_posix()] = sha256(path)
    for name in ("data_manifest.json", "core_semantics.json", "pyppbox_identity.json"):
        target = output / "sources" / "provenance" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PACKAGE / "provenance" / name, target)
    return sources


def validate_corpora(output):
    """Check the entire frozen corpus without collecting performance samples."""
    import numpy as np
    from .realbench_tracking import solve_pairs, validate_pairs
    collection = collection_path()
    precisions = ("float32", "float64")
    solvers = SOLVERS + ("lapjvr",)
    results = []
    total = 0
    for path in sorted((collection / "matrices").glob("*.npz")):
        with np.load(path, allow_pickle=False) as store:
            original = [store[key] for key in sorted(store.files)]
        total += len(original)
        for precision in precisions:
            matrices = [np.ascontiguousarray(a, dtype=precision) for a in original]
            identities = [hashlib.sha256(a.tobytes()).digest() for a in matrices]
            references = []
            for matrix, identity in zip(matrices, identities):
                reference_pairs = solve_pairs("scipy", matrix, precision)
                if hashlib.sha256(matrix.tobytes()).digest() != identity:
                    raise AssertionError("Reference solver mutated a frozen input")
                references.append(float(matrix[reference_pairs[:, 0], reference_pairs[:, 1]].sum(dtype=np.float64)))
            for solver in solvers:
                nonzero, maximum, mismatches = 0, 0., 0
                for matrix, expected, identity in zip(matrices, references, identities):
                    pairs = solve_pairs(solver, matrix, precision)
                    if hashlib.sha256(matrix.tobytes()).digest() != identity:
                        raise AssertionError(f"{solver} mutated a frozen input")
                    validate_pairs(pairs, matrix.shape)
                    value = float(matrix[pairs[:, 0], pairs[:, 1]].sum(dtype=np.float64))
                    gap = abs(value - expected)
                    nonzero += gap != 0
                    maximum = max(maximum, gap)
                    tolerance = 1e-7 * abs(expected) + 1e-4 if precision == "float32" else 1e-10 * abs(expected) + 1e-9
                    mismatches += gap > tolerance
                    if solver == "lapjvr":
                        alias = solve_pairs("laprust", matrix, precision)
                        if hashlib.sha256(matrix.tobytes()).digest() != identity:
                            raise AssertionError("laprust mutated a frozen input during alias validation")
                        alias_value = float(matrix[alias[:, 0], alias[:, 1]].sum(dtype=np.float64))
                        if value != alias_value:
                            raise AssertionError("lapjvr/laprust alias objectives differ")
                results.append({"case": path.stem, "precision": precision, "solver": solver,
                                "matrices": len(matrices), "objective_nonzero_gap_count": int(nonzero),
                                "objective_max_abs_gap": maximum, "objective_mismatch_count": int(mismatches)})
        print(f"Validated {path.stem}: {len(original)} matrices", flush=True)
    expected_corpora = len(SEQUENCES) * len(SOURCES) * len(TRACKERS)
    expected_results = expected_corpora * len(precisions) * len(solvers)
    if total != 7539 or len(results) != expected_results:
        raise AssertionError("Frozen corpus coverage changed")
    report = {"scope": "correctness_validation_no_timing", "matrices": total,
              "checks": sum(row["matrices"] for row in results), "results": results,
              "passed": not any(row["objective_mismatch_count"] for row in results)}
    write_json(output / "validation.json", report)
    if not report["passed"]:
        raise AssertionError("Frozen corpus objective validation failed")


def smoke_stage(args, output):
    """Exercise every frozen pipeline/backend with explicitly short records."""
    from copy import copy
    from .realbench_campaign import prediction_hash
    from .realbench_tracking import run_tracking
    options = copy(args)
    options.frames, options.rounds, options.warmup_frames = 16, 1, 2
    options.block_seconds, options.limit_matrices = .001, 8
    options.sources, options.sequences = list(SOURCES), list(SEQUENCES)
    options.trackers, options.solvers = list(TRACKERS), list(SOLVERS)
    options.precisions, options.cases = ["float32", "float64"], None
    options.inputs, options.collection = None, None
    options.output = output / "tracker-updates"
    options.output.mkdir()
    tracker_stage(options, options.output)
    tracking = json.loads((options.output / "summary.json").read_text())
    caches, cases = load_cases(options)
    hashes = {(row["case"], row["solver"]): row["prediction_sha256"] for row in tracking["tracking"]}
    profile_count, capture_count = 0, 0
    with cpu_session(options):
        for case in cases:
            cache = caches[f"{case['sequence']}_{case['source']}"]
            for solver in SOLVERS:
                result = run_tracking(cache, solver, case["tracker"], mode="profile", warmup_frames=2)
                if prediction_hash(result.pop("predictions")) != hashes[case["case"], solver]:
                    raise AssertionError("Profiling changed the tracker trajectory")
                write_json(output / "profiles" / f"{case['case']}__{solver}.json", result)
                profile_count += 1
            captured = run_tracking(cache, "scipy", case["tracker"], mode="capture", warmup_frames=2,
                                   capture_output=output / "captured-matrices" / (case["case"] + ".npz"))
            if prediction_hash(captured.pop("predictions")) != hashes[case["case"], "scipy"]:
                raise AssertionError("Capture changed the tracker trajectory")
            write_json(output / "profiles" / (case["case"] + "__capture.json"), captured)
            capture_count += 1
    options.output = output / "assignment-replay"
    options.output.mkdir()
    replay_stage(options, options.output)
    replay = json.loads((options.output / "summary.json").read_text())
    from .realbench_report import generate_report
    generate_report(output / "tracker-updates")
    generate_report(output / "assignment-replay")
    summary = {"scope": "smoke_not_publication_evidence", "status": "complete",
               "tracking_conditions": len(tracking["tracking"]), "replay_conditions": len(replay["replay"]),
               "profile_conditions": profile_count, "captured_pipelines": capture_count,
               "frames_per_pipeline": options.frames, "matrices_per_replay_corpus": options.limit_matrices,
               "rounds": options.rounds, "objective_mismatches": sum(r["objective_mismatch_count"] for r in replay["replay"])}
    expected_pipelines = len(SEQUENCES) * len(SOURCES) * len(TRACKERS)
    expected_conditions = expected_pipelines * len(SOLVERS)
    assert summary["tracking_conditions"] == expected_conditions
    assert summary["replay_conditions"] == expected_conditions * len(options.precisions)
    assert summary["profile_conditions"] == expected_conditions
    assert summary["captured_pipelines"] == expected_pipelines
    assert summary["objective_mismatches"] == 0
    write_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("verify-inputs", "validate", "smoke", "tracker-only", "replay-only", "campaign", "report-only", "regenerate-inputs"):
        mode.add_argument("--" + flag, action="store_true")
    parser.add_argument("--allow-full-campaign", action="store_true",
                        help="Required for measurement modes other than the fixed bounded smoke")
    parser.add_argument("--profile", choices=("full", "light"), default="full",
                        help="Paper application protocol; light retains the complete corpora with three rounds")
    parser.add_argument("--output", type=Path,
                        help="Default: benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/application-<mode>")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--pyppbox-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    parser.add_argument("--trackers", nargs="+", choices=TRACKERS, default=list(TRACKERS))
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--precisions", nargs="+", choices=("float32", "float64"), default=["float32", "float64"])
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--frames", type=int, default=999)
    parser.add_argument("--rounds", type=int, help="Timing rounds; default nine, or three for the light campaign")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--cpu", type=int, default=0, help="Logical CPU; -1 leaves affinity unchanged")
    parser.add_argument("--block-seconds", type=float, default=.03)
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument("--limit-matrices", type=int, help="Explicitly partial replay prefix")
    parser.add_argument("--rf-imgsz", type=int, default=768)
    parser.add_argument("--rf-conf", type=float, default=.2)
    args = parser.parse_args(argv)
    if args.profile == "light" and not args.campaign:
        parser.error("The light application profile requires --campaign")
    if args.rounds is None:
        args.rounds = 3 if args.profile == "light" else 9
    if args.profile == "light" and args.rounds != 3:
        parser.error("The light application profile requires three rounds")
    if args.rounds < 1 or args.frames < 1 or args.block_seconds <= 0 or args.warmup_frames < 0 or (args.limit_matrices is not None and args.limit_matrices < 1):
        parser.error("invalid nonpositive rounds/frames/block/limit or negative warmup")
    if args.verify_inputs:
        print(json.dumps(verify_inputs(), indent=2))
        return 0
    measurement = not (args.report_only or args.validate)
    if measurement and not args.smoke and not args.allow_full_campaign:
        parser.error("Full measurements require --allow-full-campaign; use --smoke for bounded validation")
    if measurement and not args.smoke:
        import platform
        if platform.python_version() not in ("3.11.18", "3.14.10"):
            parser.error("Full measurements require Python 3.11.18 or 3.14.10")
    if (args.tracker_only or args.replay_only) and "scipy" not in args.solvers:
        parser.error("include scipy in --solvers for matched baseline ratios")
    if args.report_only and args.collection is None:
        parser.error("--report-only needs --collection pointing at a tracker, replay or campaign output")
    if args.regenerate_inputs and (args.dataset_root is None or args.checkpoint is None):
        parser.error("optional regeneration requires --dataset-root and --checkpoint")
    if args.campaign and (tuple(args.solvers) != SOLVERS or tuple(args.sequences) != SEQUENCES or
                          tuple(args.trackers) != TRACKERS or args.frames != 999 or args.cases is not None or
                          args.precisions != ["float32", "float64"] or args.limit_matrices is not None):
        parser.error("--campaign measures all eight solvers/both sequences/trackers at 999 frames and both complete precision corpora; use the separate stages for subsets")
    mode = next(flag for flag in ("validate", "smoke", "tracker-only", "replay-only", "campaign",
                                  "report-only", "regenerate-inputs") if getattr(args, flag.replace("-", "_")))
    output = fresh_output(args.output, mode)
    args.output = output
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYPPBOX_DISABLE_FILE_LOG"] = "1"
    os.environ["MPLCONFIGDIR"] = str(output / "runtime" / "matplotlib")
    scope = ("smoke_not_publication_evidence" if args.smoke else
             "correctness_validation_no_timing" if args.validate else
             "application_one_session" if measurement else "saved_report")
    record = {"scope": scope, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "source_sha256": source_snapshot(output)}
    write_json(output / "run.json", record)
    started = time.perf_counter()
    try:
        from contextlib import nullcontext
        if measurement or args.validate:
            from benchmark.campaign.resources import HostLock
            lock = HostLock()
        else:
            lock = nullcontext()
        with lock:
            if measurement or args.validate:
                from benchmark.campaign.runner import environment
                write_json(output / "runtime-environment.json", environment())
                verify_inputs()
            if args.validate:
                validate_corpora(output)
            elif args.smoke:
                smoke_stage(args, output)
            elif args.tracker_only:
                tracker_stage(args, output)
            elif args.replay_only:
                replay_stage(args, output)
            elif args.report_only:
                source = collection_path(args.collection)
                for name in ("summary.json", "protocol.json", "environment.json"):
                    if (source / name).is_file():
                        write_json(output / name, json.loads((source / name).read_text(encoding="utf-8")))
                from .realbench_report import generate_report
                generate_report(output)
            elif args.regenerate_inputs:
                from .realbench_data import build_caches
                build_caches(output, args.frames, args.sources, args.sequences,
                             args.checkpoint.resolve(), rf_imgsz=args.rf_imgsz, rf_conf=args.rf_conf,
                             dataset_root=args.dataset_root, pyppbox_root=args.pyppbox_root)
            else:
                configure_pyppbox(args.pyppbox_root)
                from .realbench_campaign import collect
                args.inputs = Path(args.inputs).resolve() if args.inputs else materialize_inputs(output / "inputs")
                with cpu_session(args) as placement:
                    collect(args)
                write_json(output / "placement-checks.json", placement)
            if any(sha256(PACKAGE / name) != digest for name, digest in record["source_sha256"].items()):
                raise RuntimeError("Application source changed during this run")
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        write_json(output / "run.json", record)
        if record["status"] == "complete":
            # collect() also writes an intermediate inventory; refresh after the
            # wrapper's final status and placement records have been written.
            identities = {path.relative_to(output).as_posix():
                          {"bytes": path.stat().st_size, "sha256": sha256(path)}
                          for path in sorted(output.rglob("*"))
                          if path.is_file() and path != output / "file_manifest.json"}
            write_json(output / "file_manifest.json", identities)
    print(f"{scope}: {record['wall_seconds']:.2f} seconds; {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
