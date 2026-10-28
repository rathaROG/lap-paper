"""Real pyppbox tracking with interchangeable full-assignment solvers.

The primary ``timed`` mode times tracker.update only. ``profile`` adds assignment
hook timings; ``capture`` also streams its original cost matrices to an NPZ.
Neither instrumented mode supplies primary tracker performance measurements.
No tracker, gating, or SORT fast-path implementation is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter_ns
from types import SimpleNamespace
import zipfile

import numpy as np


SOLVERS = ("lapjv", "lapjvx", "lapjvxa", "lapjvh", "laprust", "lapcpp",
           "scipy", "lapjvs", "lapjvc", "lapjvr")
PRECISIONS = ("float32", "float64")


@lru_cache(maxsize=None)
def _backend(name):
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver {name!r}; choose one of {SOLVERS}")
    if name == "scipy":
        return importlib.import_module("scipy.optimize").linear_sum_assignment
    module = importlib.import_module(name if name in ("laprust", "lapcpp") else "lap")
    return getattr(module, "lapjv" if name in ("laprust", "lapcpp") else name)


def solve_pairs(name, cost, dtype="float32"):
    """Return contiguous int32 [row, column] full-assignment pairs.

    Costs must be finite. Rectangular inputs assign min(n,m) real pairs, with
    infinite cost_limit wherever supported. float64 replay preserves the source
    values; lapjvs uses prefer_float32=False for that explicitly requested mode.
    No backend failure is caught or substituted with a different solver.
    """
    if dtype not in PRECISIONS:
        raise ValueError(f"Unsupported precision {dtype!r}")
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver {name!r}")
    cm = np.ascontiguousarray(cost, dtype=dtype)
    if cm.ndim != 2:
        raise ValueError("Assignment costs must be two-dimensional")
    n, m = cm.shape
    if not np.isfinite(cm).all():
        raise ValueError("Real-tracking assignment costs must all be finite")
    if not min(n, m):
        return np.empty((0, 2), dtype=np.int32)
    function = _backend(name)
    if name == "lapjv":
        row_mapping, _ = function(cm, extend_cost=n != m, cost_limit=np.inf,
                                  return_cost=False)
        rows = np.flatnonzero(np.asarray(row_mapping) >= 0)
        pairs = np.column_stack((rows, np.asarray(row_mapping)[rows]))
    elif name == "lapjvxa":
        pairs = function(cm, extend_cost=n != m, cost_limit=np.inf, return_cost=False)
    elif name in ("lapjvh", "lapjvr", "laprust", "lapcpp"):
        pairs = function(cm, extend_cost=n != m, cost_limit=np.inf,
                         return_cost=False, precision="native", output="pairs")
    elif name == "lapjvx":
        rows, cols = function(cm, extend_cost=n != m, cost_limit=np.inf,
                              return_cost=False)
        pairs = np.column_stack((rows, cols))
    elif name == "lapjvs":
        rows, cols = function(cm, extend_cost=n != m, return_cost=False,
                              jvx_like=True, prefer_float32=dtype == "float32")
        pairs = np.column_stack((rows, cols))
    elif name == "lapjvc":
        rows, cols = function(cm, return_cost=False)
        pairs = np.column_stack((rows, cols))
    else:
        rows, cols = function(cm)
        pairs = np.column_stack((rows, cols))
    pairs = np.ascontiguousarray(pairs, dtype=np.int32).reshape(-1, 2)
    if pairs.shape[0] != min(n, m):
        raise RuntimeError(f"{name} returned {len(pairs)} pairs for {n}x{m}; expected {min(n,m)}")
    return pairs


def validate_pairs(pairs, shape):
    """Expensive integrity checks for replay correctness, outside primary timing."""
    pairs = np.asarray(pairs)
    n, m = shape
    if pairs.shape != (min(n, m), 2):
        raise ValueError("Assignment has wrong pair count or dimensions")
    if len(pairs) and (np.any(pairs < 0) or np.any(pairs[:, 0] >= n)
                       or np.any(pairs[:, 1] >= m)):
        raise ValueError("Assignment contains out-of-bounds indices")
    if len(np.unique(pairs[:, 0])) != len(pairs) or len(np.unique(pairs[:, 1])) != len(pairs):
        raise ValueError("Assignment reuses a row or column")


def _local_pyppbox():
    from realbench_paths import pyppbox_module
    return pyppbox_module()


@contextmanager
def _assignment_hook(tracker_name, hook):
    _local_pyppbox()
    if tracker_name == "centroid":
        module = importlib.import_module("pyppbox.modules.trackers.centroid")
        attr = "_linear_assignment"
    elif tracker_name == "sort":
        module = importlib.import_module("pyppbox.modules.trackers.sort.origin.sort")
        attr = "linear_assignment"
    else:
        raise ValueError("tracker_name must be 'centroid' or 'sort'")
    previous = getattr(module, attr)
    setattr(module, attr, hook)
    try:
        yield
    finally:
        setattr(module, attr, previous)


def _make_tracker(tracker_name, centroid_max_spread, sort_max_age, sort_min_hits, sort_iou_threshold):
    if tracker_name == "centroid":
        from pyppbox.modules.trackers.centroid import MyCentroid
        return MyCentroid(SimpleNamespace(max_spread=centroid_max_spread))
    from pyppbox.modules.trackers.sort import MySORT
    from pyppbox.modules.trackers.sort.origin.sort import KalmanBoxTracker
    KalmanBoxTracker.count = 0
    return MySORT(SimpleNamespace(max_age=sort_max_age, min_hits=sort_min_hits,
                                 iou_threshold=sort_iou_threshold))


def _people(detections):
    from pyppbox.utils.persontools import Person
    people = []
    for index, det in enumerate(detections):
        xyxy = np.asarray(det["box"], dtype=np.int32)
        if xyxy.shape != (4,) or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            raise ValueError(f"Invalid detection box {det['box']!r}")
        xywh = xyxy.copy()
        xywh[2:] -= xywh[:2]
        people.append(Person(index, index, box_xyxy=xyxy, box_xywh=xywh,
                             repspoint=tuple(det["repspoint"]), det_conf=float(det["score"]),
                             faceid="Unknown", deepid="Unknown"))
    return people


class _Profile:
    def __init__(self, solver, precision, archive=None):
        self.solver = solver
        self.precision = precision
        self.archive = archive
        self.frame = None
        self.calls = []

    def __call__(self, costs):
        started = perf_counter_ns()
        pairs = solve_pairs(self.solver, costs, self.precision)
        elapsed = perf_counter_ns() - started
        if self.archive is not None:
            validate_pairs(pairs, costs.shape)
        number = len(self.calls)
        record = {"call_index": number, "frame": self.frame,
                  "shape": list(costs.shape), "source_dtype": str(costs.dtype),
                  "solver_input_dtype": self.precision, "solver_ns": elapsed}
        if self.archive is not None:
            key = f"cost_{number:06d}"
            record["key"] = key
            # Streaming saves the native application matrix, before adapter casting.
            with self.archive.open(key + ".npy", "w", force_zip64=True) as member:
                np.save(member, np.ascontiguousarray(costs), allow_pickle=False)
        self.calls.append(record)
        return pairs


def run_tracking(cache, solver, tracker_name, precision="float32", mode="timed",
                 warmup_frames=32, capture_output=None, centroid_max_spread=64,
                 sort_max_age=1, sort_min_hits=3, sort_iou_threshold=0.3):
    """Run one sequential case on an already-loaded cache; restore hooks on exit.

    Every frame gets fresh Persons without GT identities. Warmup is an unmeasured
    prefix followed by a newly constructed tracker and reset SORT ID counter.
    ``timed`` injects a plain adapter with no hook timing or matrix capture.
    Only tracker.update is measured; decoding, Persons, prediction snapshots, and
    evaluator calls are outside that boundary. Call sequentially, never concurrently
    in one process, because pyppbox assignment hooks are module globals.
    """
    if mode not in ("timed", "profile", "capture"):
        raise ValueError("mode must be timed, profile, or capture")
    if precision not in PRECISIONS or warmup_frames < 0:
        raise ValueError("Invalid precision or negative warmup frame count")
    if mode == "capture" and capture_output is None:
        raise ValueError("capture mode requires capture_output")
    _backend(solver)  # Resolve imports before warmup and timing.
    # SORT's prefix can consist entirely of fast paths. Initialize lazy native/JIT
    # code outside measurement even when the warmup tracker makes no LAP calls.
    solve_pairs(solver, np.array([[0.2, 0.7, 0.8], [0.9, 0.1, 0.5]], dtype=precision), precision)
    frames = cache["frames"]
    frame_ids = [int(frame["frame"]) for frame in frames]
    if any(number < 1 for number in frame_ids) or frame_ids != sorted(set(frame_ids)):
        raise ValueError("Cache frame IDs must be positive, unique, and increasing")
    def make_tracker():
        return _make_tracker(tracker_name, centroid_max_spread, sort_max_age,
                             sort_min_hits, sort_iou_threshold)
    plain = lambda costs: solve_pairs(solver, costs, precision)
    with _assignment_hook(tracker_name, plain):
        warm_tracker = make_tracker()
        try:
            for frame in frames[:warmup_frames]:
                warm_tracker.update(_people(frame["detections"]))
        finally:
            warm_tracker.close()
    archive = None
    if mode == "capture":
        path = Path(capture_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        archive = zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED,
                                  compresslevel=1, allowZip64=True)
    profile = _Profile(solver, precision, archive) if mode != "timed" else None
    hook = profile if profile is not None else plain
    predictions, timings = [], []
    tracker = None
    try:
        with _assignment_hook(tracker_name, hook):
            tracker = make_tracker()
            for frame in frames:
                people = _people(frame["detections"])
                if profile is not None:
                    profile.frame = int(frame["frame"])
                    calls_before = len(profile.calls)
                started = perf_counter_ns()
                emitted = tracker.update(people)
                elapsed = perf_counter_ns() - started
                ids = [int(person.cid) for person in emitted]
                if len(set(ids)) != len(ids):
                    raise RuntimeError(f"Duplicate tracker CID in frame {frame['frame']}")
                tracks = [{"id": int(person.cid), "box": np.asarray(person.box_xyxy).astype(int).tolist()}
                          for person in emitted]
                tracks.sort(key=lambda track: (track["id"], track["box"]))
                predictions.append({"frame": int(frame["frame"]), "tracks": tracks})
                measurement = {"frame": int(frame["frame"]), "tracker_ns": elapsed,
                               "detections": len(people), "tracks": len(tracks)}
                if profile is not None:
                    current_calls = profile.calls[calls_before:]
                    measurement["solver_calls"] = len(current_calls)
                    measurement["solver_ns"] = sum(call["solver_ns"] for call in current_calls)
                timings.append(measurement)
    finally:
        if tracker is not None:
            tracker.close()
        if archive is not None:
            archive.close()
    total = sum(item["tracker_ns"] for item in timings)
    calls = profile.calls if profile is not None else None
    result = {
        "schema_version": 1, "solver": solver, "tracker": tracker_name,
        "precision": precision, "mode": mode,
        "primary_timing": mode == "timed", "frames": len(frames),
        "warmup_frames": min(warmup_frames, len(frames)),
        "adapter_warmup_calls": 1,
        "tracker_total_ns": total,
        "tracker_ms_per_frame": total / len(frames) / 1e6 if frames else None,
        "tracker_fps": len(frames) * 1e9 / total if total else None,
        "solver_calls": len(calls) if calls is not None else None,
        "solver_total_ns": sum(call["solver_ns"] for call in calls) if calls is not None else None,
        "timing_scope": "tracker.update only; excludes Persons, prediction copies, decode, inference and evaluation",
        "solver_timing_scope": "adapter conversion, finite check, native solve, pair normalization and cardinality check; excludes capture and integrity checks",
        "tracker_cost_dtype": "float32",
        "parameters": {"centroid_max_spread": centroid_max_spread, "sort_max_age": sort_max_age,
                       "sort_min_hits": sort_min_hits, "sort_iou_threshold": sort_iou_threshold},
        "per_frame": timings, "predictions": predictions,
    }
    if calls is not None:
        result["assignment_calls"] = calls
    if mode == "capture":
        result["capture_output"] = str(Path(capture_output).resolve())
        result["matrix_metadata"] = calls
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--solver", required=True, choices=SOLVERS)
    parser.add_argument("--tracker", required=True, choices=("centroid", "sort"))
    parser.add_argument("--mode", choices=("timed", "profile", "capture"), default="timed")
    parser.add_argument("--precision", choices=PRECISIONS, default="float32")
    parser.add_argument("--capture-output", type=Path)
    parser.add_argument("--warmup-frames", type=int, default=32)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--pyppbox-root", type=Path)
    parser.add_argument("--centroid-max-spread", type=float, default=64)
    parser.add_argument("--sort-max-age", type=int, default=1)
    parser.add_argument("--sort-min-hits", type=int, default=3)
    parser.add_argument("--sort-iou-threshold", type=float, default=0.3)
    args = parser.parse_args()
    from realbench_paths import configure_pyppbox
    configure_pyppbox(args.pyppbox_root)
    if args.mode == "capture" and args.capture_output is None:
        args.capture_output = args.output.with_suffix(".npz")
    import psutil
    from threadpoolctl import threadpool_limits, threadpool_info
    process = psutil.Process()
    if args.cpu is not None:
        process.cpu_affinity([args.cpu])
    with args.input.open(encoding="utf-8") as source:
        cache = json.load(source)
    with threadpool_limits(limits=1):
        result = run_tracking(cache, args.solver, args.tracker, args.precision,
                              args.mode, args.warmup_frames, args.capture_output,
                              args.centroid_max_spread, args.sort_max_age,
                              args.sort_min_hits, args.sort_iou_threshold)
        result["threadpools"] = threadpool_info()
    result["cpu_affinity"] = process.cpu_affinity()
    result["input"] = str(args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from realbench_paths import redact
    args.output.write_text(json.dumps(redact(result), indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("solver", "tracker", "mode", "frames", "tracker_total_ns", "solver_calls")}))


if __name__ == "__main__":
    main()
