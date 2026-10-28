"""Freeze video/GT and RF-DETR inputs for the real-world assignment benchmark.

This module never passes reference identities into tracker detections. Run with
``python -B realbench_data.py --output runs/example --checkpoint ABSOLUTE_PATH``.
The local pyppbox checkout supplies both detection adapters. No checkpoint is
downloaded; an existing absolute checkpoint is required for RF-DETR.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
SEQUENCES = ("mot16_04", "mot20_03")
SOURCES = ("gt", "rfdetr_nano")
RF_SETTINGS = dict(dt_name="RF_DETR", model_size="nano", conf=0.2, imgsz=768,
                   person_class_id=1, max_det=300, repspoint_calibration=0.25,
                   device=0, optimize=True)
MIN_WIDTH = 15
SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_pyppbox() -> None:
    from realbench_paths import pyppbox_module
    pyppbox_module()


def _source_identity() -> dict[str, Any]:
    import pyppbox
    package = Path(pyppbox.__file__).resolve().parent
    source_root = package.parent
    result: dict[str, Any] = {"path": str(package), "version": pyppbox.__version__}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
        result["git_status"] = subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError):
        result["git_commit"] = None
    result["files"] = {
        name: file_identity(package / name)
        for name in ("utils/gttools.py", "utils/persontools.py",
                     "modules/detectors/rfdetr/__init__.py", "config/myconfig.py")}
    return result


def read_reference_gt(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Validate the supplied converted annotations without changing coordinates."""
    result: dict[int, list[dict[str, Any]]] = {}
    previous = 0
    seen: set[tuple[int, int]] = set()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number}: expected five columns")
            frame, identity = int(fields[0]), int(fields[2])
            point = ast.literal_eval(fields[1])
            xywh = [int(x) for x in fields[3].strip("[]").split()]
            box = [int(x) for x in fields[4].strip("[]").split()]
            if (frame < 1 or frame < previous or len(point) != 2 or
                    len(xywh) != 4 or len(box) != 4 or
                    xywh[2] <= 0 or xywh[3] <= 0 or
                    box != [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]] or
                    (frame, identity) in seen):
                raise ValueError(f"{path}:{line_number}: invalid/duplicate annotation")
            if not all(isinstance(value, int) for value in point):
                raise ValueError(f"{path}:{line_number}: representative point must be integer")
            previous = frame
            seen.add((frame, identity))
            result.setdefault(frame, []).append(
                {"id": identity, "box": box, "repspoint": list(point)})
    if not result or min(result) != 1:
        raise ValueError(f"GT must begin at frame 1: {path}")
    return result


def _sort_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Identity and source annotation order do not influence assignment row order.
    return sorted(detections, key=lambda item:
                  (tuple(item["box"]), tuple(item["repspoint"]), -item["score"]))


@contextmanager
def _no_checkpoint_downloads():
    """Make RF-DETR's downloader a local-file check for this construction only."""
    import rfdetr.assets.model_weights as weights
    import rfdetr.detr as detr
    import rfdetr.models.weights as model_weights
    modules = (weights, detr, model_weights)
    previous = [module.download_pretrain_weights for module in modules]

    def require_local(path, *args, **kwargs):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Benchmark forbids checkpoint downloads: {path}")

    for module in modules:
        module.download_pretrain_weights = require_local
    try:
        yield
    finally:
        for module, original in zip(modules, previous):
            module.download_pretrain_weights = original


def _rf_settings(rf_imgsz: int, rf_conf: float) -> dict[str, Any]:
    if isinstance(rf_imgsz, bool) or not isinstance(rf_imgsz, int) or rf_imgsz <= 0 or rf_imgsz % 32:
        raise ValueError("rf_imgsz must be a positive integer multiple of 32")
    if isinstance(rf_conf, bool) or not isinstance(rf_conf, (float, int)) or not math.isfinite(rf_conf) or not 0 <= rf_conf <= 1:
        raise ValueError("rf_conf must be a finite number in [0, 1]")
    return dict(RF_SETTINGS, imgsz=rf_imgsz, conf=float(rf_conf))


def _rf_runtime_resolution(detector, requested: int) -> dict[str, Any]:
    """Check the loaded runtime and optimized state, beyond configuration alone."""
    model = detector.model
    result = {
        "requested_imgsz": requested,
        "effective_config_resolution": getattr(model.model_config, "resolution", None),
        "effective_model_resolution": getattr(model.model, "resolution", None),
        "effective_optimized_resolution": getattr(model, "_optimized_resolution", None),
        "optimized_for_inference": bool(getattr(model, "_is_optimized_for_inference", False)),
    }
    effective = [result[name] for name in (
        "effective_config_resolution", "effective_model_resolution", "effective_optimized_resolution")]
    if (any(value != requested for value in effective) or not result["optimized_for_inference"] or
            getattr(model.model, "inference_model", None) is None):
        raise RuntimeError(f"RF-DETR requested/runtime resolution mismatch: {result}")
    return result


@contextmanager
def _observe_rf_input_shapes(detector, requested: int):
    """Observe the actual optimized-network tensor during excluded warmups only."""
    observed: list[list[int]] = []

    def observe(module, inputs):
        shape = [int(value) for value in getattr(inputs[0], "shape", ())] if inputs else []
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or shape[2:] != [requested, requested]:
            raise RuntimeError(f"RF-DETR expected a 1x3x{requested}x{requested} input tensor, observed {shape}")
        observed.append(shape)

    handle = detector.model.model.inference_model.register_forward_pre_hook(observe)
    try:
        yield observed
        if not observed:
            raise RuntimeError("RF-DETR optimized network was not called during warmup")
    finally:
        handle.remove()


def _checkpoint_native_resolution(checkpoint: Path) -> int:
    from rfdetr.utilities.io import _safe_torch_load
    contents = _safe_torch_load(checkpoint)
    args = contents.get("args", {})
    resolution = args.get("resolution") if isinstance(args, dict) else getattr(args, "resolution", None)
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError(f"Checkpoint does not record a native resolution: {checkpoint}")
    return resolution


def _open_rf(checkpoint: Path, rf_imgsz: int = 768, rf_conf: float = .2):
    import torch
    from pyppbox.config.myconfig import DCFGRFDETR
    from pyppbox.modules.detectors.rfdetr import MyRFDETR
    if not torch.cuda.is_available():
        raise RuntimeError("RF-DETR benchmark requires CUDA GPU 0")
    config = DCFGRFDETR()
    config.set(dict(_rf_settings(rf_imgsz, rf_conf), model_file=str(checkpoint)))
    started = time.perf_counter_ns()
    with _no_checkpoint_downloads():
        detector = MyRFDETR(config)
    torch.cuda.synchronize(0)
    initialization_ns = time.perf_counter_ns() - started
    try:
        _rf_runtime_resolution(detector, rf_imgsz)
    except BaseException:
        detector.close()
        raise
    return detector, initialization_ns


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Cache already exists; use a fresh output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    from realbench_paths import redact
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(redact(document), stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def build_caches(output: str | Path, frames: int = 999,
                 sources: tuple[str, ...] | list[str] = SOURCES,
                 sequences: tuple[str, ...] | list[str] = SEQUENCES,
                 checkpoint: str | Path | None = None,
                 warmup: int = 3, rf_imgsz: int = 768, rf_conf: float = .2,
                 dataset_root: str | Path | None = None,
                 pyppbox_root: str | Path | None = None) -> list[Path]:
    """Decode each selected video once and cache detections for every source.

    Timings exclude JSON serialization, geometric sorting, and reference loading.
    CUDA timings include a device synchronization before and after inference.
    First-frame warmups are excluded, and the measured pass reopens the video.
    """
    if isinstance(frames, bool) or not isinstance(frames, int) or not 1 <= frames <= 999:
        raise ValueError("frames must be an integer in 1..999")
    if warmup < 1:
        raise ValueError("warmup must be at least one")
    rf_settings = _rf_settings(rf_imgsz, rf_conf)
    if dataset_root is None:
        raise ValueError("dataset_root is required for optional video/detector regeneration")
    dataset_root = Path(dataset_root).expanduser().resolve(strict=True)
    from realbench_paths import configure_pyppbox
    configure_pyppbox(pyppbox_root)
    sources, sequences = tuple(dict.fromkeys(sources)), tuple(dict.fromkeys(sequences))
    if not sources or set(sources) - set(SOURCES):
        raise ValueError(f"sources must be selected from {SOURCES}")
    if not sequences or set(sequences) - set(SEQUENCES):
        raise ValueError(f"sequences must be selected from {SEQUENCES}")
    output = Path(output).resolve()
    for sequence in sequences:
        for source in sources:
            target = output / "inputs" / f"{sequence}_{source}.json"
            if target.exists() or target.with_suffix(".json.tmp").exists():
                raise FileExistsError(f"Use a fresh output directory: {target}")
    checkpoint_path = None
    if "rfdetr_nano" in sources:
        if checkpoint is None or not Path(checkpoint).is_absolute():
            raise ValueError("RF-DETR requires --checkpoint with an existing absolute local path")
        checkpoint_path = Path(checkpoint).resolve(strict=True)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

    os.environ["PYPPBOX_DISABLE_FILE_LOG"] = "1"
    _local_pyppbox()
    import cv2
    import numpy as np
    from pyppbox.utils.gttools import GTInterpreter
    source_identity = _source_identity()
    versions = {name: _version(name) for name in
                ("numpy", "opencv-python", "opencv-python-headless", "torch", "rfdetr", "supervision")}
    checkpoint_identity = file_identity(checkpoint_path) if checkpoint_path else None
    detector = None
    initialization_ns = None
    checkpoint_native_resolution = None
    model_native_default_resolution = None
    if checkpoint_path is not None:
        import torch
        from rfdetr.assets.model_weights import ModelWeights
        from rfdetr.config import RFDETRNanoConfig
        asset = ModelWeights.from_filename(checkpoint_path.name)
        checkpoint_identity["published_md5"] = asset.md5_hash if asset else None
        with checkpoint_path.open("rb") as checkpoint_stream:
            actual_md5 = hashlib.file_digest(checkpoint_stream, "md5").hexdigest()
        checkpoint_identity["actual_md5"] = actual_md5
        checkpoint_identity["matches_published_md5"] = (
            actual_md5 == asset.md5_hash if asset and asset.md5_hash else None)
        checkpoint_native_resolution = _checkpoint_native_resolution(checkpoint_path)
        model_native_default_resolution = RFDETRNanoConfig.model_fields["resolution"].default
        detector, initialization_ns = _open_rf(checkpoint_path, rf_imgsz, rf_conf)
    written: list[Path] = []
    try:
        for sequence in sequences:
            video = dataset_root / f"{sequence}_gt_24fps_lossless.mp4"
            gt_path = dataset_root / "gt" / f"{sequence}_gt_pyppbox_nonoise.txt"
            reference = read_reference_gt(gt_path)
            if any(frame not in reference for frame in range(1, frames + 1)):
                raise ValueError(f"GT missing requested frames: {gt_path}")
            gt_detector = None
            if "gt" in sources:
                gt_detector = GTInterpreter()
                gt_detector.setGT(str(gt_path))
                gt_detector.setDetectOnly(detect_only=True)
                gt_detector.current_frame = 1
            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video}")
            try:
                reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                video_backend = cap.getBackendName()
                if reported_frames < frames or not math.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Invalid video metadata: {video}")
                warmup_ns: list[int] = []
                resolution_metadata = None
                if detector is not None:
                    ok, first = cap.read()
                    if not ok:
                        raise RuntimeError(f"Cannot decode warmup frame: {video}")
                    with _observe_rf_input_shapes(detector, rf_imgsz) as observed_shapes:
                        for _ in range(warmup):
                            torch.cuda.synchronize(0)
                            started = time.perf_counter_ns()
                            detector.detect(first, visual=False, classes=1, min_width_filter=MIN_WIDTH)
                            torch.cuda.synchronize(0)
                            warmup_ns.append(time.perf_counter_ns() - started)
                    resolution_metadata = dict(
                        _rf_runtime_resolution(detector, rf_imgsz),
                        observed_input_tensor_shapes=observed_shapes,
                        observation_scope="forward pre-hook during excluded warmups; removed before measured frames",
                        checkpoint_native_resolution=checkpoint_native_resolution,
                        model_native_default_resolution=model_native_default_resolution)
                    cap.release()
                    cap = cv2.VideoCapture(str(video))
                    if not cap.isOpened():
                        raise RuntimeError(f"Cannot reopen video after warmup: {video}")
                records: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
                decode_ns: list[int] = []
                detector_ns: dict[str, list[int]] = {source: [] for source in sources}
                for frame_number in range(1, frames + 1):
                    started = time.perf_counter_ns()
                    ok, image = cap.read()
                    elapsed = time.perf_counter_ns() - started
                    if not ok or image.shape[:2] != (height, width):
                        raise RuntimeError(f"Decode failed/dimensions changed at {video}, frame {frame_number}")
                    decode_ns.append(elapsed)
                    truth = [{"id": row["id"], "box": row["box"]}
                             for row in reference[frame_number]]
                    for source in sources:
                        if source == "gt":
                            assert gt_detector is not None
                            started = time.perf_counter_ns()
                            people, _ = gt_detector.getPeople(image, visual=False)
                            detector_ns[source].append(time.perf_counter_ns() - started)
                            detections = [{"box": [int(x) for x in person.box_xyxy],
                                           "repspoint": [int(x) for x in person.repspoint], "score": 1.0}
                                          for person in people]
                            if len(detections) != len(truth):
                                raise ValueError(f"GT cursor mismatch at frame {frame_number}")
                        else:
                            torch.cuda.synchronize(0)
                            started = time.perf_counter_ns()
                            _, _, boxes, points, scores = detector.detect(
                                image, visual=False, classes=1, min_width_filter=MIN_WIDTH)
                            torch.cuda.synchronize(0)
                            detector_ns[source].append(time.perf_counter_ns() - started)
                            detections = [{"box": [int(x) for x in box],
                                           "repspoint": [int(x) for x in point], "score": float(score)}
                                          for box, point, score in zip(boxes, points, scores)]
                        records[source].append({"frame": frame_number,
                                                "detections": _sort_detections(detections), "gt": truth})
                    if frame_number == 1 or frame_number % 100 == 0 or frame_number == frames:
                        print(f"{sequence}: cached {frame_number}/{frames} frames", flush=True)
                fully_decoded = frames == reported_frames
                if fully_decoded and cap.read()[0]:
                    raise ValueError(f"Video contains more frames than reported: {video}")
            finally:
                cap.release()
            inputs = {"video": file_identity(video), "gt": file_identity(gt_path),
                      "gt_map": file_identity(dataset_root / "gt" / "gt_map.txt")}
            for source in sources:
                metadata: dict[str, Any] = {
                    "sequence": sequence, "source": source, "frames": frames,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "inputs": inputs, "schema_version": SCHEMA_VERSION,
                    "frame_mapping": "decoded zero-based index k maps to GT frame k+1",
                    "video": {"reported_frames": reported_frames, "decoded_frames": frames,
                              "fully_decoded": fully_decoded, "fps": fps, "width": width,
                              "height": height, "backend": video_backend},
                    "reference": {"frames": len(reference), "first_frame": min(reference),
                                  "last_frame": max(reference),
                                  "policy": "supplied five-column converted GT; class/visibility/ignore metadata unavailable"},
                    "ordering": "lexicographic geometric box, representative point, descending confidence; no GT identity",
                    "pyppbox": source_identity,
                    "environment": {"python": sys.version, "executable": sys.executable,
                                    "platform": platform.platform(), "packages": versions,
                                    "numpy_version": np.__version__, "opencv_version": cv2.__version__},
                    "timings_ns": {"decode": decode_ns, "detector": detector_ns[source],
                                   "excludes": ["model initialization", "warmup", "reference loading",
                                                "geometric sorting", "serialization"]},
                }
                if source == "rfdetr_nano":
                    metadata["detector"] = {
                        "settings": dict(rf_settings, min_width_filter=MIN_WIDTH),
                        "resolution": resolution_metadata,
                        "checkpoint": checkpoint_identity, "package_version": versions["rfdetr"],
                        "initialization_ns": initialization_ns, "warmup_ns": warmup_ns,
                        "initialization_scope": "once per build_caches call, shared across sequences",
                        "warmup_frame": 1, "synchronous_cuda": True,
                        "gpu": torch.cuda.get_device_name(0),
                        "torch_num_threads": torch.get_num_threads(),
                        "adapter": "local pyppbox MyRFDETR.detect; own clipping/class/width filters"}
                else:
                    metadata["detector"] = {"adapter": "local pyppbox GTInterpreter, detect_only=True",
                                            "reference_identities_in_detections": False,
                                            "box_policy": "preserve supplied coordinates including out-of-frame boxes"}
                destination = output / "inputs" / f"{sequence}_{source}.json"
                _atomic_json(destination, {"metadata": metadata, "frames": records[source]})
                written.append(destination)
                from realbench_paths import redact
                print(f"Saved {redact(str(destination))}", flush=True)
    finally:
        if detector is not None:
            detector.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--pyppbox-root", type=Path)
    parser.add_argument("--frames", type=int, default=999)
    parser.add_argument("--sources", choices=(*SOURCES, "both"), nargs="+", default=["both"])
    parser.add_argument("--sequences", choices=SEQUENCES, nargs="+", default=list(SEQUENCES))
    parser.add_argument("--checkpoint", type=Path,
                        help="Existing absolute RF-DETR Nano .pth; required for rfdetr_nano")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rf-imgsz", type=int, default=768,
                        help="Nano input resolution set before optimization; positive multiple of 32 (default: 768)")
    parser.add_argument("--rf-conf", type=float, default=.2,
                        help="RF-DETR detection confidence threshold (default: 0.2)")
    args = parser.parse_args()
    sources = SOURCES if "both" in args.sources else tuple(args.sources)
    build_caches(args.output, args.frames, sources, args.sequences, args.checkpoint,
                 args.warmup, rf_imgsz=args.rf_imgsz, rf_conf=args.rf_conf,
                 dataset_root=args.dataset_root, pyppbox_root=args.pyppbox_root)


if __name__ == "__main__":
    main()
