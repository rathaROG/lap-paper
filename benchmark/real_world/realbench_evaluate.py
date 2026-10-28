"""Evaluate supplied GT boxes and tracking output outside benchmark timing.

The converted dataset does not preserve the complete MOTChallenge class/ignore
policy. These are supplied-box CLEAR MOT and identity metrics, not official
MOTChallenge leaderboard scores. Boxes use continuous xyxy coordinates, and a
pair is eligible at IoU >= 0.5. Predicted IDs need not equal ground-truth IDs.

Metric definitions and accumulation come from motmetrics 1.4.0. Its SciPy
assignment backend is selected independently of the tracker being measured.
NumPy computes IoU here to avoid the removed np.asfarray call in that release's
distance helper; NumPy itself is never patched.
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np


IOU_THRESHOLD = 0.5
METRICS = (
    "num_frames", "num_unique_objects", "num_objects", "num_predictions",
    "num_matches", "num_detections", "num_false_positives", "num_misses",
    "num_switches", "num_fragmentations", "idtp", "idfp", "idfn",
    "idf1", "mota", "motp", "precision", "recall",
)
COUNTS = frozenset(METRICS[:13])
SOURCE_URL = "https://github.com/cheind/py-motmetrics/tree/v1.4.0/motmetrics"


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _frame_rows(payload: Any, keys: tuple[str, ...], context: str) -> list[dict]:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, list):
        raise ValueError(f"{context} must contain a list of frames")
    if not payload:
        raise ValueError(f"{context} must contain at least one frame")
    return payload


def _index_frames(rows: list[dict], context: str) -> dict[int, dict]:
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{context} frame rows must be objects")
        frame = _integer(row.get("frame"), f"{context} frame")
        if frame in indexed:
            raise ValueError(f"duplicate {context} frame {frame}")
        indexed[frame] = row
    return indexed


def _objects(row: dict, key: str, context: str) -> tuple[list[int], np.ndarray]:
    objects = row.get(key)
    if not isinstance(objects, list):
        raise ValueError(f"{context} must contain a {key} list")
    ids = []
    boxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            raise ValueError(f"{context} objects must be mappings")
        identity = _integer(obj.get("id"), f"{context} object id")
        if identity in ids:
            raise ValueError(f"duplicate id {identity} in {context}")
        try:
            box = np.asarray(obj["box"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context} boxes must have four finite xyxy values") from exc
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError(f"{context} boxes must have four finite xyxy values")
        if np.any(box[2:] < box[:2]):
            raise ValueError(f"{context} boxes must have nonnegative extents")
        ids.append(identity)
        boxes.append(box)
    return ids, np.asarray(boxes, dtype=np.float64).reshape(-1, 4)


def iou_distances(gt_boxes: np.ndarray, predicted_boxes: np.ndarray) -> np.ndarray:
    """Return 1-IoU, with NaN for forbidden pairs and exact empty dimensions."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    predicted_boxes = np.asarray(predicted_boxes, dtype=np.float64).reshape(-1, 4)
    lower = np.maximum(gt_boxes[:, None, :2], predicted_boxes[None, :, :2])
    upper = np.minimum(gt_boxes[:, None, 2:], predicted_boxes[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    gt_area = (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1)
    pred_area = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).prod(axis=1)
    union = gt_area[:, None] + pred_area[None, :] - intersection
    iou = np.zeros_like(intersection)
    np.divide(intersection, union, out=iou, where=union > 0)
    return np.where(iou >= IOU_THRESHOLD, 1.0 - iou, np.nan)


def evaluate(cache: dict | list, prediction_rows: dict | list) -> dict:
    """Compute sequence metrics; require exactly one prediction row per frame.

    Cache frames carry ``gt: [{id, box: [x1,y1,x2,y2]}]``. Predictions may
    be the runner's ``{"predictions": [...]}`` wrapper, ``{"frames": [...]}``,
    or the raw list, with each row carrying ``frame`` and ``tracks``. Empty
    frames must be explicit so missing predictions cannot inflate a score.
    """
    import motmetrics as mm

    gt_frames = _index_frames(_frame_rows(cache, ("frames",), "ground truth"), "ground truth")
    pred_frames = _index_frames(
        _frame_rows(prediction_rows, ("predictions", "frames"), "predictions"), "predictions"
    )
    if gt_frames.keys() != pred_frames.keys():
        missing = sorted(gt_frames.keys() - pred_frames.keys())
        extra = sorted(pred_frames.keys() - gt_frames.keys())
        raise ValueError(f"prediction frame coverage differs: missing={missing[:10]}, extra={extra[:10]}")

    # Remapping avoids float64 event-table loss for large integer IDs while
    # retaining identity continuity across frames and independent ID namespaces.
    gt_id_map: dict[int, int] = {}
    pred_id_map: dict[int, int] = {}
    accumulator = mm.MOTAccumulator(auto_id=True)
    with mm.lap.set_default_solver("scipy"):
        for frame in sorted(gt_frames):
            gt_ids, gt_boxes = _objects(gt_frames[frame], "gt", f"ground truth frame {frame}")
            pred_ids, pred_boxes = _objects(pred_frames[frame], "tracks", f"prediction frame {frame}")
            mapped_gt = [gt_id_map.setdefault(i, len(gt_id_map)) for i in gt_ids]
            mapped_pred = [pred_id_map.setdefault(i, len(pred_id_map)) for i in pred_ids]
            accumulator.update(mapped_gt, mapped_pred, iou_distances(gt_boxes, pred_boxes))
        values = mm.metrics.create().compute(accumulator, metrics=list(METRICS), return_dataframe=False)

    result = {}
    for name, value in values.items():
        scalar = float(value)
        result[name] = (int(scalar) if name in COUNTS else scalar) if math.isfinite(scalar) else None
    result.update({
        "evaluation": "supplied-box evaluation (not official MOTChallenge ignore/class filtering)",
        "iou_threshold": IOU_THRESHOLD,
        "coordinate_convention": "continuous xyxy; no inclusive-pixel +1",
        "metric_library": f"motmetrics {version('motmetrics')}",
        "evaluation_assignment_solver": "scipy",
        "metric_source": SOURCE_URL,
        "ratio_units": "fraction; MOTA may be negative; undefined metrics are null",
        "motp_units": "mean 1-IoU distance over matched detections (lower is better)",
        "frame_start": min(gt_frames),
        "frame_end": max(gt_frames),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prepared dataset cache JSON")
    parser.add_argument("--predictions", type=Path, required=True, help="Runner predictions JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = json.loads(args.input.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    result = evaluate(cache, predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from realbench_paths import redact
    args.output.write_text(json.dumps(redact(result), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(redact(result), allow_nan=False))


if __name__ == "__main__":
    main()
