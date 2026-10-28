"""Analytic fixtures for the independent tracking metric evaluator."""

from copy import deepcopy
import json

import motmetrics as mm
import numpy as np
import pytest

from realbench_evaluate import evaluate, iou_distances


def obj(identity, box=(0, 0, 10, 10)):
    return {"id": identity, "box": list(box)}


def sequence(gt, predictions):
    cache = {"frames": [{"frame": i, "gt": objects} for i, objects in enumerate(gt)]}
    pred = [{"frame": i, "tracks": objects} for i, objects in enumerate(predictions)]
    return cache, pred


def test_perfect_tracking_with_arbitrary_permuted_large_ids():
    gt = [[obj(2**60), obj(2**60 + 1, (20, 0, 30, 10))]] * 3
    pred = [[obj(901, (20, 0, 30, 10)), obj(-8)]] * 3
    cache, predictions = sequence(gt, pred)
    result = evaluate(cache, {"predictions": predictions})
    assert result["idf1"] == result["mota"] == result["precision"] == result["recall"] == 1
    assert result["num_switches"] == result["num_false_positives"] == result["num_misses"] == 0
    assert result["num_frames"] == 3
    assert result["num_objects"] == result["num_predictions"] == 6
    assert result["motp"] == 0


def test_one_miss_and_one_false_positive_have_known_counts_and_ratios():
    cache, predictions = sequence(
        [[obj(1)], [obj(1)], [obj(1)]],
        [[obj(91)], [], [obj(91), obj(30, (40, 40, 50, 50))]],
    )
    result = evaluate(cache, predictions)
    assert result["num_misses"] == result["num_false_positives"] == 1
    assert result["num_switches"] == 0
    assert result["num_detections"] == result["idtp"] == 2
    assert result["idfp"] == result["idfn"] == 1
    assert result["idf1"] == pytest.approx(2 / 3)
    assert result["precision"] == result["recall"] == pytest.approx(2 / 3)
    assert result["mota"] == pytest.approx(1 / 3)


def test_identity_switch_affects_mota_and_global_identity_score():
    cache, predictions = sequence([[obj(1)], [obj(1)]], [[obj(10)], [obj(20)]])
    result = evaluate(cache, predictions)
    assert result["num_switches"] == 1
    assert result["num_detections"] == 2
    assert result["num_misses"] == result["num_false_positives"] == 0
    assert result["mota"] == result["idf1"] == 0.5
    assert result["precision"] == result["recall"] == 1


def test_empty_frames_count_and_false_positives_are_not_dropped():
    cache, predictions = sequence([[], [obj(1)], [], []], [[], [obj(40)], [obj(40)], []])
    result = evaluate(cache, predictions)
    assert result["num_frames"] == 4
    assert result["num_false_positives"] == 1
    assert result["num_misses"] == 0
    assert result["mota"] == 0
    assert result["idf1"] == pytest.approx(2 / 3)


def test_all_empty_sequence_has_valid_json_and_undefined_ratios():
    cache, predictions = sequence([[], []], [[], []])
    result = evaluate(cache, predictions)
    assert result["num_frames"] == 2
    assert result["num_objects"] == result["num_predictions"] == 0
    assert result["mota"] is None
    assert result["idf1"] is None
    assert result["precision"] is None
    assert result["recall"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("has_gt", [False, True])
def test_an_entirely_unmatched_sequence_counts_every_object(has_gt):
    cache, predictions = sequence(
        [[obj(1)], [obj(1)]] if has_gt else [[], []],
        [[], []] if has_gt else [[obj(70)], [obj(70)]],
    )
    result = evaluate(cache, predictions)
    assert result["num_misses"] == (2 if has_gt else 0)
    assert result["num_false_positives"] == (0 if has_gt else 2)
    assert result["idf1"] == 0
    assert result["mota"] == (0 if has_gt else None)
    json.dumps(result, allow_nan=False)


def test_iou_cutoff_includes_exact_half_and_excludes_lower_overlap():
    gt = np.array([[0, 0, 3, 2]], dtype=float)
    pred = np.array([[1, 0, 4, 2], [1.01, 0, 4.01, 2]], dtype=float)
    distances = iou_distances(gt, pred)
    assert distances[0, 0] == 0.5
    assert np.isnan(distances[0, 1])
    assert iou_distances(np.empty((0, 4)), pred).shape == (0, 2)
    assert iou_distances(gt, np.empty((0, 4))).shape == (1, 0)
    cache, predictions = sequence([[obj(1, gt[0])]], [[obj(30, pred[1])]])
    result = evaluate(cache, predictions)
    assert result["num_misses"] == result["num_false_positives"] == 1
    assert result["mota"] == -1
    assert result["idf1"] == 0


@pytest.mark.parametrize("side", ["gt", "tracks"])
def test_duplicate_ids_in_a_frame_are_rejected(side):
    cache, predictions = sequence([[obj(1)]], [[obj(10)]])
    target = cache["frames"][0] if side == "gt" else predictions[0]
    target[side].append(deepcopy(target[side][0]))
    with pytest.raises(ValueError, match="duplicate id"):
        evaluate(cache, predictions)


def test_prediction_frame_coverage_and_duplicate_frames_are_rejected():
    cache, predictions = sequence([[obj(1)], []], [[obj(10)], []])
    with pytest.raises(ValueError, match="coverage differs"):
        evaluate(cache, predictions[:1])
    with pytest.raises(ValueError, match="duplicate predictions frame"):
        evaluate(cache, predictions + [predictions[0]])


@pytest.mark.parametrize("box", [[0, 0, float("nan"), 1], [2, 2, 1, 1], [0, 1, 2]])
def test_malformed_boxes_are_rejected(box):
    cache, predictions = sequence([[obj(1)]], [[obj(10, box)]])
    with pytest.raises(ValueError, match="boxes"):
        evaluate(cache, predictions)


def test_evaluator_forces_scipy_and_restores_previous_solver(monkeypatch):
    cache, predictions = sequence([[obj(1)]], [[obj(10)]])
    scipy_solver = mm.lap.solver_map["scipy"]
    calls = []

    def observed(cost):
        calls.append(cost.shape)
        return scipy_solver(cost)

    def forbidden(cost):
        raise AssertionError("the tracker's solver leaked into evaluation")

    monkeypatch.setitem(mm.lap.solver_map, "scipy", observed)
    with mm.lap.set_default_solver(forbidden):
        assert evaluate(cache, predictions)["idf1"] == 1
        assert mm.lap.default_solver is forbidden
    assert calls
