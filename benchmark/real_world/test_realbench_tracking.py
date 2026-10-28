"""Correctness tests for the real-data adapters and actual pyppbox hooks."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from realbench_tracking import SOLVERS, _assignment_hook, _local_pyppbox, run_tracking, solve_pairs, validate_pairs


class AssignmentTests(unittest.TestCase):
    def test_rectangular_square_negative_and_tied_objectives(self):
        rng = np.random.default_rng(731)
        matrices = [rng.normal(size=shape) for shape in ((2, 5), (5, 2), (4, 4), (1, 8), (8, 1))]
        matrices += [np.zeros((3, 5)), np.array([[1e6, 0, 1e6], [1e6, 1e6, 1e6]])]
        for dtype in ("float32", "float64"):
            for matrix in matrices:
                source = matrix.astype(dtype)
                expected = source[linear_sum_assignment(source)].astype(np.float64).sum()
                for name in SOLVERS:
                    with self.subTest(dtype=dtype, shape=source.shape, solver=name):
                        pairs = solve_pairs(name, source, dtype)
                        validate_pairs(pairs, source.shape)
                        self.assertEqual(pairs.dtype, np.int32)
                        actual = source[pairs[:, 0], pairs[:, 1]].astype(np.float64).sum()
                        self.assertAlmostEqual(actual, expected, places=5)

    def test_empty_and_invalid_inputs(self):
        for name in SOLVERS:
            for shape in ((0, 0), (0, 5), (4, 0)):
                self.assertEqual(solve_pairs(name, np.empty(shape)).shape, (0, 2))
        for matrix in (np.array([1, 2]), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.assertRaises(ValueError):
                solve_pairs("scipy", matrix)
        with self.assertRaises(ValueError):
            validate_pairs(np.array([[0, 0], [0, 1]]), (2, 2))

    def test_lapjvr_alias_matches_laprust_objective(self):
        matrix = np.array([[3.5, 1., 4.], [0., 8., 2.]], dtype=np.float64)
        for precision in ("float32", "float64"):
            one = solve_pairs("lapjvr", matrix, precision)
            two = solve_pairs("laprust", matrix, precision)
            self.assertEqual(matrix[one[:, 0], one[:, 1]].sum(), matrix[two[:, 0], two[:, 1]].sum())


class TrackerTests(unittest.TestCase):
    def setUp(self):
        _local_pyppbox()

    @staticmethod
    def cache():
        return {"metadata": {}, "frames": [
            {"frame": frame, "detections": [
                {"box": [x + frame, y, x + frame + 20, y + 40],
                 "repspoint": [x + frame + 10, y + 10], "score": 0.5}
                for x, y in ((0, 0), (100, 100))],
             "gt": [{"id": 8765, "box": [1, 0, 21, 40]}]}
            for frame in (1, 2, 3)]}

    def test_actual_trackers_fresh_runs_and_hook_restoration(self):
        import pyppbox.modules.trackers.centroid as centroid
        import pyppbox.modules.trackers.sort.origin.sort as sort
        hooks = (centroid._linear_assignment, sort.linear_assignment)
        for tracker in ("centroid", "sort"):
            reference = run_tracking(self.cache(), "scipy", tracker, warmup_frames=2)
            for name in SOLVERS:
                with self.subTest(tracker=tracker, solver=name):
                    result = run_tracking(self.cache(), name, tracker, warmup_frames=2)
                    self.assertTrue(result["primary_timing"])
                    self.assertEqual(result["frames"], 3)
                    self.assertEqual(result["predictions"], reference["predictions"])
                    self.assertEqual([t["id"] for t in result["predictions"][0]["tracks"]], [0, 1])
            self.assertIs(centroid._linear_assignment, hooks[0])
            self.assertIs(sort.linear_assignment, hooks[1])

    def test_gating_and_sort_fast_path_are_preserved(self):
        from types import SimpleNamespace
        from pyppbox.modules.trackers.centroid import MyCentroid
        from pyppbox.modules.trackers.sort.origin.sort import associate_detections_to_trackers
        for name in SOLVERS:
            with self.subTest(solver=name):
                with _assignment_hook("centroid", lambda c: solve_pairs(name, c)):
                    matches, unmatched_cur, unmatched_prev = MyCentroid(SimpleNamespace(max_spread=3))._lap_match(
                        np.array([[0, 0], [2, 0]], np.float32), np.array([[0, 0], [100, 0]], np.float32), 3)
                    self.assertEqual(matches.tolist(), [[0, 0]])
                    self.assertEqual(unmatched_cur.tolist(), [1])
                    self.assertEqual(unmatched_prev.tolist(), [1])
        calls = []
        def counted(costs):
            calls.append(costs.copy())
            return solve_pairs("scipy", costs)
        with _assignment_hook("sort", counted):
            boxes = np.array([[0, 0, 10, 20, .1], [100, 100, 110, 120, .9]], np.float32)
            matches, _, _ = associate_detections_to_trackers(boxes, boxes, .3)
            self.assertEqual(len(matches), 2)
            self.assertEqual(len(calls), 0)
            ambiguous = np.array([[0, 0, 10, 20, .1], [1, 0, 11, 20, .9]], np.float32)
            associate_detections_to_trackers(ambiguous, ambiguous, .3)
            self.assertEqual(len(calls), 1)

    def test_capture_roundtrip_and_profile_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.npz"
            result = run_tracking(self.cache(), "scipy", "centroid", mode="capture",
                                  warmup_frames=1, capture_output=path)
            self.assertFalse(result["primary_timing"])
            self.assertEqual(result["solver_calls"], 2)
            with np.load(path, allow_pickle=False) as matrices:
                for call in result["matrix_metadata"]:
                    source = matrices[call["key"]]
                    self.assertEqual(source.dtype, np.float32)
                    self.assertEqual(list(source.shape), call["shape"])
            timed = run_tracking(self.cache(), "scipy", "centroid", warmup_frames=1)
            self.assertIsNone(timed["solver_calls"])
            self.assertEqual(timed["predictions"], result["predictions"])

    def test_hook_restored_on_solver_failure(self):
        import pyppbox.modules.trackers.centroid as centroid
        previous = centroid._linear_assignment
        def fail(_):
            raise RuntimeError("injected solver failure")
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with _assignment_hook("centroid", fail):
                centroid._linear_assignment(np.zeros((2, 2)))
        self.assertIs(centroid._linear_assignment, previous)


if __name__ == "__main__":
    unittest.main()
