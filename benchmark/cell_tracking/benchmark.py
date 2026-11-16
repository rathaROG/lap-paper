"""Measure three dense optional-assignment configurations in ByoTrack.

Run with --ctc pointing to the eight training-set directories containing
02_ERR_SEG and --hydra pointing to the example video. Inputs stay local.
"""

import argparse
import datetime as dt
import gc
import hashlib
import itertools
import json
import os
import platform
import random
import re
import struct
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from time import perf_counter_ns
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["TQDM_DISABLE"] = "1"

import byotrack
import byotrack.dataset.ctc
import cv2
import numpy as np
import psutil
import pylapy
import torch
from byotrack.implementation.detector.wavelet import WaveletDetector
from byotrack.implementation.linker.frame_by_frame.base import AssociationMethod
from byotrack.implementation.linker.frame_by_frame.kalman_linker import KalmanLinker, KalmanLinkerParameters
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

CTC = (
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-MSC",
    "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "PhC-C2DH-U373", "PhC-C2DL-PSC",
)
ARMS = ("scipy_stock", "scipy_capped", "lapx_rust")
SEED = 20261113


def array_hash(digest, array):
    digest.update(f"{array.dtype.str}:{array.shape}:".encode("ascii"))
    digest.update(array.tobytes())


def track_record(tracks):
    byotrack.Track.check_tracks(tracks, warn=False)
    digest = hashlib.sha256()
    for track in sorted(tracks, key=lambda item: item.identifier):
        digest.update(struct.pack("<4q", track.identifier, track.start, track.parent_id, track.merge_id))
        points = track.points.numpy().copy()
        missing = np.isnan(points)
        points[missing | (points == 0)] = 0
        array_hash(digest, missing)
        array_hash(digest, points)
        array_hash(digest, track.detection_ids.numpy())
    return {"count": len(tracks), "sha256": digest.hexdigest()}


def solve_capped(dist, alternative_cost):
    rows, cols = linear_sum_assignment(np.minimum(dist, alternative_cost))
    keep = dist[rows, cols] < alternative_cost
    return np.stack((rows[keep], cols[keep]), axis=1).astype(np.uint16)


@contextmanager
def backend(arm):
    solver = pylapy.LapSolver
    implementation = "laprust" if arm == "lapx_rust" else "scipy"
    with patch.object(solver, "implementations", {implementation: solver.implementations[implementation]}):
        if arm != "scipy_capped":
            yield
            return

        stock_init = solver.__init__

        def compact_init(instance):
            stock_init(instance)
            instance.cost_extension = "none"

        with patch.object(solver, "__init__", compact_init), patch.object(solver, "_scipy", staticmethod(solve_capped)):
            yield


def profile(linker, detections):
    solve = AssociationMethod.solve
    observations = []
    cost_digest = hashlib.sha256()
    scores = []

    def capture(method, cost, eta):
        assert method == AssociationMethod.OPT_ALT and np.isfinite(eta)
        start = perf_counter_ns()
        links = solve(method, cost, eta)
        elapsed = perf_counter_ns() - start
        matrix = cost.numpy()
        array_hash(cost_digest, matrix)
        cost_digest.update(struct.pack("<d", eta))
        n, m = matrix.shape
        rows, cols = links.numpy().T
        assert len(set(rows)) == len(set(cols)) == len(rows)
        selected = matrix[rows, cols].astype(np.float64)
        assert np.all(selected <= eta)
        scores.append(float(np.sum(selected - eta, dtype=np.float64)))
        observations.append((n, m, elapsed))
        return links

    with patch.object(AssociationMethod, "solve", capture):
        tracks = linker.run(None, detections)

    nonempty = [(n, m) for n, m, _ in observations if n and m]
    return {
        "calls": len(observations),
        "nonempty": len(nonempty),
        "max_shape": list(max(nonempty, key=lambda shape: shape[0] * shape[1])),
        "association_ns_per_frame": sum(elapsed for _, _, elapsed in observations) / len(detections),
        "cost_sha256": cost_digest.hexdigest(),
        "tracks": track_record(tracks),
    }, scores


def benchmark(detections):
    parameters = KalmanLinkerParameters(association_method="OPT_ALT")
    parameters.estimate(detections)
    assert parameters.split_factor == parameters.merge_factor == 0
    linkers = {arm: KalmanLinker(parameters) for arm in ARMS}
    for arm, linker in linkers.items():
        with backend(arm):
            linker.run(None, detections)

    orders = list(itertools.permutations(ARMS))
    random.Random(SEED).shuffle(orders)
    rounds = []
    for order in orders:
        timings = {}
        for arm in order:
            with backend(arm):
                gc.disable()
                try:
                    start = perf_counter_ns()
                    tracks = linkers[arm].run(None, detections)
                    elapsed = perf_counter_ns() - start
                finally:
                    gc.enable()
            timings[arm] = {"ns_per_frame": elapsed / len(detections), "tracks": track_record(tracks)}
        rounds.append(timings)

    profiles = {}
    scores = {}
    for arm, linker in linkers.items():
        with backend(arm):
            profiles[arm], scores[arm] = profile(linker, detections)

    stock = profiles["scipy_stock"]
    validation = {}
    for arm, current in profiles.items():
        matches = current["cost_sha256"] == stock["cost_sha256"]
        gap = max(abs(a - b) for a, b in zip(scores[arm], scores["scipy_stock"], strict=True)) if matches else None
        validation[arm] = {"same_costs_as_stock": matches, "max_abs_objective_gap": gap}
        assert all(round_[arm]["tracks"] == current["tracks"] for round_ in rounds), arm

    return {
        "frames": len(detections), "mean_detections": mean(map(len, detections)),
        "rounds": rounds, "profiles": profiles, "validation": validation,
    }


def ctc_input(root, name):
    files = sorted((root / name / "02_ERR_SEG").glob("*.tif"))[-100:]
    digest = hashlib.sha256()
    frames = []
    for file in files:
        digest.update(f"{file.name}:{hashlib.sha256(file.read_bytes()).hexdigest()}\n".encode("ascii"))
        frames.append(int(re.search(r"\d+$", file.stem)[0]))
    if not frames or frames != list(range(frames[0], frames[-1] + 1)):
        raise ValueError(f"Missing or nonconsecutive CTC masks for {name}")
    video = byotrack.Video(root / name / "02_ERR_SEG")[-100:]
    detections = byotrack.GroundTruthDetector().run(video)
    return detections, {"start": frames[0], "stop": frames[-1] + 1, "sha256": digest.hexdigest()}


def hydra_input(path):
    video = byotrack.Video(path)[130:230]
    video.add_preprocessor(byotrack.video.ChannelProjection("mean"))
    video = video.normalize(q_min=0.01, q_max=0.999, smooth_clip=1.0)
    detector = WaveletDetector(scale=1, k=2.5, min_area=5, batch_size=20, device=torch.device("cpu"))
    return detector.run(video), {"start": 130, "stop": 230, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctc", type=Path, required=True)
    parser.add_argument("--hydra", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=(*CTC, "Hydra"), default=(*CTC, "Hydra"))
    args = parser.parse_args()
    psutil.Process().cpu_affinity([0])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)

    report = {
        "measured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(), "cpu": 0, "seed": SEED,
        "python": platform.python_version(),
        "versions": {name: version(name) for name in (
            "byotrack", "pylapy", "lapx", "scipy", "numpy", "torch", "numba", "llvmlite", "torch-kf",
            "torch-tps", "tifffile", "imagecodecs", "opencv-python", "scikit-image", "threadpoolctl", "psutil", "tqdm",
        )},
        "datasets": {},
    }
    with threadpool_limits(limits=1):
        for name in args.datasets:
            print(f"Preparing {name}", flush=True)
            detections, identity = hydra_input(args.hydra) if name == "Hydra" else ctc_input(args.ctc, name)
            print(f"Timing {name} ({len(detections)} frames)", flush=True)
            report["datasets"][name] = {"input": identity, **benchmark(detections)}
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
