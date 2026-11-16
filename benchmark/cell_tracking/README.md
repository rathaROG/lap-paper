# Microscopy: dense optional association

This separate saved session was measured on 13 November 2026 with ByoTrack
2.0.8, Pylapy 1.2.0 and LAPX 0.13.1. It is not part of the four October
assignment/pedestrian environments, which retain LAPX 0.13.0 and
laprust/lapcpp 1.2.0. It is not run by the combined paper launcher.

## Reproduce the saved tables without running benchmarks

From the repository root, using only the Python standard library:

```sh
python -B -m benchmark.cell_tracking audit
python -B -m benchmark.cell_tracking audit --compact-table latex/generated/T_microscopy.tex --profile-table latex/generated/A_microscopy_profiles.tex
python -B -m pytest benchmark/cell_tracking -q
```

The main saved-data generator also includes this study. `manifest.json` binds
the original measurement driver and observations by SHA-256. Both are copied
unchanged from the evidence used for the revised conference manuscript.
`recalculate.py` checks recorded orders, frame coverage, track/cost digests
and objective gaps and regenerates tables; it never imports a solver or tracker.
Table export destinations must resolve outside `benchmark/cell_tracking/`;
all destinations are checked before any output is written.
Tests exercise the saved-record checks, aggregation and safe command entry point.

## Measurement contract

All three arms use dense `OPT_ALT`, shared estimated motion parameters, no
optical flow or splitting/merging, and prepared detections. Stock means the
SciPy backend forced through Pylapy's standard square expansion; it does not
mean ByoTrack's default sparse mode or Pylapy's automatic backend selection.
Capped SciPy keeps the original dimensions, caps costs at the finite rejection
limit and removes returned edges at or above it. Rust uses `lap.lapjvr` through
Pylapy's `laprust` backend. The same adapter handles all three arms.

One warm-up per arm precedes six whole-clip timing rounds, covering all six arm
orders with seed 20261113. Medians of the six clip means are reported in
ms/frame; ratios divide those medians. The boundary is the complete linker,
including track collection and final smoothing, excluding detection preparation
and parameter estimation. The single Windows/Python 3.14.10 session uses
logical CPU 0 on an Intel Core i9-13900KS, one-thread numerical libraries,
the Balanced power scheme and disabled garbage collection during timing.

Separate instrumented runs time the association call, including the Pylapy
adapter, solver and pair conversion. Cost construction and profiling checks
are outside that interval. The lower table panel sums each unrounded profile
mean multiplied by the excerpt's frame count, across all nine excerpts and
816 frames. It is one profile run per arm/excerpt, not six timing rounds, a
mean of speedups, or a pure solver-kernel measurement. Separate profiling and
whole-linker runs do not give an exact time decomposition.

All timed and profiled track digests agree within each arm. Across arms, saved
track/cost digests and recorded objectives agree over 816 frames (807 nonempty
association calls per arm). The audit checks these saved records; images,
detections, matrices and tracks are not redistributed, so it cannot recompute
their digests or independently re-solve the objective checks. Package versions
and source hashes are recorded; the session has no native-wheel binary hash.

## New measurements, only when explicitly requested

Use a separate environment matching the recorded Python and package versions
in `observations.json`; the PyTorch build used for this session was CPU-only.
`requirements-tracking.txt` records version pins, not wheel hashes. Supply the
datasets locally and choose a fresh output file outside the frozen study:

```sh
python -B -m benchmark.cell_tracking run --allow-full-campaign --ctc CTC --hydra tdt_contrxn-1.avi --output benchmark/runs/microscopy-new/observations.json
```

The wrapper preserves the checksum-bound original `benchmark.py` and refuses
to overwrite frozen evidence. Its `run` command collects all nine excerpts.
Compare input hashes before interpreting new observations; do not relabel the
saved session with later installed package versions.

## Data and attribution

The eight Cell Tracking Challenge excerpts use sequence 02 and the final up
to 100 `02_ERR_SEG` masks. Hydra uses zero-based video frames 130 through 229,
mean-channel projection, 0.01/0.999 quantile normalization with smooth clipping
at 1.0, and CPU wavelet detection (scale 1, k 2.5, minimum area 5, batch size 20).

The excerpt selection and preparation recipe follow Raphael Reme's
[Cell Tracking with the new laprust](https://github.com/raphaelreme/cell_tracking_with_laprust)
at revision `1dae22e68c4dd1d6c3915a7a416dbfca39284b31`. See
[ATTRIBUTION.md](ATTRIBUTION.md), the retained MIT notice and `provenance.json`.
This study adds capped SciPy, balanced round orders, CPU/thread controls and
stronger saved validation; none of the earlier benchmark's timings is reused.

The previously reviewed [CTC conditions](https://celltrackingchallenge.net/datasets/)
require explicit permission for public scientific use outside the challenge.
That permission remains a release follow-up for this local revision, including
an arXiv update or public evidence. No input images or masks are included.
