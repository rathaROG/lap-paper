# Benchmarks

These are standalone measurement programs for every measured result in the paper. Each folder runs independently with the declared packages and bundled inputs.

## What reproduces what

| Folder | Paper results | Contents | Time on the paper's machine |
| --- | --- | --- | --- |
| [`base/`](base/run.py) | Tables 4–6, Figures 3–5; A1, A2, B1–B4 | exhaustive enumeration validation; 12 single-matrix cases × 7 solvers; 3 batch workloads (loops, 1 and 4 workers); memory for 256 × 4096 | ~1 min |
| [`formulations/`](formulations/run.py) | Table 7; A3, B5, B6 | 10 finite-limit cases: JVRUST and JVCPP against square-expanded, dummy-column and capped SciPy | < 1 min |
| [`scaling/`](scaling/run.py) | Figures 6–8 (right panels of A1, B1) | 47-case size grid: square, 4:1 wide, fixed N=64, congested | ~12 min |
| [`extended/`](extended/run.py) | Table 8, Figures 9–15 (A2, B3 inputs) | 132 timing cases (large, extreme, structure and rejection sweeps); 405 fresh-process memory trials; 129 edge checks | ~11 min |
| [`real_world/`](real_world/README.md) | Table 12; C1–C4 | tracker updates on two frozen MOT excerpts; assignment replay of 7,539 captured matrices | replay: minutes |
| [`lsape/`](lsape/README.md) | Tables 9–11 | MSVC build, strict verification, 67-case × 3-session campaign, paired summary | ~20 min total |

The cross-environment results (Tables A4 and B7, Figures A1, B1 and B2) compare runs of these same suites in four environments. To reproduce them, run the suites in each environment and compare their summaries. Figures 1 and 2 are explanatory diagrams.

## Quick start

Use Python 3.14.9 for the main-text environment. From this directory:

```bash
python -m pip install -r requirements-py314.txt
python base/run.py --check-inputs
python formulations/run.py --check-inputs
python scaling/run.py --check-inputs
python extended/run.py --check-inputs
python run_all.py
```

For the Python 3.11.17 replication, install `requirements-py311.txt` instead.

`--check-inputs` regenerates the inputs without solving anything and compares them with the paper's measurement records. That covers all 1,212 timing and memory inputs: 48 single, 3 batch, 40 formulation, 188 scaling, 528 extended timing and 405 extended memory. Each line should report `identical to the paper`.

`run_all.py` runs the four timing suites in sequence, one process each, and logs per-CPU load. With no `--output`, it writes to `benchmark/results/full-YYYYMMDD/` using the local date, regardless of the working directory. It refuses to reuse a non-empty run directory; pass a different `--output` for another run that day. Each suite can also be run directly, for example:

```bash
python scaling/run.py --output runs/scaling --cpu 2
python extended/run.py --output runs/extended          # add --resume after an interruption
python run_all.py --output runs/smoke --smoke          # wiring check only, not paper timings
```

Every suite refuses to write into a non-empty output directory, prints paper-style tables at the end, and saves raw samples, summaries, checks and its environment.

## Requirements

Use the requirements file matching the measured interpreter:

| Interpreter | Requirements | NumPy | SciPy |
| --- | --- | --- | --- |
| Python 3.14.9 | [`requirements-py314.txt`](requirements-py314.txt) | 2.5.4 | 1.18.1 |
| Python 3.11.17 | [`requirements-py311.txt`](requirements-py311.txt) | 2.4.6 | 1.17.1 |

Both files pin LAPX 0.12.0, laprust 1.1.0, lapcpp 1.1.0, psutil 7.2.2 and threadpoolctl 3.7.0, as recorded for the Windows and Linux measurements. Use matching solver wheels through your package index or `--find-links` when required. Each run records the versions and native-module hashes it used.

`requirements.txt` is a compatibility recipe with broader numerical-library ranges; it does not select an exact measured environment.

The timing suites need Windows or Linux: CPU affinity comes from psutil, and peak memory uses the Windows peak working set or Linux `ru_maxrss`. `real_world/` and `lsape/` list their own requirements.

## Measurement protocol (four timing suites)

- **Inputs.** Seeds are fixed: 20261021 for base, formulations and scaling; 20261024 for extended. Each case has four float64 matrices, generated outside timing and shared by every method. Extended transposes are exact transposes, and its threshold sweeps reuse the same values.
- **Validation before timing.** Every method's objective on every timed input must match the reference within rtol/atol 1e-10. The references are ordinary SciPy for required problems, and square-expanded or capped SciPy for finite limits. Exhaustive enumeration checks small binary-rational grids, forbidden entries, signed costs, infeasibility and penalty boundaries.
- **Timing.**
  - Each block is calibrated to about 30 ms, with nine rounds per case.
  - Method order is shuffled each round from a seeded generator, and garbage collection is off inside blocks.
  - Timing uses one numerical-library thread (threadpoolctl) and one logical CPU; batch timing uses four CPUs.
  - Adapter work is timed: square expansion, dummy columns, capping, filtering and int64 pair output. Input generation, validation, calibration and checkpoint writes are not.
- **Summaries.** Each case/method summary is the median and interquartile range of its nine round means. These ranges are descriptive, not confidence intervals.
- **Memory.** Each trial runs in a fresh process, with the input resident before the baseline is taken. The recorded value is the high-water increment (Windows peak working set, Linux high-water RSS), not an allocation count. Windows and Linux values are not directly comparable.

**CPU selection.** By default, single-problem timing uses the first allowed CPU and batches use the first four, as in the paper's Windows runs (CPU 0 and CPUs 0–3). Override this with `--cpu` / `--batch-cpus`, or with `LAP_BENCH_SINGLE_CPU` / `LAP_BENCH_CPUS`, which the paper's Linux runs used to select CPUs 0, 2, 4 and 6. On hybrid processors choose performance cores, and report what you used.

## Solvers

| Label | Call |
| --- | --- |
| JVRUST | `laprust.lapjv` (timed); `lap.lapjvr` is validated as the same algorithm |
| JVHYBRID | `lap.lapjvh` (Rust, with C++ square completion) |
| JVCPP | `lapcpp.lapjv` |
| JVX | `lap.lapjvxa` |
| JVS | `lap.lapjvsa`, float64 (`prefer_float32=False`) |
| JVC | `lap.lapjvc` |
| SciPy | `scipy.optimize.linear_sum_assignment`; for finite limits, an explicit square (n+m) expansion |
| SciPy dummy cols / capped | SciPy on C with r appended dummy columns of cost L, or on `min(C, L)` with filtering by the original costs |

Native calls use `extend_cost=True`, `cost_limit=L`, `precision="float64"` and `output="pairs"`. Finite-limit objectives use the square-expanded total F = Σ selected + (L/2)(n + m − 2K).

## How the folders stay standalone

- **Self-contained suites.** Each timing suite is one `run.py` with no imports from the others. The solver adapters, objectives and timing loop sit in a marked block copied into all four, and [`check_consistency.py`](check_consistency.py) fails if those copies drift.
- **Input checks.** `--check-inputs` verifies that the fixed seeds and generators reproduce the input hashes recorded for the paper. Each suite records its package versions, binary identities, CPU placement and measurement settings.
- **Real-world replay.** `real_world/` bundles the eight captured assignment corpora (9 MiB, checked against the published archive manifest), so replay needs no archive. Tracker updates still need the pinned pyppbox checkout described there.
- **LSAPE build.** `lsape/` locates MSVC by itself and builds for the running Python.
- **Paper results.** [`results.zip`](results.zip) contains the saved measurements and summaries reported in the paper; see [`RESULTS.md`](RESULTS.md). `python verify_results.py` checks its file hashes, sample coverage, summary arithmetic and paper table values. New runs write separate observations under `results/full-YYYYMMDD/` by default.
- **Portable records.** Saved records use role placeholders for local paths and identify files by version and SHA-256. Input checksums are listed in `real_world/provenance/data_manifest.json`.
