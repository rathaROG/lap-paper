# Paper benchmarks

Run from the repository root using Python 3.11.18 or 3.14.10.

## Install

```sh
python -m pip install -r benchmark/campaign/requirements.txt -r benchmark/application/requirements-tracking.txt -r benchmark/application/requirements-report.txt
python -m pip install --no-deps pyppbox==4.0.0
```

On a fresh machine without compatible cached builds, first-time preparation
requires Git, Rust (`cargo`/`rustc`), a C++17 compiler and matching Python development
headers. On Windows, use MSVC Build Tools with the Windows SDK. On Linux, a conda
environment with either supported Python version is also suitable; activate it
before running the commands. Released solver packages still install from PyPI.

## Run the paper benchmark

```sh
python -m benchmark run --allow-full-campaign
```

For the light paper protocol:

```sh
python -m benchmark run --allow-full-campaign-light
```

The full command retains its existing coverage and five assignment/nine
application rounds. Light keeps five inputs and every input size, with three
assignment and application rounds. Its geometric size sweep uses the
high-survival/low-motion (q=0.9, motion=0.25δ) and low-survival/high-motion
(q=0.6, motion=1δ) settings. The complete 256×320 and 1024×1280 panels retain
all four settings, including both objectives and the 256×320 sensitivity grid.
In geometry only, JVX, JVS and square-expanded SciPy stop above 1024 rows;
this cutoff includes unpadded required-square JVX/JVS calls. Native solvers
and the main required/capped SciPy comparisons continue to the largest sizes.
Other workload families, layouts, policy arms, five memory trials, validation
and the Windows LSAPE protocol retain their coverage.

Light defaults to `paper-run-light/`; full still defaults to `paper-run/`.
Records identify the profile and its frozen manifest. Summaries use the saved
protocol and reject combining full and light sessions. Completed light sessions took 66.23 and 65.39 minutes on Linux
(Python 3.11.18 and 3.14.10), and 85.71 and 85.32 minutes on Windows,
including the recorded preparation state. These are observed launcher times,
not predictions for a fresh machine.

The launcher selects CPUs, finds compatible cached builds, prepares any missing
paper-specific variants, validates inputs, runs the suites sequentially, and
writes estimates, summaries and reports. Released solvers use the PyPI wheels.
Windows runs all three suites; Linux runs assignment and applications because
LSAPE uses the frozen Windows/MSVC protocol.

## Bounded smoke check

```sh
python -m benchmark smoke
```

Smoke runs validation and short checks without starting full measurements.
Existing prepared environments are detected automatically; no build-index,
CPU or output paths need to be supplied.

Results and detailed stage logs go under
`benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/paper-<command>/`.
The launcher shows live progress, saves the same output in stage logs, and
retains `status.json` if a stage fails.
An optional `--output` selects a different fresh directory.

Optional CPU selection, using verified CPU IDs for your machine:

```sh
python -m benchmark run --allow-full-campaign --cpus 0 2 4 6 --single-cpu 0
```

The single CPU must be one of the four selected CPU IDs. Overrides retain
the placement checks; they cannot substitute for missing hybrid core-class data.
They apply to assignment, diagnostics and applications; LSAPE keeps its fixed CPU 0.

Detailed protocols: [assignment](campaign/README.md),
[applications](application/README.md), [LSAPE](lsape/README.md).
