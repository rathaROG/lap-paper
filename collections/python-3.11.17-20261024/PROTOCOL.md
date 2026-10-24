# Python 3.11.17 replication protocol

This directory is a separate replication of both completed Python 3.14.9
collections. It contains the original single-call, batch, formulation, and
scaling suites, followed by the larger and special cases, rejection sweeps,
repeated memory measurements, and focused charts. The manuscript and both
Python 3.14.9 collections remain unchanged.

## Environment and source identity

The interpreter is `C:\dev\exc\python\p311\python.exe`, version 3.11.17.
Installed solver versions are lapx 0.11.1, lapcpp 1.0.0, and laprust 1.0.0.
The user-prepared NumPy 2.4.6 and SciPy 1.17.1 are retained. Missing benchmark
support packages were added: psutil 7.2.2, threadpoolctl 3.7.0, and matplotlib
3.11.2 with its dependencies. Full installed metadata is recorded locally.

The three original measurement scripts are byte-identical copies of the
Python 3.14.9 sources. The extension changes only its interpreter guards and
the scope of its preservation check. Other local adaptations point package
tests at the existing sibling checkouts and label figures with the correct
interpreter. `source_adaptations.json` records each change and both hashes.
The solver adapters, generators, objectives, seeds, timing rounds, calibration,
affinity settings, and budgets are unchanged.

The Python 3.14.9 collection used NumPy 2.5.4 and SciPy 1.18.1. These are
separate environment runs with different native wheels and dependency
versions; differences cannot be attributed to the Python interpreter alone.
The audit compares recorded input hashes across runs where available.

## Sequence and coverage

1. Record the interpreter, packages, installed native binary hashes, and source
   checkout state; check dependency consistency.
2. Run all 17 paper-harness regression checks, the installed-wheel Python test
   suites for LAPX/lapcpp/laprust, Rust source tests, and native C++ tests.
3. Independently validate all 11,040 batch outputs.
4. Run the original single-call/batch, formulation, and scaling suites
   sequentially, with CPU-load monitoring. Planned timing samples: 756, 270,
   450, and 2,718 respectively. The original memory comparison uses ten
   fresh-process measurements.
5. Run the 132 extension timing cases and 405 fresh-process memory trials.
   Its saved plan must match the Python 3.14.9 plan exactly. The expected
   timing total is 4,752 samples across 528 method/case settings, using four
   inputs and nine rounds per setting.
6. Audit raw samples, summary quantiles, objectives, pairing, native binaries,
   and preservation. Generate and visually review six base charts and thirteen
   extended/focused charts, without building or revising the paper.

The extended shapes reach 5000 x 5000 and 64 x 16384 with exact transposes.
Structured cases vary congestion, forbidden-entry density, ties, and signed
costs. Rejection sweeps use eleven limits from zero through 4 on identical
inputs. Memory sweeps cover 256 x M, M = 256 through 4096, and nonsquare
transposes, with five fresh processes per method/condition. The extension
also validates empty, infeasible, tied, signed, and penalty-boundary fixtures
against independent exhaustive enumeration.

The exact extension protocol is preserved in
`../../extensions/python-3.14.9-20261024/PROTOCOL.md`; its references to the
interpreter and collection location describe that earlier run. The local
`extended/plan.json` defines this run's case grid and resource rules.

## Interpretation

Times include normal adapter/output work and exclude input generation,
validation, and calibration. Timing IQRs describe round-to-round variation
on the chosen inputs, not population confidence intervals. No timing samples
are pooled across suites or interpreter environments.

Memory values measure increments in the process working-set high-water mark
after the input is resident. They are not exact allocation counts. The
repeated extension records allocator headroom and transformed-buffer sizes
separately. Zero increments remain visible in its memory charts.

Focused timing charts omit JVX/JVS/JVC consistently; complete original-suite
comparisons remain available. Compact SciPy alternatives and unfavorable
congested cases are retained. Ratios use paired rounds within a suite.

## Reproduction

Run commands from this collection directory. `collect.py` requires a fresh
collection and refuses to overwrite a started measurement run. It preserves
an early metadata-export retry made necessary by Windows text encoding;
that retry occurred before any tests or measurements.

```powershell
C:\dev\exc\python\p311\python.exe -u collect.py
C:\dev\exc\python\p311\python.exe check_collection.py --data-only
C:\dev\exc\python\p311\python.exe make_figures.py
C:\dev\exc\python\p311\python.exe render_figures.py
C:\dev\exc\python\p311\python.exe check_collection.py
```

The final audit additionally requires `figure_review.json`, produced after
rendering and inspecting the PDFs and the separate base PNG layouts. Original
artifacts are protected by `protected_previous_files.json`; final evidence
is inventoried in `file_manifest.json`.
