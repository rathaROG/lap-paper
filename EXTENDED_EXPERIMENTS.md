# Extended measurements in both Python environments

This protocol adds large and structured inputs, rejection sweeps, repeated
memory measurements, and focused charts. The original Python 3.14.9 extension
and its Python 3.11.17 replication are complete and incorporated into the
revised manuscript. Their measurements, original figures, logs, and collection
audit reports remain unchanged. Publication-specific layouts are separate.

The full collection is saved in `extensions/python-3.14.9-20261024/`. Its
`plan.json` is written before measurement and contains the complete case grid,
solver lists, seeds, timing settings, and resource rules. This is a separate
measurement session, not a replacement for the original results. The matching
replication is in `collections/python-3.11.17-20261024/extended/`; its outer
collection records interpreter/output guard adaptations. Both runs completed
all 4,752 extension timing samples, 405 memory trials, and 129 edge checks,
without measurement failures or resource exclusions. See the root README for
the environment comparison and publication audit.

## Coverage

| Group | Inputs |
| --- | --- |
| Large squares | 1000, 2000, 3000, and 5000 rows/columns |
| Large rectangles | 1000 x 2000, 2000 x 3000, 4000 x 5000, and exact transposes |
| Extreme rectangles | 64 x 4096, 64 x 8192, 64 x 16384, and exact transposes |
| Congestion | 256 x 320 and 512 x 640; 25%, 50%, 62.5%, 75%, or 90% of rows compete for the first quarter of columns |
| Forbidden entries | The same two shapes; probabilities 0, 0.5, 0.9, and 0.99; a finite diagonal is retained to guarantee required-assignment feasibility |
| Ties | The same two shapes; 1, 2, 8, or 256 quantization levels |
| Signed costs | 256 x 320 and its transpose; costs in [-1,1), 50% nominal forbidden entries, finite feasible diagonal |
| Rejection | 128 x 160 and 256 x 320; uniform costs with 50% forbidden entries and congested costs; 11 limits from zero through 4 |
| Memory | 256 x M for M = 256, 512, 1024, 2048, 4096, plus the nonsquare transposes; required and finite-limit conditions |

There are 132 timing cases, each with four candidate methods, four input
matrices, and nine rounds. The planned total is 4,752 timing samples. There are
81 memory method/shape/regime conditions with five fresh processes each, for
405 trials. A separate exhaustive check covers empty inputs, single-row/column
inputs, all-forbidden inputs, infeasibility, tied optima, signed costs, and
floating-point values immediately around a penalty boundary.

## Solver selection and comparability

Required timing cases use JVRUST, JVHYBRID, JVCPP, and native rectangular SciPy.
Finite-limit timing cases use JVRUST, JVCPP, capped SciPy, and dummy-column
SciPy. JVHYBRID shares the Rust finite-limit path. This selection is fixed for
all sizes and structures; these charts describe the selected implementations.
The full original comparisons with JVX, JVS, and JVC remain available.

Memory compares JVRUST, JVCPP, SciPy, and JVX for required assignment. Finite
limits compare JVRUST, JVCPP, capped SciPy, dummy-column SciPy, and
square-expanded SciPy. This includes compact alternatives as well as a dense
expansion reference.

All timed costs and native precision settings are float64. Calls return int64
pairs and include their normal adapter/output work. The existing paper's solver
adapters are reused without modification. The shared workspace benchmark grid
in `../benchmarks/` informed the larger shapes, but its default float32 JVS,
single-round 5000-sized cases, and post-filter tracking objective are not used.

## Inputs, repetitions, and validation

The master seed is 20261024. Stable input-group keys and instance indices define
four matrices per timing setting. Limits and structure parameters do not enter
the random seed. Threshold sweeps therefore use identical costs and masks;
structure sweeps share their underlying random arrays; tall and wide timing
inputs are exact transposes, presented as C-contiguous arrays. Hashes of both
actual and canonical arrays are recorded.

Large finite-limit inputs use L = 2 / max(N,M), chosen in advance to retain a
rejection-active regime as size grows. Actual match counts are recorded for
every method and input. Congestion sweeps use L = 0.5, forbidden sweeps use
L = 2/M, quantized-cost sweeps use L = 0.25, and signed optional inputs use
L = -0.25. Rejection sweeps use L = 0, 0.0005, 0.001, 0.002, 0.005, 0.01,
0.02, 0.05, 0.1, 0.5, and 4.

Each method is validated and calibrated outside timing. Nine rounds shuffle
the candidate order. A block targets 30 ms, with at least one cycle across all
four matrices. Large cases retain all nine rounds. Timing excludes input
generation, validation, hashing, and checkpoint writes. Each child process uses
the original run's single logical CPU and limits library thread pools to one.
The controller and CPU monitor use other available CPUs.

Every output is checked for feasibility and objective equality against SciPy
or the capped formulation, at relative and absolute tolerances of 1e-10. The
small edge fixtures use an independent exhaustive objective enumerator.
Distinct optimal pairs are accepted in tied cases. Input hashes are checked
again after each timing case to detect mutation.

Interquartile ranges describe timing variation on the four chosen inputs;
they are not confidence intervals over an application population. Changing a
rejection limit changes the optimization problem as well as its cardinality,
so the sweep characterizes behavior rather than isolating a causal effect of
the number of rejected rows.

## Resource rules and incomplete data

The predeclared timing budget is 240 seconds per method and case, excluding
validation/calibration. A method whose probe projects more than this is
recorded as `budget_excluded` before its rounds. A method exceeding the budget
during measurement retains its partial samples with `budget_exceeded`, but
receives no summary. The whole-case wall timeout is 1200 seconds; memory trials
have a 120-second wall timeout. Missing measurements appear as gaps rather than
zeroes. `method_status.json`, case records, and logs retain the reasons.

Cases are checkpointed and the runner can resume with the same saved plan and
byte-identical measurement sources. Unexpected correctness errors stop the run.

## Memory interpretation

Each trial launches a fresh process, warms its method on a small input, creates
the resident float64 input, and records the process working-set high-water mark
before and after solving. Validation and reference solving happen after the
measured peak is captured. Hashing uses a memory view and does not allocate an
extra dense input copy before the baseline.

Memory orientations use independent, distribution-matched arrays, avoiding a
large temporary transpose allocation before the baseline. Within a shape and
trial, all methods receive identical input values. Additional peak, current
working set, baseline high-water headroom, and exact transformed-buffer sizes
are stored separately. Transformed-buffer sizes do not include internal solver
workspaces or orientation copies; zero transformed bytes do not mean zero
allocation. Peak increments are affected by allocator reuse and residency and
are not exact allocation counts.

The five trial seeds generate five paired inputs per condition. Their spread
includes both fresh-process and input variation.

The memory figure includes zero increments using a linear scale below 0.25 MiB
and a logarithmic scale above it. Medians and IQRs summarize five fresh-process
trials per condition.

## Commands and files

From this repository, using the prepared Python 3.14.9 interpreter:

```powershell
C:\dev\exc\python\p314\python.exe -m pytest -q tests/test_extended_experiments.py
C:\dev\exc\python\p314\python.exe run_extended_experiments.py --output tmp/new-extension-run
C:\dev\exc\python\p314\python.exe plot_extended_results.py tmp/new-extension-run
C:\dev\exc\python\p314\python.exe check_extended_results.py tmp/new-extension-run
```

These commands are for a new campaign, not the frozen completed directory.
For the revised paper, use `python check_artifact.py`; historical preservation
guards include the paper state at collection time. The measurement command
requires a new output directory; add `--resume` only
to continue that exact saved run. `--smoke` selects small inputs in a separate
output directory. The collector deliberately verifies Python 3.14.9, the
prepared interpreter path, installed package versions, and unchanged native
binaries from the original run.

`samples.csv`, `summary.csv`, `checks.json`, `datasets.json`, and individual
`cases/` records hold the timing evidence. Memory evidence is in
`memory_samples.json`, `memory_summary.csv`, and `memory_cases/`. `cpu_load.csv`
and `logs/` retain run conditions and progress. `sources/` preserves the exact
measurement code. `audit.json` verifies counts, summaries, objective checks,
paired-input hashes, binary identity, and preservation of earlier artifacts.

The figure generator writes only this extension's `figures/`. It creates new
views of the extension and focused views of the original data. The formulation
speedup chart uses the median and IQR of nine paired round ratios; its numbers
are distinct from ratios of two independently summarized medians. No timing
samples are pooled across the old and new sessions.
