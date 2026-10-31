# Dense benchmark campaign

This module provides deterministic inputs, controlled native builds, correctness
validation, bounded smoke checks, timing, diagnostics and analysis.

Run commands from the repository root. Install the pinned PyPI packages with
`python -m pip install -r benchmark/campaign/requirements.txt`. Controlled builds
also require Rust and C++ toolchains. Replace
`RUST_SOURCE_DIRECTORY` and `CPP_SOURCE_DIRECTORY` below with paths to your solver
source checkouts. Use a separate output directory for each environment.

If an environment already contains the same solver versions from another wheel
source, replace them explicitly before validation; a normal requirements install
keeps already satisfied versions:

```powershell
python -m pip install --force-reinstall --no-deps --only-binary=:all: --no-cache-dir --index-url https://pypi.org/simple lapx==0.13.0 laprust==1.2.0 lapcpp==1.2.0
```

Every `--output` below is optional. Without it, a command writes to
`benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/campaign-<command>` and
prints the path; `estimate` and `summarize` write beside their input. The
PowerShell examples use an explicit `$BenchOut` folder instead. Verify the
intended Python version before running:

```powershell
python --version
$BenchOut = Join-Path $env:TEMP "lap-paper-bench/windows-py31118"
```

## Inventory and controlled builds

```powershell
python -m benchmark.campaign manifest --output "$BenchOut/campaign-plan.json"
python -m benchmark.campaign topology
python -m benchmark.campaign.native_build --rust-source RUST_SOURCE_DIRECTORY --cpp-source CPP_SOURCE_DIRECTORY --output "$BenchOut/builds"
```

The inventory has 374 cases: 340 single-matrix cases (148 existing cases and
192 geometric cases), 14 batch/workspace workloads and 20 memory cases. One
environment session has 3,826 timing case/method conditions and 91 memory
conditions.

The optional `--profile light` selects 302 cases: 268 single-matrix cases
(148 existing and 120 geometric), the same 14 batch/workspace workloads and
20 memory cases. It has 3,286 timing conditions and the same 91 memory
conditions. All input sizes, all 25 layout-panel cases and all 33 policy-panel
cases are retained. The default profile is `full` and its catalog is unchanged.

| Coverage | Sizes |
| --- | --- |
| Geometric tracks × detections | Doubling sweep from 8 to 4096 rows, square and 5:4 |
| Uniform scaling | Square 16–1024, wide 16×64–512×2048, fixed 64×64–64×4096, blocks 16×20–1024×1280 |
| Large and extreme | Squares 1000–5000, rectangles up to 4000×5000, 64×4096–64×16384 and transposes |
| Structure at 256×320 | Congestion, forbidden entries, ties, signed costs and an 11-limit rejection sweep |
| Batches and memory | Batches of 64 × 64×80, 16 × 128×160 and 1024 × 8×10; workspace batches of 8 × 256×4096 and 8 × 256×8192 with transposes; memory from 256×256 to 256×4096 with transposes |

Every timing case runs the canonical methods. Required cases use JVRUST,
JVHYBRID, JVCPP, JVX, JVS and SciPy. Optional cases use JVRUST, JVCPP, JVX,
capped, dummy-column, diagonal-dummy and square-expanded SciPy, and two sparse
solvers: SciPy's `min_weight_full_bipartite_matching` (CSGRAPH) and LAPX
`lapmod` (LAPMOD). The sparse solvers receive only admissible edges, cost < L
after any gate, in Pylapy's `diag_row_no_pad` and `diag_split_sparse` layouts.
Building the sparse graph from the dense input is inside the timed call.
The geometry design is inspired by [Pylapy](https://github.com/raphaelreme/pylapy),
by Raphael Reme (MIT license). `adapters.py` identifies the adapted padded and
sparse layouts, source revision and retained MIT notice. This credits both the
upstream design and the adaptations; it does not claim identical benchmark
implementations or protocols.

Geometric inputs cross survival q ∈ {0.9, 0.6} with motion {0.25, 1}δ, where
δ = r^-1/2 is the motion scale. Each has required assignment and L = 1.5δ. At
256×320 only, a sensitivity grid adds L ∈ {0.5, 1.5, 3}δ with gates
{0.75, 1.5, none}δ.

Two explanatory studies use fixed panels, chosen before any timing:

- The **layout panel** (25 cases) runs all eleven cost and six shape layouts with
  SciPy, JVX and JVS. It covers the base and formulation cases and the primary
  256×320 geometric contracts.
- The **policy panel** (33 cases) runs the controlled builds: eight required or
  six optional arms per language. It covers the base and formulation cases and
  the primary 256×320 and 1024×1280 geometric contracts. Required square cases
  receive the reference arm only.

Padded baselines (JVX, JVS, square-expanded SciPy and every layout) are timed
while the padded square has at most 4096² entries, which is 128 MiB in float64.
Required square inputs are not padded, so they keep every method at all sizes.
The memory cases keep their padded methods to measure the expansion itself.

Light additionally stops JVX, JVS and square-expanded SciPy above 1024 rows
in geometry only, including JVX/JVS required squares without padding. Outside
256×320 and 1024×1280, light geometry retains only (q=0.9, motion=0.25δ) and
(q=0.6, motion=1δ). Both panel shapes retain all four q/motion combinations,
both objectives and their existing layout/policy/sensitivity coverage.

The 20 primary artifacts comprise eight policy settings and two additional
workspace settings in each language. W0 uses the controlled reference. Every
artifact starts from a pinned `git archive`, uses the same wrapper scaffold,
disables profiling and runtime policy overrides, and has a unique import name.
The build index records source revisions, archive/patch/artifact hashes, Python
identity, release flags and compiler logs. Isolated sources and patches remain
beside the artifacts. Original repositories are not modified. Use a separate
build directory for each interpreter/platform configuration.
Build arms sequentially: do not run concurrent build commands against the same
output directory and shared Cargo cache.

## Validation and bounded smoke

```powershell
python -m benchmark.campaign validate --build-index "$BenchOut/builds/build-index.json" --output "$BenchOut/validation.json"
$env:CAMPAIGN_NATIVE_BUILD_INDEX="$BenchOut/builds/build-index.json"
python -m pytest benchmark/campaign -q
python -m benchmark.campaign smoke --build-index "$BenchOut/builds/build-index.json" --output "$BenchOut/smoke"
python -m benchmark.campaign estimate --smoke-root "$BenchOut/smoke" --output "$BenchOut/estimate"
```

Smoke is a fixed 13-case subset, selected by coverage rather than outcomes. It
uses two separately calibrated inputs, two rounds, a 5 ms block target and one
memory trial. It exercises full small layout/policy sweeps, the sparse solvers,
a nonredundant gate, negative penalties, the largest square (5000×5000) and
geometric (4096×5120) inputs, standard batches and tall workspace inputs on both
sides of the orientation retention threshold. These
records are explicitly unavailable to the publication aggregator. The runtime
estimate is a scheduling model with disclosed anchor quality and sensitivity
scenarios, not a confidence interval or a measured performance claim.

Tiny gates use exhaustive binary-rational enumeration. They compare the
penalized objective, not matching identity or optional cardinality. The adjacent
representable threshold fixture retains exact numerical-boundary discrepancies
as distinct diagnostics; it does not grant a blanket tolerance waiver. Strict
dyadic, feasibility, gate, mapping, output-type and immutability failures remain
blocking. Every materialized timing input additionally passes an independent
SciPy required/capped objective check (`rtol=1e-11`, `atol=1e-12`) outside timing.
Agreement within that tolerance is not a proof of universal exactness.

## Separate explanatory diagnostics

```powershell
python -m benchmark.campaign.diagnostic_build --rust-source RUST_SOURCE_DIRECTORY --cpp-source CPP_SOURCE_DIRECTORY --output "$BenchOut/diagnostic-builds"
python -m benchmark.campaign diagnose --build-index "$BenchOut/diagnostic-builds/diagnostic-build-index.json" --output "$BenchOut/diagnostics"
```

This creates another 20 artifacts with separate namespaces and a separate index.
The primary loader rejects them. Seventeen atomic event/capacity fields describe
row scans, displacement chains, augmentations, rejection endings, actual
orientation-vector capacity, cleanup, retention and fresh-state destruction.
Definitions and scope are stored with the artifacts. Rectangular event counters
do not describe the separate required-square algorithm. Capacity maxima are
actual allocations, not payload estimates or summed per-worker memory.

The default diagnostic run is a bounded seven-case check with two inputs. It
records the admissible graph's density, degrees and isolated vertices, geometric
latent points/correspondences, independently checked objectives and cardinalities,
and construction-only timings/dimensions/buffer bytes for each layout. Construction
is measured three times per input/layout, deduplicating identical constructions
across backends. Native counter samples follow one excluded warmup call. Inputs
stream one at a time in fresh case/placement processes under R1. None of these
times can substitute for primary complete-call latency.

A complete diagnostic pass requires `diagnose --full --allow-full-campaign`.
Run it once per environment, after the timing session. Runtime estimates budget
this extra work explicitly.

## Measurement contract

Each environment runs one primary session. Every condition uses five frozen
inputs, one screening and one warmup call per input, five rounds, and separately
calibrated approximately 20 ms blocks per method/input. These settings are
`INPUT_COUNT`, `ROUNDS`, `BLOCK_SECONDS` and `WARMUP_CALLS` in `catalog.py`.
Light retains five inputs and the same 20 ms target but uses three rounds;
the frozen manifest supplies the session and aggregation settings.
One call is the minimum; slow calls are never truncated. Repetitions remain fixed across rounds. Input order is
shared across comparison arms and worker counts, with a separate seeded shuffle
of methods within each input. Complete calls include dtype/layout conversion,
gating, transformation, solve, decoding and int64 output. Selection/imports,
new generic domain scans, objective checks, IPC and placement probes are outside
the timed blocks; the backend's own public API validation remains inside.

The worker records input and source identities, method/build metadata, screening
objectives/cardinalities, separate phase accounting, every primary block, and
placement checks. Each session executes frozen copies of its Python sources.
Campaign identities prevent mixing implementation generations. Environment
identities include hardware, OS, Python, library/native artifact hashes, power
settings and placement, so different configurations cannot be pooled by accident.

The supervisor runs off all logical threads of the selected physical cores.
Serial and one-worker arms use the designated single hardware thread. Four-worker
arms use one hardware thread on each of four performance cores under a shared
affinity mask; migration within that mask is allowed. Calling and persistent
worker thread masks are checked before warmup and after measurement. Each
workload/worker-count group starts a fresh process. The released JVX batch API
creates its Python pool inside every call; that cost remains included and is
labelled `pool_lifetime=per_call`. A separate excluded initializer probe checks
its worker-mask inheritance. This is not a claim about actual CPU residency.

W1 recreates native solver scratch while retaining orientation storage. W2
recreates the whole native worker. Neither changes Python conversion or output
buffers. Destruction follows output extraction/mapping. Rust's stock parallel
cleanup is per matrix; C++'s stock parallel cleanup is per worker task loop, and
serial cleanup occurs after the batch. Compare workspace arms within a language.

R1 limits a cost array to 4 GiB, the conservative allocation estimate to 12 GiB,
and measured process-tree RSS to 16 GiB, with at least 24 GiB available before
preparation. An external watchdog samples nominally every 50 ms and records its
observed maximum interval/overshoot. It limits screen calls to 60 s, timed blocks
to 120 s, and cumulative method/case/session solver activity to two hours across
restarts. The 1 GiB runtime/cached-state reserve and transformed-array/worker
allocation model are explicit in `resources.py`; actual RSS enforcement is
independent. Host-headroom shortages postpone work instead of censoring a solver.

A deterministic resource breach retains the failed attempt, censors its arm,
and restarts a complete paired schedule using only the remaining admitted arms.
No partial rounds are stitched. Infrastructure or placement failure makes the
attempt unavailable; input mutation invalidates its entire corpus. Resource
exclusions are neither zero timings nor wins. Session output directories are
immutable and cannot be reused by a new invocation.

Insufficient host headroom stops the shard with `status=postponed` and exit code
2. `remaining_case_ids` retains the interrupted case and all later selected
cases. Retry that complete case in a fresh output directory once headroom is
available. Any earlier placements or memory trials from that interrupted case
remain recorded but are excluded from aggregation, so a complete retry does not
create duplicate estimates. Earlier completed cases stay valid.
Headroom is checked before every fresh memory trial.

Combined summaries retain admission outcomes, failure and censoring reasons,
and attempt history separately from latency statistics. A recorded rejection
before timing is distinguished from work that has not been attempted.

Memory trials use five fresh processes per condition, a resident input before
baseline, current RSS plus platform high-water counters, and a fixed 50 ms output
hold. The peak is captured before reference validation. Baseline headroom is
retained because previous import/input-allocation peaks can conceal increments.
Exact transformed buffer arithmetic is reported separately from measured peaks.

## Full session and summary

Full sessions are intentionally not part of smoke. The executable guard requires
Windows/Linux Python 3.11.18 or 3.14.10 and controlled artifacts built with that
same patch version. Validate each target configuration, then run the smoke and
the estimate to check that a session fits the host's time budget.

```powershell
python -m benchmark.campaign run --allow-full-campaign --build-index "$BenchOut/builds/build-index.json" --output "$BenchOut/session"
python -m benchmark.campaign summarize "$BenchOut/session" --output "$BenchOut/summary.json"
```

For light, add `--profile light` to `manifest`, `validate`, `smoke`, `run`,
`estimate` and `diagnose` as needed. `run` still requires the explicit
`--allow-full-campaign` opt-in at this suite-level interface; the top-level
launcher selects everything through `run --allow-full-campaign-light`.
`estimate` and `summarize` infer the profile from saved manifests by default.
Summarize full and light sessions separately; mixed profiles or an explicit
profile that disagrees with the saved session are rejected.

One invocation runs every case in manifest order. `--case-start`/`--case-stop`
(zero based, stop exclusive) or repeated `--case-id` select a fixed shard, for
example to resume after a postponement; pass all shard directories of the
session to `summarize`. Each shard records the full planned manifest, selected
and unselected case IDs, and the remaining selected cases.

The primary summary takes the median of the five (light: three) round means, each giving equal
weight to the five inputs. Ratios are paired within the session. Missing inputs,
rounds or reference arms make an estimate unavailable. Smoke cannot fill missing
cells. Do not pool OS, Python, objective, family or implementation generations
into a single headline ratio. Retain planned coverage, censoring, losses,
absolute latency and worst slowdowns alongside any aggregate.

[Application replay and tracker updates](../application/README.md) and the
[LSAPE study](../lsape/README.md) have separate runnable suites and measurement
contracts.
