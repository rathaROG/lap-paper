# Application benchmarks

This suite measures assignment replay and cached Centroid/SORT tracker updates
using pyppbox. It is separate from the dense-assignment campaign because its
objective, float32 tracking costs and int32 output contract differ.

The repository includes four frozen detection/reference caches, each with 999
frames, and eight captured assignment corpora containing 7,539 matrices. These
inputs cover MOT16-04 and MOT20-03 excerpts, ground-truth and RF-DETR Nano
detections, and both trackers. Input hashes, tracker source identities and the
unchanged solver/timing function identities are checked. See
[ATTRIBUTION.md](ATTRIBUTION.md) for data and upstream source terms.

Cached experiments require no video decoding, detector inference, model weights
or GPU. Install `requirements-tracking.txt` and `requirements-report.txt`, plus
LAPX 0.13.0 and laprust/lapcpp 1.2.0. Select pyppbox 4.0.0 at commit
`f3f1e8e671d7cad7e0df2f2469ae0f5ee6c27258` using `--pyppbox-root` or an installed
package with the identical tracker sources. Use one OpenCV distribution in the
environment; the supplied requirement selects its headless CPU build.

Run these commands from the repository root, replacing output and checkout
paths with your own:

```powershell
python -B -m benchmark.application --verify-inputs
python -B -m benchmark.application --validate --output VALIDATION_DIRECTORY
python -B -m benchmark.application --smoke --pyppbox-root PYPPBOX_DIRECTORY --output SMOKE_DIRECTORY
```

Validation checks all 7,539 matrices in both precision views against SciPy using
all eight comparison interfaces plus the Rust alias: 135,702 backend/matrix
checks. Exact nonzero objective gaps and tolerance failures are retained
separately; a tolerance pass is not a claim of exact arithmetic.

The bounded smoke uses all eight pipelines and eight methods, 16 frames per
pipeline, two warmup frames, one timing round and eight frozen matrices per
replay corpus in both precisions. Separate profiling/capture passes check that
instrumentation preserves each method's predictions. Smoke records are explicitly
unavailable as publication measurements. Missing dependencies or failed solvers
fail the run.

Each invocation requires a fresh output folder. Without `--output` it writes to
`benchmark/runs/{plat}-py{version}-{yyyymmdd}-{hhmm}/application-<mode>`. Its expanded inputs, source
snapshots, environment/binary identities, predictions, metrics and timings stay
inside that folder. Measurements share the host lock with the dense campaign;
run them sequentially. Calling-thread affinity is checked before and after
measurement. A failed run keeps its failure record and is never overwritten.

Full measurements require an explicit opt-in and Python 3.11.18 or 3.14.10:

```powershell
python -B -m benchmark.application --campaign --allow-full-campaign --pyppbox-root PYPPBOX_DIRECTORY --output NEW_CAMPAIGN_DIRECTORY
python -B -m benchmark.application --report-only --collection NEW_CAMPAIGN_DIRECTORY --output NEW_REPORT_DIRECTORY
```

`--tracker-only` and `--replay-only` also require `--allow-full-campaign` and can
select cases or solvers; include SciPy for the fixed comparison. Full campaigns
retain nine seeded-shuffled rounds, 32 warmup frames and 30 ms replay targets.
The top-level `run --allow-full-campaign-light` uses three rounds for both
tracker updates and replay, retaining every pipeline, solver, frame and matrix,
both precisions, validation and the separate profile/capture/evaluation passes.
The suite-level equivalent is `--campaign --allow-full-campaign --profile light`;
the chosen profile and round count are recorded in the run and protocol files.
Primary tracker timings include only sequential `tracker.update` calls. Person
construction, detection, video decoding, serialization and supplied-box accuracy
evaluation are outside that boundary. Normal solver adapter conversion,
validation and int32 pair normalization remain inside replay calls. Float64
replay widens the exact captured float32 values outside timing.

Centroid's gate and SORT's IoU post-filter and unique-feasible bypass are retained.
These experiments do not substitute optional assignment for the stock required
matching contract. Full campaigns separately record profiles and common SciPy
matrix captures; instrumented elapsed times do not fill primary timing tables.
Each run is one application session; do not pool its ratios with dense timings
or treat within-session rounds as independent days.

Optional detector-input regeneration is available through `--regenerate-inputs
--allow-full-campaign`, with explicit `--dataset-root`, `--checkpoint`,
`--pyppbox-root` and `--output`. It additionally requires the detector/GPU
dependencies and the supplied optional resolution patch. It is unnecessary for
the cached benchmark and is excluded from the smoke readiness check.

For the application unit tests, set `LAP_REALWORLD_PYPPBOX_ROOT` to the same
checkout and run `python -B -m pytest benchmark/application -q`.
