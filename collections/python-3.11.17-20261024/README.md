# Python 3.11.17 paper measurements

Collection completed successfully on 24 October 2026. All validation and
audits passed, with no measurement failures or resource exclusions. All
19 charts were rendered and visually reviewed.

| Completed evidence | Count |
| --- | ---: |
| Original-suite timing samples | 4,194 |
| Extended timing samples | 4,752 across 132 cases |
| Total timing samples | 8,946 |
| Memory measurements | 415, including 405 repeated extension trials |
| Independent enumeration solver calls in the original suites | 27,320 |
| Independently checked batch outputs | 11,040 |
| Extended timed-input objective checks | 2,112 |
| Edge fixtures | 23 with 129 solver checks, including 15 expected infeasibility checks |
| Package Python tests | 5,038 passed; native Rust/C++ tests also passed |
| Paper-harness regression tests | 17 passed |
| Previous files preserved and checked by SHA-256 | 863 |

This is a separate replication of both Python 3.14.9 collections. Existing
results, figures, manuscript, and arXiv files are preserved. The prepared
interpreter is `C:\dev\exc\python\p311\python.exe`.

| Package | Python 3.11.17 run | Previous Python 3.14.9 run |
| --- | --- | --- |
| lapx | 0.11.1 | 0.11.1 |
| laprust | 1.0.0 | 1.0.0 |
| lapcpp | 1.0.0 | 1.0.0 |
| NumPy | 2.4.6 | 2.5.4 |
| SciPy | 1.17.1 | 1.18.1 |

These are separate environment runs. Different dependency versions and native
wheels mean that timing differences cannot be attributed to Python alone.
The seeds, case grids, adapters, repetition counts, affinity settings, and
budgets match the previous collection. Recorded input hashes are compared
across environments as part of the audit.
All recorded hashes matched: 40 formulation inputs, 188 scaling inputs,
528 extended timing inputs, 405 memory-trial inputs, and three batches.
Method order also matched in all five timing suites.

## Charts

[Combined 19-page chart PDF](charts.pdf) contains the six base-suite charts
followed by the thirteen extended/focused charts. Every chart also has a PNG
preview and an SVG export alongside its individual PDF.

| Chart | Preview | PDF |
| --- | --- | --- |
| All-solver single-call comparison | [PNG](figures/single_runtime.png) | [PDF](figures/single_runtime.pdf) |
| Batch comparison | [PNG](figures/batch_runtime.png) | [PDF](figures/batch_runtime.pdf) |
| Original memory comparison | [PNG](figures/memory_peak.png) | [PDF](figures/memory_peak.pdf) |
| Original required scaling | [PNG](figures/scaling_required.png) | [PDF](figures/scaling_required.pdf) |
| Original finite-limit scaling | [PNG](figures/scaling_optional.png) | [PDF](figures/scaling_optional.pdf) |
| Original congested-block scaling | [PNG](figures/scaling_blocks.png) | [PDF](figures/scaling_blocks.pdf) |
| Large squares | [PNG](extended/figures/large_square.png) | [PDF](extended/figures/large_square.pdf) |
| Large rectangles and transposes | [PNG](extended/figures/large_rectangles.png) | [PDF](extended/figures/large_rectangles.pdf) |
| Extreme aspect ratios | [PNG](extended/figures/extreme_rectangles.png) | [PDF](extended/figures/extreme_rectangles.pdf) |
| Required structure sweeps | [PNG](extended/figures/structure_required.png) | [PDF](extended/figures/structure_required.pdf) |
| Finite-limit structure sweeps | [PNG](extended/figures/structure_optional.png) | [PDF](extended/figures/structure_optional.pdf) |
| Signed costs and orientation | [PNG](extended/figures/signed_orientation.png) | [PDF](extended/figures/signed_orientation.pdf) |
| Runtime and realized rejection | [PNG](extended/figures/rejection_sweep.png) | [PDF](extended/figures/rejection_sweep.pdf) |
| Repeated memory sweeps | [PNG](extended/figures/memory_sweeps.png) | [PDF](extended/figures/memory_sweeps.pdf) |
| Focused base cases | [PNG](extended/figures/focused_existing_cases.png) | [PDF](extended/figures/focused_existing_cases.pdf) |
| Paired formulation speedups | [PNG](extended/figures/formulation_speedup.png) | [PDF](extended/figures/formulation_speedup.pdf) |
| Focused base scaling | [PNG](extended/figures/focused_existing_scaling.png) | [PDF](extended/figures/focused_existing_scaling.pdf) |
| Focused congested blocks | [PNG](extended/figures/focused_existing_blocks.png) | [PDF](extended/figures/focused_existing_blocks.pdf) |
| Large-input implementation ratios | [PNG](extended/figures/large_implementation_ratios.png) | [PDF](extended/figures/large_implementation_ratios.pdf) |

Focused views consistently select the faster candidate set while retaining
compact SciPy baselines and unfavorable congested cases. The base-suite charts
retain the full solver comparisons. Samples are not pooled between suites or
interpreter environments. Timing IQRs are descriptive variation, and measured
memory increments are process high-water changes rather than allocation counts.

## Evidence and reproduction

- [Protocol](PROTOCOL.md), [collection plan](collection_plan.json), and [extension plan](extended/plan.json)
- [Runtime and native binary identities](results/runtime.json), [installed packages](installed_packages.json), and [source adaptations](source_adaptations.json)
- [Single-call samples](results/single_samples.csv), [batch samples](results/batch_samples.csv), [formulation samples](results/formulations/samples.csv), and [scaling samples](results/scaling/samples.csv)
- [Extended timing samples](extended/samples.csv), [summary](extended/summary.csv), [memory trials](extended/memory_samples.json), and [memory summary](extended/memory_summary.csv)
- [Package validation](results/test_validation.json), [batch validation](results/batch_validation.json), and [edge validation](extended/edge_validation.json)
- [Audit](audit.json), [visual review](figure_review.json), [run status](collection_run.json), [logs](logs/), and [file inventory](file_manifest.json)

The paper has not been revised to incorporate these measurements.
