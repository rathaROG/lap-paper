# Python 3.14.9 extended results

Collection completed successfully on 24 October 2026 using the prepared
Python 3.14.9 environment: lapx 0.11.1, laprust 1.0.0, and lapcpp 1.0.0.
No methods failed or reached a resource exclusion. The previous results,
figures, and manuscript are unchanged; Python 3.11 was not run.

| Evidence | Completed |
| --- | ---: |
| Timing cases | 132 |
| Method/case settings | 528 |
| Timing samples | 4,752 |
| Timed-input objective checks | 2,112 |
| Fresh-process memory trials | 405 across 81 conditions |
| Edge fixtures | 23 with 129 solver checks, including 15 expected infeasibility checks |
| Standalone charts | 13, each in PNG, SVG, and PDF |
| Preserved files checked by SHA-256 | 100 |

The large inputs reach 5000 x 5000 and 64 x 16384, including exact transposes.
Special cases cover congestion, forbidden entries, tied and signed costs,
empty inputs, infeasibility, and values immediately around a rejection boundary.
Rejection sweeps reuse fixed inputs at eleven limits. Memory sweeps use five
fresh processes per condition and include compact and square-expanded SciPy
formulations.

## Figures

All charts show the selected solver set explicitly. The complete earlier
comparisons, including the slower solvers, remain in the original collection.
The focused views reuse saved measurements; no samples are pooled between
measurement sessions. Congested cases and performance reversals are retained.

| Chart | Preview | Vector PDF |
| --- | --- | --- |
| Large squares | [PNG](figures/large_square.png) | [PDF](figures/large_square.pdf) |
| Large rectangles and transposes | [PNG](figures/large_rectangles.png) | [PDF](figures/large_rectangles.pdf) |
| Extreme aspect ratios | [PNG](figures/extreme_rectangles.png) | [PDF](figures/extreme_rectangles.pdf) |
| Structure sweeps, required assignment | [PNG](figures/structure_required.png) | [PDF](figures/structure_required.pdf) |
| Structure sweeps, finite limits | [PNG](figures/structure_optional.png) | [PDF](figures/structure_optional.pdf) |
| Signed costs and orientation | [PNG](figures/signed_orientation.png) | [PDF](figures/signed_orientation.pdf) |
| Runtime and realized rejection | [PNG](figures/rejection_sweep.png) | [PDF](figures/rejection_sweep.pdf) |
| Memory sweeps | [PNG](figures/memory_sweeps.png) | [PDF](figures/memory_sweeps.pdf) |
| Focused original cases | [PNG](figures/focused_existing_cases.png) | [PDF](figures/focused_existing_cases.pdf) |
| Paired formulation speedups | [PNG](figures/formulation_speedup.png) | [PDF](figures/formulation_speedup.pdf) |
| Focused original scaling | [PNG](figures/focused_existing_scaling.png) | [PDF](figures/focused_existing_scaling.pdf) |
| Focused original congested blocks | [PNG](figures/focused_existing_blocks.png) | [PDF](figures/focused_existing_blocks.pdf) |
| Large-input implementation ratios | [PNG](figures/large_implementation_ratios.png) | [PDF](figures/large_implementation_ratios.pdf) |

SVG files use the same stems in [figures/](figures/). All PDFs were rendered
and visually reviewed. [Figure review](figure_review.json) records the checked
versions; [figure manifest](figure_manifest.json) records captions and provenance.

## Data and reproduction

- [Protocol](PROTOCOL.md), [predeclared plan](plan.json), and [environment](environment.json)
- [Timing samples](samples.csv), [timing summaries](summary.csv), and [objective checks](checks.json)
- [Input identities](datasets.json) and [method completion status](method_status.json)
- [Memory samples](memory_samples.json) and [memory summaries](memory_summary.csv)
- [Edge validation](edge_validation.json)
- [Paired formulation ratios](paired_formulation_speedups.json) and [implementation ratios](paired_implementation_speedups.json)
- [Audit](audit.json), [file inventory](file_manifest.json), [logs](logs/), and [source snapshots](sources/)

Timing error bars describe variation across nine rounds on four chosen inputs;
they are not population confidence intervals. Memory values are increments in
process peak working set after the input is resident, not exact allocation
counts. The protocol describes pairing, solver selection, resource rules,
validation, and these interpretation limits in detail.

The paper has deliberately not been revised to incorporate these measurements.
