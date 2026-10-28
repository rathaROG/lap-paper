# Native Rectangular Linear Assignment with Implicit Rejection

Paper, benchmark programs and results for *Native Rectangular Linear Assignment
with Implicit Rejection: A Jonker–Volgenant Solver without Square Expansion*.

**[arXiv:2610.35948](https://arxiv.org/abs/2610.35948)** ·
**[Paper PDF](output/pdf/native_rectangular_assignment.pdf)** ·
[LaTeX source](latex/main.tex) ·
[arXiv source ZIP](output/arxiv/native_rectangular_assignment_arxiv.zip) ·
[HTML preview](output/arxiv/html/index.html)

The paper evaluates **LAPX 0.12.0, laprust 1.1.0 and lapcpp 1.1.0**,
with Windows results in the main text, a second Windows environment in
Appendix A, Linux replications in Appendix B, and a tracking application
study in Appendix C.

This repository provides the paper's LaTeX source, publication files and the
standalone [benchmark programs](benchmark/README.md) behind every measured
table and figure, and the [saved measurements](benchmark/RESULTS.md)
used for the reported results.

## Reproduce the results

Use Python 3.14.9 for the primary paper environment:

```sh
pip install -r benchmark/requirements-py314.txt
python benchmark/run_all.py
```

For the Python 3.11.17 replication, use `benchmark/requirements-py311.txt`.
New runs are saved under `benchmark/results/full-YYYYMMDD/`. The LSAPE and
real-world studies have separate instructions in their benchmark folders.

To verify the saved paper measurements without rerunning benchmarks:

```sh
python benchmark/verify_results.py
```

## Findings

The following findings are reported in the checked-in manuscript:

- JVRUST beats ordinary SciPy in all 71 required non-batch cases in each
  environment, capped SciPy in 110–115 of 125 finite-limit cases, and expanded
  SciPy in five additional finite-limit cases.
- Finite-limit congestion retains counterexamples: the largest measured
  native/SciPy median-time ratio is about 1.50. The reduction caps were
  developed using the paper's workload families, so evaluation is not held out
  from heuristic development.
- Across the three batching workloads and four environments, four-worker
  JVRUST reduces sequential-loop latency by 1.85–5.86 times. For 1024 tiny
  problems, one-worker batching alone gives 2.04–2.48 times.
- All 3,925 compared timing and repeated-memory objectives agree exactly
  across environments. Windows peak-working-set increments and Linux
  high-water-RSS increments require separate interpretation; a zero increment
  does not establish zero allocation.
- The controlled LSAPE study separates formulation, specialization, layout
  and rejection-slack tracking. Matched-layout specialization records 66 gains
  and one estimate inside the descriptive 3% band; compact layout has no gain
  beyond that band, and rejection-slack tracking has no consistent benefit.
- The tracking study reports assignment-replay speedups of 1.17–3.16 times
  against SciPy, with smaller tracker-update gains and near-equal SORT medians.

These results characterize particular implementations and environments.
Timing IQRs and ratio bands are descriptive. The comparisons do not isolate
operating-system, interpreter, compiler or implementation-language effects.
The contribution is software integration and performance evaluation, building
on established rectangular and optional-assignment methods.

## Solver labels

| Label | Implementation / entry point |
| --- | --- |
| JVRUST | laprust 1.1.0 `lapjv`; the same native algorithm through LAPX 0.12.0 `lapjvr` |
| JVHYBRID | LAPX 0.12.0 `lapjvh`, Rust with C++ square completion |
| JVCPP | lapcpp 1.1.0 `lapjv`, C++ throughout |
| JVX | LAPX gatagat/lap-derived baseline; timed entry point `lapjvxa` |
| JVS | LAPX src-d/lapjv-derived baseline; `lapjvsa`, configured float64 |
| JVC | LAPX py-lapsolver-derived baseline; `lapjvc` |
| SciPy | `scipy.optimize.linear_sum_assignment`, with the formulation specified in the paper |

## Repository layout

| Location | Purpose |
| --- | --- |
| `latex/main.tex` | Editable manuscript |
| `latex/references.bib`, `latex/main.bbl` | Bibliography source and supplied formatted bibliography |
| `latex/images/` | Figure PDFs used by the manuscript |
| `figures/` | Exported publication figures |
| `output/pdf/native_rectangular_assignment.pdf` | Paper PDF and default build output |
| `output/arxiv/` | Source archive and HTML preview |
| `.latexmkrc` | Default LaTeX build settings |
| `benchmark/` | Standalone benchmark programs for every measured result; see its README |
| `benchmark/results.zip` | Saved publication measurements, summaries and provenance; see `benchmark/RESULTS.md` |
| `benchmark/results/` | New benchmark runs, created as `full-YYYYMMDD/` by default and ignored by Git |

The mathematical formulation and rejection-terminal principle build on
established assignment literature. The paper makes no claim that rectangular
or optional assignment is new.
