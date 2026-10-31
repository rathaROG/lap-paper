# Native Rectangular Linear Assignment with Implicit Rejection

Paper and benchmark code for *Native Rectangular Linear Assignment with Implicit Rejection: A Jonker–Volgenant Solver without Square Expansion*.

**[arXiv:2610.35948](https://arxiv.org/abs/2610.35948)** · [Paper PDF](output/pdf/native_rectangular_assignment.pdf) · [LaTeX source](latex/main.tex) · [arXiv source ZIP](output/arxiv/lap-paper-arxiv.zip)

The manuscript reports the four saved light-profile environments from 30–31
October 2026, with separate application and Windows LSAPE evidence.

Regenerate figures, tables and source-indexed supplementary data without running
a benchmark:

```sh
python -m pip install -r benchmark/requirements-artifacts.txt
python benchmark/paper_evidence.py extract
python benchmark/generate_paper_artifacts.py
latexmk -pdf latex/main.tex
```

The [frozen evidence bundle](benchmark/evidence/README.md) contains five
checksummed archives: four primary light sessions and one separate sensitivity
session. Extraction restores the saved files under `benchmark/runs/`; the
generator then reads them without executing solvers. Its
[artifact guide](benchmark/paper-artifacts/README.md) identifies the primary
inputs and supplementary exports. Exact values for every case remain in those
indexed exports; the optional figure-atlas source is
`latex/supplementary_figures.tex`.

The configured LaTeX build writes the main paper to
`output/pdf/native_rectangular_assignment.pdf`. Keep only that paper in
`output/pdf/`; the submission archive is `output/arxiv/lap-paper-arxiv.zip`.
Both the PDF and ZIP are final deliverables to include in version control.
Within `output/`, only `output/build/` is ignored; it contains temporary LaTeX
files and compilation logs.

For arXiv, include the generated `latex/main.bbl` with `main.tex`. After changing
the bibliography, rebuild and copy `output/build/native_rectangular_assignment.bbl`
to `latex/main.bbl`. Put `main.tex`, `main.bbl` and `references.bib` at the upload
archive root, with only the figures and table fragments used by the paper in
their `images/` and `generated/` subdirectories.
Refresh `output/arxiv/lap-paper-arxiv.zip` after changing the manuscript,
bibliography, figures or tables, so it matches the final PDF.

## Benchmarks

The [benchmark guide](benchmark/README.md) covers three suites:

- the assignment campaign: dense and sparse solvers, geometric tracking-style
  inputs from 8 to 4096×5120, uniform and structured costs, padding layouts,
  initialization policies, batching and memory
- cached pyppbox tracking and assignment replay
- the controlled LSAPE study

The assignment campaign and the application suite run one session per
environment, and LSAPE runs three short fresh-process sessions. A complete run is
sized to take a few hours. Every suite provides validation and a bounded smoke check before
full measurements. Without an explicit output directory, commands write under
`benchmark/runs/`; each suite guide describes its defaults.

`python -m benchmark run --allow-full-campaign` keeps the full protocol.
`python -m benchmark run --allow-full-campaign-light` selects a shorter paper
protocol with every input size and five inputs, three timing rounds, and a
reduced geometric sweep while preserving the full explanatory panels. See the
benchmark guide for its precise coverage and the geometry baseline cutoff.

## Solver labels

| Label | Implementation / entry point |
| --- | --- |
| JVRUST | laprust 1.2.0 `lapjv`; the same native algorithm through LAPX 0.13.0 `lapjvr` |
| JVHYBRID | LAPX 0.13.0 `lapjvh`, Rust with C++ square completion |
| JVCPP | lapcpp 1.2.0 `lapjv`, C++ throughout |
| JVX | LAPX gatagat/lap-derived baseline; timed entry point `lapjvxa` |
| JVS | LAPX src-d/lapjv-derived baseline; `lapjvsa`, configured float64 |
| SciPy | `scipy.optimize.linear_sum_assignment`, with the formulation specified in the paper |
| CSGRAPH | `scipy.sparse.csgraph.min_weight_full_bipartite_matching` on admissible edges |
| LAPMOD | LAPX `lapmod`, sparse Jonker–Volgenant on admissible edges |

## Repository layout

| Location | Purpose |
| --- | --- |
| `latex/main.tex` | Authoritative editable manuscript |
| `latex/supplementary_figures.tex` | Paginated figure atlas |
| `latex/references.bib` | Bibliography source (BibTeX generates the formatted bibliography during the build) |
| `latex/main.bbl` | Generated bibliography included for arXiv submission |
| `latex/images/` | Figures included by the manuscript |
| `output/pdf/native_rectangular_assignment.pdf` | Final main-paper PDF; not ignored |
| `output/arxiv/lap-paper-arxiv.zip` | arXiv source upload, including `main.bbl`; not ignored |
| `output/build/` | Temporary LaTeX files and logs; ignored |
| `benchmark/README.md` | Entry point for the benchmark suites |
| `benchmark/evidence/` | Frozen observations and checksum manifest for regeneration |
| `benchmark/campaign/` | Assignment campaign implementation and tests |
| `benchmark/application/` | Frozen application inputs, tracking/replay code and tests |
| `benchmark/lsape/` | Controlled LSAPE sources, case catalogue and tests |
