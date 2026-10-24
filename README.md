# Native Rectangular Linear Assignment with Implicit Rejection

Manuscript and reproducibility artifact for *Native Rectangular Linear Assignment with Implicit Rejection: A Jonker–Volgenant Solver without Square Expansion*, revised 24 October 2026.

**[Paper PDF](output/arxiv/native_rectangular_assignment.pdf)** · [arXiv upload ZIP](output/arxiv/native_rectangular_assignment_arxiv.zip) · [HTML preview](output/arxiv/html/index.html) · [Manuscript](manuscript.md)

The arXiv edition is the sole paper version. It incorporates both completed Python collections, the larger and special-case experiments, repeated memory and rejection sweeps, and the cross-environment comparison. Python 3.14.9 supplies the main runtime plots; Python 3.11.17 supplies the replication tables and comparison. See [ARXIV.md](ARXIV.md) for building, verification, and submission settings.

## Environments and coverage

Both collections ran on 24 October 2026 on the same Intel Core i9-13900KS / Windows host. They use installed PyPI releases **lapx 0.11.1, laprust 1.0.0, and lapcpp 1.0.0**. Runtime records contain package metadata, source-checkout identities, and native-binary hashes. Source checkouts used for tests do not establish every compiler flag of downloaded wheels.

| Environment | NumPy | SciPy | Evidence |
| --- | --- | --- | --- |
| Python 3.14.9 | 2.5.4 | 1.18.1 | [Base results](results/), [extension](extensions/python-3.14.9-20261024/) |
| Python 3.11.17 | 2.4.6 | 1.17.1 | [Complete replication](collections/python-3.11.17-20261024/README.md) |

Each environment contains 8,946 timing samples across 994 method/case settings in 204 cases, plus 415 memory observations. The extension includes 5000-square matrices; rectangles through 4000 × 5000 and their transposes; extreme shapes through 64 × 16384; congestion, forbidden entries, ties, signed costs, eleven rejection limits, and five fresh-process memory trials per condition. The original all-solver comparisons remain available; focused extension plots deliberately select the native methods and compact SciPy baselines.

## Findings

- JVRUST and JVCPP each preserve their ordering against the appropriate SciPy baseline in all 201 non-batch cases. Both beat capped SciPy in the same nine of ten formulation cases.
- Congestion remains a counterexample. The base scaling crossover occurs between N=64 and N=128 in both runs; capped SciPy is 9.42× and 9.15× faster at 1024 × 1280 in Python 3.14 and 3.11.
- For 1024 problems of size 8 × 10, four-worker JVRUST batching improves on its Python loop by 5.10× and 4.73×.
- The added large square/rectangle grid keeps JVCPP and JVRUST within about 6.1%, while extreme tall inputs retain a roughly 3–3.8× JVCPP/JVRUST ratio. The original uniform grid has larger differences; neither implementation is universally fastest.
- All 3,520 recorded formulation/scaling/extended objectives and match counts agree exactly across environments. All package tests, batch checks, and edge fixtures pass.
- Repeated-memory medians differ by less than 0.09 MiB across all 81 conditions. Wide native inputs have small increments; tall inputs have appreciably larger increments. These high-water measurements are not allocation counts.
- Small gaps need qualification: the four-worker 64 × 80 JVCPP batch changes from approximately 37% slower to approximately tied with JVRUST. JVC and SciPy exchange order on the 32-square required case.

The [full comparison](comparisons/python-3.11.17-vs-3.14.9-20261024/README.md) records rank reversals and absolute timing shifts. NumPy, SciPy, native wheels, and sessions differ, so these results do not isolate a Python-version effect. Ratios in the paper divide within-run medians; supplementary paired-round-ratio charts are labeled separately. IQRs are descriptive, not confidence intervals.

## Build and verify from saved evidence

No command in this sequence reruns solver measurements:

```powershell
python build_pdf.py --refresh-figures
python check_arxiv_html.py
python check_artifact.py
```

Use a prepared interpreter with NumPy and Matplotlib, plus TeX Live 2025, LaTeXML, and Poppler. [ARXIV.md](ARXIV.md) includes the portable Windows HTML-check command. The default build reuses the saved publication PDFs; `--refresh-figures` prepares all 15 publication figures without overwriting collection figures. `python update_manuscript_tables.py` refreshes the eight base/appendix numeric tables only; interpretations and the extended summary table require review.

The publication audit verifies the 1,691-file frozen-evidence manifest, independently recalculates summaries and the cross-environment comparison, checks nine numeric manuscript tables and the quantitative base narrative, verifies extended rankings, and matches the PDF and HTML to the current ZIP. It writes [evidence_report.json](output/arxiv/evidence_report.json), leaving collection records unchanged.

Collection-local READMEs and audits describe their state when measurements finished, before this revision. Their preservation guards intentionally include the older paper and source state. Use the current publication audit for this revised checkout; do not rewrite historical manifests or treat their old paper-preservation guards as current-paper checks.

## New measurements

Completed directories are frozen evidence. Use a fresh output directory for a new campaign and review changes before promoting results. The measurement scripts are unchanged. See [extension protocol and commands](EXTENDED_EXPERIMENTS.md), the [Python 3.11 protocol](collections/python-3.11.17-20261024/PROTOCOL.md), and each saved plan. The extension collectors check exact interpreter and binary identities; switching PATH alone does not change their intended environment.

Python 3.14 dependency pins are in [requirements.txt](requirements.txt); Python 3.11 pins are in [requirements-py311.txt](requirements-py311.txt). Full installed inventories remain in the collections. Solver releases are pinned separately to lapx 0.11.1, laprust 1.0.0, and lapcpp 1.0.0. Do not install one environment's NumPy/SciPy pins over the other merely to build the paper.

Harness checks without a measurement campaign:

```powershell
python -m pytest -q -p no:cacheprovider tests
```

## Solver labels

| Label | Implementation / entry point |
| --- | --- |
| JVRUST | laprust 1.0.0 `lapjv`; identical native algorithm through LAPX 0.11.1 `lapjvr` |
| JVHYBRID | LAPX 0.11.1 `lapjvh`, Rust with C++ square completion |
| JVCPP | lapcpp 1.0.0 `lapjv`, C++ throughout |
| JVX | LAPX gatagat/lap-derived baseline; timed entry point `lapjvxa` |
| JVS | LAPX src-d/lapjv-derived baseline; `lapjvsa`, configured float64 |
| JVC | LAPX py-lapsolver-derived baseline; `lapjvc` |
| SciPy | `scipy.optimize.linear_sum_assignment`, native rectangular or explicitly labeled finite-limit formulation |

## Artifact map

| Location | Purpose |
| --- | --- |
| manuscript.md, references.bib | Editable paper and bibliography |
| templates/arxiv-preamble.tex | PDFLaTeX/LaTeXML-compatible article template |
| latex/ | Generated upload sources and vector figures |
| output/arxiv/ | PDF, source ZIP, HTML, build/evidence/visual reports |
| results/, figures/ | Original Python 3.14 base evidence and charts; publication charts are in figures/paper/ |
| extensions/python-3.14.9-20261024/ | Frozen larger/special-case Python 3.14 collection |
| collections/python-3.11.17-20261024/ | Frozen Python 3.11 replication, source copies, and 19-page chart packet |
| comparisons/python-3.11.17-vs-3.14.9-20261024/ | Frozen comparison, complete tables, and four review charts |
| publication/ | Evidence-preservation manifest and publication-figure provenance |
| paper_figures.py, paper_evidence.py | Publication figure preparation and evidence audit |
| run_experiments.py, compare_optional.py, benchmark_scaling.py, run_extended_experiments.py | Original measurement and validation programs |

The mathematical formulation and rejection-terminal principle build on established assignment literature. The paper is an implementation study, not a claim that rectangular or optional assignment is new.
