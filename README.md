# Native Rectangular Linear Assignment with Implicit Rejection

Paper, benchmark programs, and saved measurements for *Native Rectangular Linear Assignment with Implicit Rejection: A Jonker–Volgenant Solver without Square Expansion*.

**[arXiv:2610.35948](https://arxiv.org/abs/2610.35948)** · **[Revised paper PDF](output/pdf/native_rectangular_assignment.pdf)** · [LaTeX source](latex/main.tex) · [arXiv source ZIP](output/arxiv/native_rectangular_assignment_arxiv.zip) · [HTML preview](output/arxiv/html/index.html)

The 29 October 2026 revision evaluates **LAPX 0.13.0, laprust 1.2.0, and lapcpp 1.2.0**. Main controlled results use Windows Python **3.14.10**; Appendix A gives Python **3.11.18**. Each campaign contains 204 cases, 8,946 timing samples, and 415 memory observations. All 3,925 compared numerical objectives agree exactly across the four Windows/Linux environments.

Appendix B reports fresh Linux Python **3.14.10** and **3.11.18** campaigns with the same current releases. Table B7 and the platform figures use these four current environments. The separate controlled LSAPE study was freshly rebuilt and rerun on Python 3.11.18, retaining its unchanged Hungarian algorithms and patch.

## Findings from the refreshed Windows controlled study

- Both native implementations beat ordinary SciPy in **71/71 required cases** in each current Windows environment, and expanded SciPy in five additional finite-limit cases.
- Against capped SciPy, JVRUST wins **120/125** cases on Python 3.11.18 and **119/125** on Python 3.14.10. JVCPP wins **124/125** and **121/125**.
- Remaining native/capped latency ratios reach about **1.16** for Rust and **1.06** for C++. These are workload-specific results, not universal superiority.
- Four-worker Rust batching reduces loop latency by **1.82–5.01 times**. For 1024 tiny problems, one-worker batching gives **2.22–2.37 times**. The four logical processors used for Windows batching span two physical cores.
- The revised initializer uses a common bounded reduction allowance, free-column tie preference, and a required-only displacement cutoff. Float32 costs retain their storage dtype while numerical working state uses float64.

The fresh Linux replication also beats ordinary SciPy in **71/71 required cases** for both native implementations. Against capped SciPy, Rust and C++ each win **123/125** cases in both Linux environments.

Timing IQRs are descriptive. The experiments do not isolate language, compiler, interpreter, individual policy components, or operating-system effects. The formulation and rejection-terminal principle build on established assignment literature; no algorithmic priority is claimed.

The refreshed cached-video study uses the same current solver releases and Python
3.11.18. Native assignment replay is **1.86–3.26 times** faster than SciPy;
Centroid tracker updates gain **1.04–1.15 times**, while SORT medians remain
near equal (**0.98–1.05 times**). All **150,780** backend/matrix checks have
zero objective gaps. Accuracy remains workload dependent.

The [current application bundle](benchmark/application-windows-20261029.zip)
contains fresh tracking/replay observations and independent audits. Frozen
detector inputs were reused; detector inference was not repeated.

The [fresh LSAPE bundle](benchmark/lsape-windows-20261029.zip) contains all
67 cases in three new process sessions, **9,045 timing-round averages**, and
strict validation of **10,190 cases**. Matched-layout specialization exceeds
the descriptive 3% gain threshold in **67/67** cases; the total padded adaptation
has **38 gains, 18 regressions, and 11 cases inside the band**. The patch already
uses float64 arithmetic and a guarded rejection cache, so no JV-specific update
was required. These classifications are descriptive, not significance tests.

## Reproduce and verify

Use Python 3.14.10 for the primary controlled environment:

```sh
pip install -r benchmark/requirements-py314.txt
python benchmark/run_all.py
```

For Python 3.11.18, use `benchmark/requirements-py311.txt`. The application and LSAPE studies have separate instructions in their benchmark folders.

To verify the saved measurements and manuscript numeric tables without running solvers or measurements:

```sh
python benchmark/verify_results.py
```

See [benchmark instructions](benchmark/README.md) and [saved-results provenance](benchmark/RESULTS.md). The current controlled evidence is preserved in [controlled-windows-20261029.zip](benchmark/controlled-windows-20261029.zip) and [controlled-linux-20261029.zip](benchmark/controlled-linux-20261029.zip). The default verifier requires these two archives plus the application and LSAPE archives; it does not use superseded measurements.

## Regenerate publication outputs

```sh
pip install -r benchmark/requirements-publication.txt
python benchmark/update_publication.py
python benchmark/linux_publication.py
python benchmark/apply_publication_tables.py benchmark/publication-windows-20261029-tables.json
python benchmark/apply_publication_tables.py benchmark/publication-linux-20261029-projections.json
python benchmark/build_publication.py
```

The last command compiles `latex/main.tex`, packages an explicit source allowlist, then verifies that the extracted archive compiles independently without BibTeX or shell escape. The LaTeX manuscript remains authoritative. The optional `benchmark/check_publication_html.py` requires LaTeXML and Poppler.

## Solver labels

| Label | Implementation / entry point |
| --- | --- |
| JVRUST | laprust 1.2.0 `lapjv`; the same native algorithm through LAPX 0.13.0 `lapjvr` |
| JVHYBRID | LAPX 0.13.0 `lapjvh`, Rust with C++ square completion |
| JVCPP | lapcpp 1.2.0 `lapjv`, C++ throughout |
| JVX | LAPX gatagat/lap-derived baseline; timed entry point `lapjvxa` |
| JVS | LAPX src-d/lapjv-derived baseline; `lapjvsa`, configured float64 |
| JVC | LAPX py-lapsolver-derived baseline; `lapjvc` |
| SciPy | `scipy.optimize.linear_sum_assignment`, with the formulation specified in the paper |

## Repository layout

| Location | Purpose |
| --- | --- |
| `latex/main.tex` | Authoritative editable manuscript |
| `latex/references.bib`, `latex/main.bbl` | Bibliography and supplied formatted bibliography |
| `latex/images/`, `figures/` | Included and exported figures |
| `output/pdf/` | Revised paper PDF |
| `output/arxiv/` | Source archive, independent build report, and HTML preview |
| `benchmark/` | Standalone experiments, verification, and publication generation |
| `benchmark/controlled-windows-20261029.zip` | Refreshed Windows observations and audits |
| `benchmark/application-windows-20261029.zip` | Refreshed cached tracking, replay, accuracy, and audits |
| `benchmark/lsape-windows-20261029.zip` | Fresh LSAPE build, validation, three sessions, and audits |
| `benchmark/controlled-linux-20261029.zip` | Refreshed Linux observations, logs, and audits |
