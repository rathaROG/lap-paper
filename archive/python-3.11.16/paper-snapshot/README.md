# Native Rectangular Linear Assignment with Implicit Rejection

Manuscript and reproducibility artifact for *Native Rectangular Linear Assignment with Implicit Rejection: A Jonker–Volgenant Solver without Square Expansion*.

The manuscript is [manuscript.md](manuscript.md); the typeset PDF is `output/pdf/native_rectangular_assignment.pdf`. It describes one solver, implemented three ways, and compares it with SciPy and three established Jonker–Volgenant families. It is an implementation and engineering study; it does not claim that rectangular assignment or implicit unmatched choices are new.

## Solver labels

| Label | Implementation | Package and function |
| --- | --- | --- |
| JVRUST | Rust throughout | laprust 1.0.0 `lapjv`; LAPX 0.11.0 `lapjvr` (identical native code) |
| JVHYBRID | Rust, with C++ square completion | LAPX 0.11.0 `lapjvh` |
| JVCPP | C++ throughout | lapcpp 1.0.0 `lapjv` |
| JVX | gatagat/lap-derived baseline | LAPX `lapjv` / `lapjvx` / `lapjvxa` (timed: `lapjvxa`) |
| JVS | src-d/lapjv-derived baseline | LAPX `lapjvs` / `lapjvsa` (timed: `lapjvsa`, float64) |
| JVC | py-lapsolver-derived baseline | LAPX `lapjvc` |
| SciPy | Native rectangular baseline | `scipy.optimize.linear_sum_assignment` |

The experiments time JVRUST through laprust. Enumeration validation covers it through both laprust and LAPX.

## Reproduction

Install the three packages and the Python requirements into one environment, then run from this folder:

```sh
python -m pip install lapx==0.11.0 lapcpp==1.0.0 laprust==1.0.0 -r requirements.txt
python record_runtime.py          # results/runtime.json: versions, wheel and binary hashes
python validate_release.py        # package test suites and native unit tests (needs the sibling checkouts)
python rerun_measurements.py      # main, formulation and scaling suites, run sequentially into tmp/rerun
python report_results.py          # tables, dataset summary, Figures 1-3
python plot_scaling.py            # Figures 4-6
python update_manuscript_tables.py
python build_pdf.py               # LaTeX sources, vector figures, PDF (local TeX Live)
python check_artifact.py          # consistency of data, tables, narrative values and binaries
```

`rerun_measurements.py` writes to a new directory so earlier results are never overwritten. Its `main`, `optional` and `scaling` outputs are promoted to `results/`, `results/formulations/` and `results/scaling/`, and its manifest, logs and CPU-load samples to `results/rerun/`. The validation step runs the LAPX, lapcpp and laprust test suites from a neutral working directory so they import the installed wheels, not the source checkouts.

The archived run used wheels built from the release sources with each package's default settings (MSVC 14.44, Rust 1.99.1, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1) on an Intel Core i9-13900KS under Windows. The LAPX wheel was built before its version string was changed from 0.11.0rc1 to 0.11.0, so `results/runtime.json` records `0.11.0rc1`; the solver code is that of the release.

## Files

| File or directory | Purpose |
| --- | --- |
| manuscript.md | Full manuscript |
| references.bib | BibTeX entries corresponding to the manuscript references |
| record_runtime.py | Record installed versions, wheel provenance, native-binary hashes and source state |
| validate_release.py | Run the packages' test suites and native unit tests against the installed wheels |
| run_experiments.py | Enumeration validation, single-matrix, batch and memory experiments |
| compare_optional.py | JVRUST and JVCPP against square-expanded, dummy-column and capped-cost SciPy |
| benchmark_scaling.py | Paired size sweeps: square, 4:1 wide, fixed 64 rows, and congested finite-limit costs |
| rerun_measurements.py | Run the three timing suites sequentially and record per-CPU load |
| report_results.py | Generated tables, dataset summary and Figures 1–3 (PNG/SVG) |
| plot_scaling.py | Figures 4–6 from the scaling results |
| update_manuscript_tables.py | Refresh the numeric result tables from the archived summaries |
| narrative.py | Every number quoted in the text, derived from the results, with checks of the claims made about them |
| build_pdf.py | Convert the manuscript to LaTeX, draw vector figures and typeset the PDF |
| check_artifact.py | Check samples, summaries, tables, narrative values, script and binary hashes, without native execution |
| results/ | Raw samples, summaries, environment, runtime and validation records |
| figures/ | Six figures as PNG and editable SVG; vector PDFs are in latex/images/ |
| latex/ | Generated LaTeX project and bibliography |

## Main measured findings

- All implementations agree exactly with enumeration on 16,800 calls, including 80 correctly reported infeasible problems. The formulation comparison adds 10,520 enumeration calls with zero error.
- The installed packages pass 2,704 LAPX, 1,271 lapcpp and 1,271 laprust Python tests, plus 23 and 20 Rust unit tests and lapcpp's native C++ tests.
- An implementation of the proposed solver is fastest in six of seven required cases and all five finite-limit cases. SciPy is 6.0 times faster on congested required blocks.
- Against capped-cost SciPy, JVRUST is faster in nine of ten finite-limit cases (2.7–12.8 times). Capped SciPy is 2.3 times faster on the congested case and wins from N=128 upward in the congested scaling sweep.
- For 1024 problems of size 8 × 10, the JVRUST batch call is 2.2 times faster than a Python loop with one worker and 4.7 times faster with four.
- The complete Rust implementation is faster than the complete C++ implementation under MSVC: by 4–12% on most required cases, 1.1–1.7 times on finite-limit cases with forbidden entries, and 3.1 times on a tall input. The C++ completion kernel (JVHYBRID) gives no consistent benefit.
