# Native Rectangular Linear Assignment with Implicit Rejection

Manuscript and reproducibility artifact for *Native Rectangular Linear Assignment with Implicit Rejection: A Jonker–Volgenant Solver without Square Expansion*.

**[Read the paper](output/arxiv/native_rectangular_assignment.pdf)** · [arXiv upload ZIP](output/arxiv/native_rectangular_assignment_arxiv.zip) · [HTML preview](output/arxiv/html/index.html) · [Markdown source](manuscript.md)

The arXiv version is the sole current edition. Its generated LaTeX sources are in
`latex/`; the PDF, upload ZIP, HTML preview, and validation reports are in
`output/arxiv/`. See [the build and submission guide](ARXIV.md) for requirements,
verification, and arXiv settings.

To rebuild from the current manuscript and saved vector figures, run
`python build_pdf.py` with TeX Live 2025 on PATH. To check HTML conversion, run
`python check_arxiv_html.py` with LaTeXML and Poppler available.

The paper describes one solver, implemented three ways, and compares it with SciPy and three established Jonker–Volgenant families. It is an implementation and engineering study; it does not claim that rectangular assignment or implicit unmatched choices are new.

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
python validate_release.py        # package test suites and native unit tests (see below)
python run_experiments.py --validate-batch-only  # correctness only; no timing campaign
python rerun_measurements.py      # main, formulation and scaling suites, run sequentially into tmp/rerun
python report_results.py          # tables, dataset summary, Figures 1-3
python plot_scaling.py            # Figures 4-6
python update_manuscript_tables.py
python build_pdf.py --refresh-figures  # arXiv LaTeX, vector figures, PDF and source ZIP
python check_artifact.py          # consistency of data, tables, narrative values and binaries
```

`validate_release.py` also runs the Rust unit tests and lapcpp's native C++ tests, so it expects source checkouts of [lapx](https://github.com/rathaROG/lapx), [lapcpp](https://github.com/rathaROG/lapcpp) and [laprust](https://github.com/rathaROG/laprust) next to this repository, in directories with those names. The other steps need only the installed packages.

`rerun_measurements.py` writes to a new directory so earlier results are never overwritten. Its `main`, `optional` and `scaling` outputs are promoted to `results/`, `results/formulations/` and `results/scaling/`, and its manifest, logs and CPU-load samples to `results/rerun/`. The validation step runs the LAPX, lapcpp and laprust test suites from a neutral working directory so they import the installed wheels, not the source checkouts.

The run in `results/` used wheels built from the release sources with each package's default settings (MSVC 14.44, Rust 1.99.1, Python 3.14.9, NumPy 2.5.4, SciPy 1.18.1) on an Intel Core i9-13900KS under Windows. The LAPX wheel reports its version string as `0.11.0rc1`, which `results/runtime.json` records; its solver code is that of the 0.11.0 release.

### Batch-validation correction

The batch validator now requires exactly one output per input and checks every output with strict iteration. Run `python run_experiments.py --validate-batch-only` to repeat the three recorded batch workloads, using their original seed and the one- and four-worker methods. It writes only `results/batch_validation.json`, including per-method checked-output counts, input hashes, and the current script and native-binary hashes. Run `python -m pytest -q tests/test_batch_validation.py` for the harness regression tests.

The retained rechecks cover all 11,040 outputs in each of the Python 3.14.9 and Python 3.11.16 environments, using the same native binaries and Python/NumPy/SciPy versions as their respective timing runs. The timing samples, summaries, and original environment records are unchanged. Their exact experiment script is preserved byte for byte in `archive/measurement-scripts/run_experiments.py`; `check_artifact.py` verifies it against the original recorded hash and verifies the corrected script against the separate recheck. The preserved source documents the historical measurements and is not the entry point for new runs.

The Python 3.11 recheck is saved separately in `archive/python-3.11.16/batch-revalidation/batch_validation.json`. To repeat it, use the Python 3.11 environment with `run_experiments.py --validate-batch-only --output archive/python-3.11.16/batch-revalidation`. Both rechecks are verified by `check_artifact.py`; all original logs remain untouched.

## Files

| File or directory | Purpose |
| --- | --- |
| manuscript.md | Full manuscript |
| references.bib | BibTeX entries corresponding to the manuscript references |
| record_runtime.py | Record installed versions, wheel provenance, native-binary hashes and source state |
| validate_release.py | Run the packages' test suites and native unit tests against the installed wheels |
| run_experiments.py | Enumeration validation, single-matrix, batch and memory experiments; batch-only correctness recheck |
| compare_optional.py | JVRUST and JVCPP against square-expanded, dummy-column and capped-cost SciPy |
| benchmark_scaling.py | Paired size sweeps: square, 4:1 wide, fixed 64 rows, and congested finite-limit costs |
| rerun_measurements.py | Run the three timing suites sequentially and record per-CPU load |
| report_results.py | Generated tables, dataset summary and Figures 1–3 (PNG/SVG) |
| plot_scaling.py | Figures 4–6 from the scaling results |
| update_manuscript_tables.py | Refresh the numeric result tables from the archived summaries |
| narrative.py | Every number quoted in the text, derived from the results, with checks of the claims made about them |
| build_pdf.py | Build and verify the arXiv PDF and source ZIP; optionally refresh vector figures |
| check_arxiv_html.py | Validate the upload ZIP with LaTeXML and generate a local HTML preview |
| templates/arxiv-preamble.tex | The single PDFLaTeX/LaTeXML-compatible paper template |
| ARXIV.md | Build requirements, validation workflow, output locations, and upload settings |
| check_artifact.py | Check samples, summaries, tables, narrative values, script and binary hashes, without native execution |
| results/ | Raw samples, summaries, environment, runtime and validation records, including the separate strict batch recheck |
| archive/measurement-scripts/ | Exact historical experiment source retained for measurement provenance |
| tests/test_batch_validation.py | Regression checks for incomplete/extra outputs, incorrect objectives, and validation-only execution |
| figures/ | Six main figures and one appendix figure as PNG and editable SVG; vector PDFs are in latex/images/ |
| archive/python-3.11.16/ | The earlier Python 3.11.16 run, summarized in Appendix A: results and logs, and the manuscript and figures built from them |
| latex/ | Canonical arXiv source project: main.tex, main.bbl, references.bib, and seven PDF figures |
| output/arxiv/ | Current paper PDF, upload ZIP, HTML preview, and build/HTML validation reports |

## Main measured findings

- All implementations agree exactly with enumeration on 16,800 calls, including 80 correctly reported infeasible problems. The formulation comparison adds 10,520 enumeration calls with zero error.
- The installed packages pass 2,704 LAPX, 1,271 lapcpp and 1,271 laprust Python tests, plus 23 and 20 Rust unit tests and lapcpp's native C++ tests.
- An implementation of the proposed solver is fastest in six of seven required cases and all five finite-limit cases. SciPy is 6.3 times faster on congested required blocks.
- Against capped-cost SciPy, JVRUST is faster in nine of ten finite-limit cases (2.6–11.9 times). Capped SciPy is 2.4 times faster on the congested case and wins from N=128 upward in the congested scaling sweep.
- For 1024 problems of size 8 × 10, the JVRUST batch call is 2.5 times faster than a Python loop with one worker and 5.2 times faster with four.
- JVRUST is faster than JVCPP in all twelve single-matrix cases under MSVC: by 4–16% on most required cases, 1.2–1.8 times on finite-limit cases, and 3.2 times on a tall input. The scaling sweeps do not preserve that ordering at every point; at the largest size of each uniform family the two are within 8%. The C++ completion kernel (JVHYBRID) gives no consistent benefit.
