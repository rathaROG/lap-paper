# Build and submit the arXiv edition

Edit `manuscript.md`, `references.bib`, or `templates/arxiv-preamble.tex`, then rebuild. The paper covers the Python 3.14.9 and 3.11.17 collections from 24 October 2026. Generated files in `latex/` are replaced by a successful build. Measurements, source snapshots, collection charts, and the original comparison remain frozen.

## Outputs

| Output | Purpose |
| --- | --- |
| [Paper PDF](output/arxiv/native_rectangular_assignment.pdf) | Reading and review |
| [Upload ZIP](output/arxiv/native_rectangular_assignment_arxiv.zip) | File to upload to arXiv |
| [LaTeX project](latex/main.tex) | Main source, bibliography, and vector figures |
| [HTML preview](output/arxiv/html/index.html) | Local conversion of the actual upload sources |
| [Build report](output/arxiv/build_report.json) | Current page count, hashes, and compiler diagnostics |
| [HTML report](output/arxiv/html_report.json) | Matching ZIP hash, structural checks, and diagnostics |
| [Evidence report](output/arxiv/evidence_report.json) | Both collections, numeric tables, and preservation checks |
| [Visual review](output/arxiv/visual_review.json) | PDF visual review and HTML verification scope |

## Rebuild from saved data

Use TeX Live 2025 with `pdflatex` and `bibtex` on PATH. The default build needs Python's standard library and the saved vector figures:

```powershell
python build_pdf.py
```

To prepare all publication figures first, use either prepared Python environment with NumPy and Matplotlib:

```powershell
python build_pdf.py --refresh-figures
```

This copies six unchanged base figures and regenerates seven extension layouts and two comparison figures with labels sized for the paper. Publication-specific charts go to `figures/paper/`; collection charts stay unchanged. No solver measurements run. `--sources-only` prepares figures and LaTeX without updating the PDF, ZIP, or reports; complete a full build afterward. Use `--tex-bin` if needed.

The upload allowlist contains `main.tex`, `main.bbl`, `references.bib`, and 15 referenced PDF figures. The builder extracts the candidate ZIP into a fresh directory and runs three PDFLaTeX passes using the included `.bbl`, without BibTeX, shell escape, or repository-specific TeX search paths. Undefined references or commands, missing glyphs, overfull boxes, and final LaTeX warnings fail the build. Five existing BibTeX missing-year warnings for software/web references are recorded; no years are invented.

The standard `article` template uses `newtxtext`/`newtxmath`, title/author/date fields, structural sectioning, and image alt text. It avoids the OpenType font commands that leaked into the earlier HTML. The fixed revision date is 24 October 2026; `\today` is not used.

## Verify HTML and evidence

With LaTeXML and Poppler's `pdftoppm` available:

```powershell
python check_arxiv_html.py
python check_artifact.py
```

This checkout retains a portable HTML runtime under ignored `tmp/arxiv/tools/`. In PowerShell:

```powershell
$perlRoot = (Resolve-Path tmp/arxiv/tools/strawberry).Path
$env:PATH = "$perlRoot/perl/bin;$perlRoot/c/bin;" + $env:PATH
python check_arxiv_html.py --perl tmp/arxiv/tools/strawberry/perl/bin/perl.exe --latexml-bin tmp/arxiv/tools/latexml/bin --perl-lib tmp/arxiv/tools/latexml/lib/perl5
python check_artifact.py
```

The checker converts unmodified TeX and bibliography extracted from the ZIP. It renders source PDF figures to PNG for local display; those PNGs are not uploaded. It verifies front matter, all 15 figures with alt text, eleven tables, one algorithm, nineteen references, all visible formulas, internal links, converter diagnostics, and absence of leaked font settings. Counts derive from current sources.

The publication audit independently recalculates both collections' summaries and comparison, checks manuscript tables and quantitative findings, verifies unchanged evidence, and matches PDF/HTML reports to the current ZIP. Historical collection audits remain evidence of their completed runs and then-current paper-preservation checks. They are not rewritten after editorial changes.

After substantive edits, render and inspect every PDF page, then inspect HTML at desktop and narrow widths. Intermediates belong under `tmp/`; final report hashes identify the reviewed artifacts. Opening a preview alone is not validation.

## Upload

Upload only `output/arxiv/native_rectangular_assignment_arxiv.zip`. Select **main.tex**, **PDFLaTeX**, and **TeX Live 2025**. Replace any saved XeLaTeX selection from the earlier attempt. Let arXiv's interface generate its compiler configuration; do not add a handwritten `00README.json`.

Review arXiv's generated PDF and HTML before final submission. Local checks cannot reproduce every detail of its deployed converter and styling. Do not upload the compiled paper PDF, local HTML, logs, reports, scripts, unused figures, or full repository alongside the source ZIP.

Guidance checked for this revision: [TeX submission requirements](https://info.arxiv.org/help/submit_tex.html), [HTML markup best practices](https://info.arxiv.org/help/submit_latex_best_practices.html), [TeX Live processors](https://info.arxiv.org/help/faq/texlive.html), and [compiler configuration](https://info.arxiv.org/help/00README.html).
