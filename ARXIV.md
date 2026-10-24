# Build and submit the paper

The arXiv edition is the only current paper version. Edit `manuscript.md`,
`references.bib`, or `templates/arxiv-preamble.tex`, then rebuild. Files in
`latex/` are generated; direct edits there will be replaced by the next build.
The measurement records and historical Python 3.11 snapshot under `archive/`
are retained as reproducibility evidence for the paper and its appendix.

## Outputs

| Location | Purpose |
| --- | --- |
| [Paper PDF](output/arxiv/native_rectangular_assignment.pdf) | Current paper for reading and review |
| [Upload ZIP](output/arxiv/native_rectangular_assignment_arxiv.zip) | The single file to upload to arXiv |
| [LaTeX source](latex/main.tex) | Canonical source project, with bibliography and PDF figures in the same directory |
| [HTML preview](output/arxiv/html/index.html) | Local LaTeXML conversion for review |
| [PDF build report](output/arxiv/build_report.json) | Source/PDF/ZIP hashes, compiler settings, diagnostics, and commands |
| [HTML report](output/arxiv/html_report.json) | ZIP/HTML hashes, structural checks, diagnostics, and commands |

## Build

Use Python and TeX Live 2025 with `pdflatex` and `bibtex` on PATH:

```sh
python build_pdf.py
```

The default build uses the saved vector figures in `latex/images/` and needs
only Python's standard library and TeX Live. If TeX Live is not on PATH, pass
`--tex-bin` with its binary directory.

After changing measurements or plot code, install `requirements.txt` and refresh
the figures as part of the build:

```sh
python build_pdf.py --refresh-figures
```

`python build_pdf.py --sources-only` refreshes the figures and generated LaTeX
without compiling or packaging; it does not update the PDF, ZIP, or validation
reports. Run the full build afterward to publish consistent outputs. No build
command reruns solver benchmarks.

The builder converts the Markdown using the single arXiv template, generates
`main.bbl`, and packages an explicit allowlist: `main.tex`, `main.bbl`,
`references.bib`, and seven referenced PDF figures. It then extracts the ZIP into
a fresh directory and runs three PDFLaTeX passes using the included `.bbl`.
Shell escape is disabled and repository-specific TeX search paths are cleared.
Missing citations, undefined commands, missing glyphs, overfull boxes, and final
LaTeX warnings fail the build. Successful builds refresh the canonical `latex/`
project and the outputs together.

The template uses `article`, `newtxtext`/`newtxmath`, standard title/author/date
fields, and image alt text. It avoids the OpenType font commands that leaked
into the earlier HTML preview. The fixed paper date is preserved. BibTeX's five
existing missing-year warnings for software/documentation entries are recorded
in the build report; publication years are not invented to suppress them.

## Check HTML

With LaTeXML and Poppler's `pdftoppm` on PATH, run this after each full build:

```sh
python check_arxiv_html.py
```

The check extracts the actual upload ZIP, converts its unmodified TeX with
LaTeXML, and generates the preview. Poppler renders the original PDF figures to
PNG for local HTML display; these PNGs are not added to the upload. The original
converter XML and build logs remain in the temporary directory recorded in the
report. The preview uses LaTeXML's stylesheet plus small-screen image/table
sizing rules; arXiv applies its own website styling.

Validation covers standard front matter, seven figures with alt text, ten
tables, one algorithm, 19 references, all displayed formulas, resolved internal
links, and absence of conversion warnings or leaked font settings. The current
PDF has 26 pages. LaTeXML 0.8.8 parses 216 math objects, of which 207 are displayed;
the others are duplicate captions or linear alternatives to aligned equations.
Reports identify the ZIP they checked by SHA-256, so an HTML report for an older
ZIP does not establish that a new build has passed.

This checkout retains a portable HTML-check runtime under the ignored
`tmp/arxiv/tools/` directory. To use it in PowerShell:

```powershell
$perlRoot = (Resolve-Path tmp/arxiv/tools/strawberry).Path
$env:PATH = "$perlRoot/perl/bin;$perlRoot/c/bin;" + $env:PATH
python check_arxiv_html.py --perl tmp/arxiv/tools/strawberry/perl/bin/perl.exe --latexml-bin tmp/arxiv/tools/latexml/bin --perl-lib tmp/arxiv/tools/latexml/lib/perl5
```

## Upload

Upload only `output/arxiv/native_rectangular_assignment_arxiv.zip`. Select
**main.tex**, **PDFLaTeX**, and **TeX Live 2025** in arXiv's file-processing
interface. When replacing an earlier attempt, replace its sources and change
any saved XeLaTeX setting to PDFLaTeX. The interface generates `00README.json`;
no hand-written copy is needed.

Review arXiv's generated PDF and HTML before final submission. Do not upload the
compiled paper PDF, HTML preview, build reports, scripts, or the full repository
alongside the ZIP. Local conversion checks cannot reproduce every detail of
arXiv's deployed converter and stylesheet.

Guidance: [TeX submission requirements](https://info.arxiv.org/help/submit_tex.html),
[HTML best practices](https://info.arxiv.org/help/submit_latex_best_practices.html),
[supported TeX Live versions and processors](https://info.arxiv.org/help/faq/texlive.html),
and [submission-generated compiler settings](https://info.arxiv.org/help/00README.html).
