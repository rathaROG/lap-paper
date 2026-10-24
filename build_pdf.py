"""Build the paper's arXiv PDF, source ZIP, and independently verified LaTeX project.

Use --refresh-figures after changing measurements or plot code. The default build
uses the existing vector figures and requires only Python's standard library and
TeX Live. --sources-only refreshes figures and LaTeX without compiling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER / "latex"
KEYS = ("zhang2022bytetrack", "gatagatlap", "jonker1987", "crouse2016", "bougleux2016",
        "jones2017", "bougleux2020", "bijsterbosch2010", "volgenant1996",
        "crouse2017tcl", "mathworks2018assign", "srcdlapjv", "laprust2026", "lapx2026",
        "lapcpp2026", "pylapsolver", "scipylsa", "vanderwiel1997", "lsape2017")


def escaped(text):
    mapping = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
        "$": r"\$", "&": r"\&", "#": r"\#", "_": r"\_",
        "%": r"\%", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}",
        "×": r"\ensuremath{\times}", "≤": r"\ensuremath{\leq}",
        "≥": r"\ensuremath{\geq}", "∞": r"\ensuremath{\infty}",
        "²": r"\textsuperscript{2}", "∎": r"\hfill\ensuremath{\blacksquare}",
        "–": "--", "—": "---", "“": chr(96) * 2, "”": "''", "’": "'",
    }
    return "".join(mapping.get(char, char) for char in text)


def inline(text):
    tick = chr(96)
    pattern = re.compile(
        r"\$([^$]+)\$|"
        r"\[([^\]]+)\]\(([^)]+)\)|"
        r"\*\*(.+?)\*\*|\*([^*]+)\*|"
        + re.escape(tick) + r"([^" + re.escape(tick) + r"]+)" + re.escape(tick) + "|"
        r"\[([1-9][0-9]*(?:[,\s–-]+[1-9][0-9]*)*)\]|"
        r"(?<!\w)(O\([^)]*\))|"
        r"(?<!\w)(p=2 round_T\(L/2\))|"
        r"(?<!\w)([a-f0-9]{40})(?!\w)|"
        r"(?<!\w)([A-Za-z]_[A-Za-z0-9]+(?:[+\-=≤≥](?:[A-Za-z](?:_[A-Za-z0-9]+)?|[-+]?\d+(?:\.\d+)?))*|[A-Za-z]=(?:[-+]?\d+(?:\.\d+)?|[A-Za-z]+))|"
        r"(?<!\w)([a-z]+_[a-z_]+)(?!\w)"
    )
    out, start = [], 0
    for match in pattern.finditer(text):
        out.append(escaped(text[start:match.start()]))
        explicit_math, label, url, bold, italic, code, citation, order, penalty, sha, math, identifier = match.groups()
        if explicit_math is not None:
            out.append(r"\(" + explicit_math + r"\)")
        elif url is not None:
            out.append(r"\href{" + url + "}{" + inline(label) + "}")
        elif bold is not None:
            out.append(r"\textbf{" + inline(bold) + "}")
        elif italic is not None:
            out.append(r"\emph{" + inline(italic) + "}")
        elif code is not None:
            out.append(r"\texttt{" + escaped(code) + "}")
        elif citation is not None:
            indices = []
            for part in citation.replace("–", "-").split(","):
                values = [int(value) for value in part.strip().split("-")]
                indices.extend(range(values[0], values[-1] + 1))
            out.append(r"\cite{" + ",".join(KEYS[index - 1] for index in indices) + "}")
        elif order is not None:
            out.append(r"\(\mathcal{O}(" + order[2:-1].replace("²", "^2") + r")\)")
        elif penalty is not None:
            out.append(r"\(p=2\operatorname{round}_T(L/2)\)")
        elif sha is not None:
            out.append(r"\nolinkurl{" + sha + "}")
        elif math is not None:
            math = re.sub(r"_([A-Za-z0-9]+)", r"_{\1}", math)
            out.append(r"\(" + math.replace("≤", r"\leq ").replace("≥", r"\geq ") + r"\)")
        elif identifier is not None:
            out.append(r"\texttt{" + escaped(identifier) + "}")
        start = match.end()
    out.append(escaped(text[start:]))
    return "".join(out)


def caption_after(lines, index, kind):
    while index < len(lines) and not lines[index].strip():
        index += 1
    match = re.fullmatch(r"\*\*" + kind + r" ([A-Z]?\d+)\.\*\* (.+)", lines[index])
    if not match:
        raise ValueError(f"Missing {kind} caption at line {index + 1}")
    return match.group(1), inline(match.group(2)), index + 1


def convert(destination=PROJECT):
    lines = (PAPER / "manuscript.md").read_text(encoding="utf-8").splitlines()
    out = [(PAPER / "templates" / "arxiv-preamble.tex").read_text(encoding="utf-8")]
    i = lines.index("## Abstract") + 1
    abstract = []
    while not lines[i].startswith("**Keywords:"):
        abstract.append(lines[i])
        i += 1
    out += [r"\begin{abstract}", inline(" ".join(abstract).strip()), r"\end{abstract}"]
    out += [r"\noindent{\small " + inline(lines[i]) + "}", ""]
    i += 1
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line == "## References":
            # BibTeX typesets the list; any appendix follows the bibliography.
            out += [r"\FloatBarrier", r"\renewcommand{\bibfont}{\small}", r"\setlength{\bibsep}{1pt plus 1pt}",
                    r"\bibliographystyle{unsrtnat}", r"\bibliography{references}", ""]
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        if line.startswith("## Appendix"):
            out += [r"\FloatBarrier", r"\clearpage", r"\appendix", r"\setcounter{table}{0}",
                    r"\renewcommand{\thetable}{\Alph{section}\arabic{table}}",
                    r"\setcounter{figure}{0}", r"\renewcommand{\thefigure}{\Alph{section}\arabic{figure}}"]
            title = re.sub(r"^Appendix [A-Z]:\s*", "", line[3:])
            out += [r"\section{" + inline(title) + "}", ""]
            i += 1
        elif line.startswith("## "):
            out.append(r"\FloatBarrier")
            title = re.sub(r"^\d+\s+", "", line[3:])
            command = r"\section" if re.match(r"^\d+\s+", line[3:]) else r"\section*"
            out += [command + "{" + inline(title) + "}", ""]
            i += 1
        elif line.startswith("### "):
            title = re.sub(r"^(?:\d+|[A-Z])(?:\.\d+)*\s+", "", line[4:])
            out += [r"\subsection{" + inline(title) + "}", ""]
            i += 1
        elif line == "$$":
            math = []
            i += 1
            while lines[i] != "$$":
                math.append(lines[i])
                i += 1
            out += [r"\[", *math, r"\]", ""]
            i += 1
        elif line.startswith("| "):
            table = []
            while i < len(lines) and lines[i].startswith("| "):
                table.append([cell.strip() for cell in lines[i].strip("|").split("|")])
                i += 1
            number, caption, i = caption_after(lines, i, "Table")
            columns = len(table[0])
            # Batch tables have long numeric headers: wrap them in fixed right-aligned columns.
            numeric = ">{\\raggedleft\\arraybackslash}p{.2\\linewidth}"
            layouts = {"1": "@{}p{.18\\linewidth}XX@{}",
                       "2": "@{}p{.13\\linewidth}p{.13\\linewidth}p{.15\\linewidth}X@{}",
                       "4": "@{}X" + numeric * 3 + "@{}", "A2": "@{}X" + numeric * 3 + "@{}"}
            layout = layouts.get(number, "@{}X" + "r" * (columns - 1) + "@{}")
            size = r"\footnotesize" if number in ("2", "3", "6", "A1", "A3") else r"\small"
            # Appendix tables stay with their headings; main-text tables float.
            placement = "[H]" if number[0].isalpha() else "[!htbp]"
            out += [
                r"\begin{table}" + placement, r"\centering", r"\caption{" + caption + "}",
                r"\label{tab:" + number + "}", r"\begingroup" + size,
                r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.13}",
                r"\begin{tabularx}{\linewidth}{" + layout + "}", r"\toprule",
                " & ".join(r"\textbf{" + inline(cell) + "}" for cell in table[0]) + r" \\",
                r"\midrule",
            ]
            for row in table[2:]:
                out.append(" & ".join(inline(cell) for cell in row) + r" \\")
            out += [r"\bottomrule", r"\end{tabularx}", r"\endgroup", r"\end{table}", ""]
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            image = Path(match.group(2)).stem
            number, caption, i = caption_after(lines, i + 1, "Figure")
            alt = "alt={" + escaped(match.group(1)) + "},"
            out += [
                r"\begin{figure}" + ("[H]" if number[0].isalpha() else "[!htbp]"), r"\centering",
                r"\includegraphics[" + alt + r"width=\linewidth,height=.77\textheight,keepaspectratio]{images/" + image + ".pdf}",
                r"\caption{" + caption + "}", r"\label{fig:" + number + "}",
                r"\end{figure}", "",
            ]
        elif line.startswith("- "):
            out.append(r"\begin{itemize}\setlength{\itemsep}{2pt}")
            while i < len(lines) and lines[i].startswith("- "):
                out.append(r"\item " + inline(lines[i][2:]))
                i += 1
            out += [r"\end{itemize}", ""]
        elif line.startswith("~~~"):
            code = []
            i += 1
            while lines[i] != "~~~":
                code.append(lines[i])
                i += 1
            number, caption, i = caption_after(lines, i + 1, "Algorithm")
            out += [
                r"\begin{algorithm}[!htbp]", r"\caption{" + caption + "}",
                r"\label{alg:" + number + "}", r"\begin{lstlisting}", *code,
                r"\end{lstlisting}", r"\end{algorithm}", "",
            ]
        else:
            paragraph = []
            while i < len(lines) and lines[i].strip():
                paragraph.append(lines[i])
                i += 1
            out += [inline(" ".join(paragraph)), ""]
    out += [r"\FloatBarrier", r"\end{document}", ""]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "main.tex").write_text("\n".join(out), encoding="utf-8")
    shutil.copy2(PAPER / "references.bib", destination / "references.bib")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    images = PROJECT / "images"
    images.mkdir(parents=True, exist_ok=True)
    with (PAPER / "results" / "single_summary.csv").open(newline="") as stream:
        single = list(csv.DictReader(stream))
    with (PAPER / "results" / "batch_summary.csv").open(newline="") as stream:
        batch = list(csv.DictReader(stream))
    memory = json.loads((PAPER / "results" / "memory.json").read_text())
    from report_results import NAMES as solvers, LABELS as labels, COLORS as colors, LABEL, COLOR, \
        batch_label, batch_order
    import run_experiments
    cases = (
        "square128", "square512", "wide64x512", "tall512x64", "near256x320",
        "blocks256x320", "ties128x160", "optional128x160", "optional256x320",
        "tight128x160", "negative128x160", "optional_square128",
    )
    names = (
        "Square\n128 × 128", "Square\n512 × 512", "Wide\n64 × 512",
        "Tall\n512 × 64", "Uniform\n256 × 320", "Blocks\n256 × 320",
        "Ties\n128 × 160", "128 × 160\nL = 0.5", "256 × 320\nL = 0.5",
        "128 × 160\nL = 0.05", "Signed 128 × 160\nL = -0.25",
        "Square 128\nL = 0.5",
    )
    by_case = {case: {row["solver"]: row for row in single if row["case"] == case} for case in cases}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.9))
    for ax, indices, title in zip(axes, (range(7), range(7, 12)),
                                ("Required assignment", "Finite rejection limit")):
        indices = list(indices)
        x = np.arange(len(indices))
        for k, (solver, label, color) in enumerate(zip(solvers, labels, colors)):
            rows = [by_case[cases[index]][solver] for index in indices]
            values = np.array([float(row["median_ns"]) / 1e6 for row in rows])
            low = np.array([float(row["q25_ns"]) / 1e6 for row in rows])
            high = np.array([float(row["q75_ns"]) / 1e6 for row in rows])
            positions = x + (k - 3) * .115
            ax.bar(positions, values, width=.11, color=color, label=label)
            ax.errorbar(positions, values, yerr=[values-low, high-values],
                        fmt="none", ecolor="#333333", linewidth=.6, capsize=1)
        ax.set_xticks(x, [names[index] for index in indices], fontsize=7.3)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("Median time per matrix (ms)")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 1.36), frameon=False, fontsize=8)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(images / "single_runtime.pdf", bbox_inches="tight")
    plt.close(fig)

    titles = {"batch64x64x80": "Required: 64 problems, 64 × 80",
              "batch16x128x160_optional": "Finite limit: 16 problems, 128 × 160, L = 0.5",
              "batch1024x8x10": "Required: 1024 problems, 8 × 10"}
    batch_cases = [spec["case"] for spec in run_experiments.batch_specs()]
    fig, axes = plt.subplots(len(batch_cases), 1, figsize=(6.5, 2.5 * len(batch_cases)))
    for ax, case in zip(axes, batch_cases):
        title = titles[case]
        rows = {row["solver"]: row for row in batch if row["case"] == case}
        group = [rows[name] for name in batch_order(rows)]
        ticks = [batch_label(row["solver"]).replace(", ", "\n") for row in group]
        values = [float(row["median_ns"]) / 1e6 for row in group]
        bar_colors = [COLOR[row["solver"].split("_")[0]] for row in group]
        low = [float(row["median_ns"]) - float(row["q25_ns"]) for row in group]
        high = [float(row["q75_ns"]) - float(row["median_ns"]) for row in group]
        ax.bar(range(len(group)), values, color=bar_colors)
        ax.errorbar(range(len(group)), values, yerr=np.array([low, high]) / 1e6,
                    fmt="none", color="#333333", linewidth=.7, capsize=2)
        ax.set_xticks(range(len(group)), ticks, fontsize=6.2)
        ax.set_ylabel("Median time per whole batch (ms)")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(images / "batch_runtime.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    methods = run_experiments.MEMORY_SOLVERS
    for k, case in enumerate(("required", "optional")):
        group = {row["solver"]: row for row in memory if row["case"] == case}
        values = [group[method]["additional_peak"] / 2**20 for method in methods]
        positions = np.arange(len(methods)) + (k - .5) * .34
        bars = ax.bar(positions, values, width=.32, color=("#176ca4", "#d17c26")[k],
                      label=("required", "finite limit L = 0.5")[k])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(methods)), [LABEL[method] for method in methods])
    ax.set_yscale("log")
    ax.set_ylim(.07, 500)
    ax.set_ylabel("Additional peak working set (MiB)")
    ax.set_title("256 × 4096 float64; 8 MiB input already resident", fontsize=10, loc="left")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(images / "memory_peak.pdf", bbox_inches="tight")
    plt.close(fig)
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(log):
    return [line for line in log.splitlines() if any(term in line for term in (
        "Overfull", "Missing character", "undefined", "Rerun to get",
        "LaTeX Warning", "Package natbib Warning",
        "Package rerunfilecheck Warning",
    ))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-bin", type=Path,
                        help="TeX Live binary directory; defaults to PATH lookup")
    parser.add_argument("--refresh-figures", action="store_true",
                        help="Regenerate vector figures from the recorded measurements before building")
    parser.add_argument("--sources-only", action="store_true",
                        help="Regenerate figures and LaTeX without compiling or packaging")
    options = parser.parse_args()
    if options.refresh_figures or options.sources_only:
        figures()
        from plot_scaling import generate, generate_runs
        generate()
        generate_runs()
    if options.sources_only:
        convert()
        print(PROJECT / "main.tex")
        return
    env = os.environ.copy()
    if options.tex_bin:
        env["PATH"] = str(options.tex_bin.resolve()) + os.pathsep + env["PATH"]
    # Do not allow the standalone check to find missing sources in the repo.
    for name in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFOUTPUT"):
        env.pop(name, None)
    engine = shutil.which("pdflatex", path=env["PATH"])
    bibtex = shutil.which("bibtex", path=env["PATH"])
    if not engine or not bibtex:
        parser.error("PDFLaTeX and BibTeX are required (use --tex-bin if needed).")

    scratch = PAPER / "tmp" / "arxiv"
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="build-", dir=scratch))
    source = work / "source"
    convert(source)
    tex = (source / "main.tex").read_text(encoding="utf-8")
    # Catch the commands responsible for leaked font settings in the first
    # arXiv HTML preview, even if someone later changes the source template.
    if re.search(r"fontspec|unicode-math|\\set(?:main|sans|mono|math)font|\\today|\\pdfoutput", tex):
        raise RuntimeError("The arXiv source contains unsupported font or output settings.")
    for command in ("title", "author", "date", "maketitle"):
        if not re.search(r"\\" + command + r"\b", tex):
            raise RuntimeError(f"Missing standard front matter command: {command}")
    members = ["main.tex", "main.bbl", "references.bib"]
    images = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))
    if not images:
        raise RuntimeError("The generated manuscript contains no figures.")
    for name in images:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".pdf":
            raise RuntimeError(f"Expected a relative PDF figure path: {name}")
        original = PROJECT
        for part in relative.parts:
            # Windows is case insensitive, while arXiv's filesystem is not.
            if part not in {entry.name for entry in original.iterdir()}:
                raise RuntimeError(f"Missing figure or filename case mismatch: {name}")
            original = original / part
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        members.append(name)
    # arXiv recommends generating 00README.json in its submission interface.
    # Select main.tex, PDFLaTeX and TeX Live 2025 there instead of uploading it.
    commands = []

    def run(command, directory, label):
        result = subprocess.run(command, cwd=directory, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                errors="replace")
        (work / f"{label}.log").write_text(result.stdout, encoding="utf-8")
        commands.append({"step": label, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {work / (label + '.log')}\n"
                               + result.stdout[-6000:])
        print(f"{label} completed.", flush=True)

    tex_command = [engine, "-no-shell-escape", "-interaction=nonstopmode",
                   "-halt-on-error", "-file-line-error", "-recorder", "main.tex"]
    run(tex_command, source, "source-pass-1")
    run([bibtex, "main"], source, "bibliography")
    run(tex_command, source, "source-pass-2")
    run(tex_command, source, "source-pass-3")
    bib_log = (source / "main.blg").read_text(encoding="utf-8", errors="replace")
    bib_warnings = [line for line in bib_log.splitlines() if line.startswith("Warning--")]
    # Existing software/web references omit publication years. Preserve those
    # entries and record the style's warnings; never invent dates to silence it.
    if any(not line.startswith("Warning--empty year in ") for line in bib_warnings):
        raise RuntimeError(f"BibTeX warnings; see {source / 'main.blg'}")

    # Package an explicit allowlist, with the main source at the archive root.
    candidate = work / "native_rectangular_assignment_arxiv.zip"
    with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (source / name).read_bytes())

    # Compile only what the user will upload, in an empty directory, with no
    # BibTeX invocation and no access through TeX's input path to this repository.
    verified = work / "extracted"
    with zipfile.ZipFile(candidate) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed.")
        archive.extractall(verified)
    for index in range(1, 4):
        run(tex_command, verified, f"archive-pass-{index}")
    issues = diagnostics((verified / "main.log").read_text(encoding="utf-8", errors="replace"))
    if issues:
        raise RuntimeError("Final PDFLaTeX diagnostics:\n" + "\n".join(issues))
    page_matches = re.findall(r"Output written on .*?\((\d+) pages?",
                              (work / "archive-pass-3.log").read_text(encoding="utf-8"))
    if not page_matches:
        raise RuntimeError("Could not verify the compiled page count.")
    bbl = (verified / "main.bbl").read_text(encoding="utf-8")
    cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex)
             for key in group.split(",")}
    bibliography = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", bbl))
    if not cited.issubset(bibliography):
        raise RuntimeError(f"Missing bibliography entries: {sorted(cited - bibliography)}")

    output = PAPER / "output" / "arxiv"
    pdf = output / "native_rectangular_assignment.pdf"
    output.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    upload = output / candidate.name
    shutil.copy2(candidate, upload)
    shutil.copy2(verified / "main.pdf", pdf)
    # Publish one canonical LaTeX project, containing the exact upload sources.
    PROJECT.mkdir(parents=True, exist_ok=True)
    for name in ("main.tex", "main.bbl", "references.bib"):
        shutil.copy2(verified / name, PROJECT / name)
    report = {
        "compiler": "pdflatex", "texlive_version": 2025,
        "compiler_settings": "Select main.tex, PDFLaTeX and TeX Live 2025 in arXiv's submission interface.",
        "engine": subprocess.check_output([engine, "--version"], env=env, text=True).splitlines()[0],
        "manuscript_sha256": sha256(PAPER / "manuscript.md"),
        "source_project": PROJECT.relative_to(PAPER).as_posix(),
        "archive": upload.relative_to(PAPER).as_posix(),
        "archive_sha256": sha256(upload),
        "archive_bytes": upload.stat().st_size,
        "pdf": pdf.relative_to(PAPER).as_posix(), "pdf_sha256": sha256(pdf),
        "pages": int(page_matches[-1]), "figures": len(images),
        "references": len(bibliography),
        "files": {name: sha256(verified / name) for name in sorted(members)},
        "verified_from_extracted_archive": True,
        "verification_uses_bibtex": False, "shell_escape": False,
        "html_validation": "Not run by this script. See html_report.json when separately verified with LaTeXML.",
        "log_diagnostics": issues, "bibliography_warnings": bib_warnings,
        "commands": commands,
        "build_directory": str(work.relative_to(PAPER)),
        "guidance": ["https://info.arxiv.org/help/submit_tex.html",
                     "https://info.arxiv.org/help/faq/texlive.html",
                     "https://info.arxiv.org/help/00README.html",
                     "https://info.arxiv.org/help/submit_latex_best_practices.html"],
    }
    (output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Upload: {upload}\nPreview: {pdf}\n"
          f"Verified: {report['pages']} pages, {len(images)} figures, "
          f"{len(bibliography)} references, no final diagnostics.")


if __name__ == "__main__":
    main()
