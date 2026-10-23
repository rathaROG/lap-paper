"""Typeset the Markdown manuscript using local TeX Live and vector figures.

The converter supports the controlled Markdown used by this paper. It changes
presentation, not the scientific text or archived measurements.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER / "latex"
OUTPUT = PAPER / "output" / "pdf"
BUILD = PAPER / "tmp" / "pdfs" / "build"
KEYS = ("zhang2022bytetrack", "gatagatlap", "jonker1987", "crouse2016", "bougleux2016",
        "jones2017", "bougleux2020", "bijsterbosch2010", "volgenant1996",
        "crouse2017tcl", "mathworks2018assign", "srcdlapjv", "laprust2026", "lapx2026",
        "lapcpp2026", "pylapsolver", "scipylsa", "vanderwiel1997", "lsape2017")
# Figures typeset where they appear, so the float does not split a sentence across pages.
PINNED = {"1"}
PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm,headheight=14pt]{geometry}
\usepackage{fontspec,amsmath,amssymb}
\usepackage{unicode-math}
\setmainfont{texgyretermes-regular.otf}[ItalicFont=texgyretermes-italic.otf,
 BoldFont=texgyretermes-bold.otf,BoldItalicFont=texgyretermes-bolditalic.otf]
\setsansfont{texgyreheros-regular.otf}[ItalicFont=texgyreheros-italic.otf,
 BoldFont=texgyreheros-bold.otf,BoldItalicFont=texgyreheros-bolditalic.otf]
\setmonofont{texgyrecursor-regular.otf}[Scale=MatchLowercase,
 ItalicFont=texgyrecursor-italic.otf,BoldFont=texgyrecursor-bold.otf,
 BoldItalicFont=texgyrecursor-bolditalic.otf]
\setmathfont{texgyretermes-math.otf}
\usepackage{microtype}
\usepackage{graphicx,booktabs,tabularx,array,xcolor}
\usepackage{caption,float,placeins,listings,titlesec}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xurl}
\usepackage[unicode,colorlinks=true,linkcolor=black,citecolor=black,
urlcolor=blue!45!black]{hyperref}
\hypersetup{
 pdftitle={Native Rectangular Linear Assignment with Implicit Rejection},
 pdfauthor={Ratha SIV},
 pdfsubject={Implementation and experimental study},
 pdfkeywords={linear assignment, Jonker-Volgenant, rectangular matching, implicit rejection}
}
\urlstyle{same}
\Urlmuskip=0mu plus 2mu
\setlength{\emergencystretch}{3em}
\setlength{\parindent}{1.2em}
\setlength{\parskip}{2pt}
\linespread{1.025}
\clubpenalty=10000
\widowpenalty=10000
\raggedbottom
\captionsetup{font=small,labelfont=bf,justification=justified,singlelinecheck=false}
\captionsetup[table]{position=top,skip=7pt}
\setlength{\textfloatsep}{15pt plus 3pt minus 3pt}
\setlength{\floatsep}{13pt plus 3pt minus 3pt}
\renewcommand{\topfraction}{.92}
\renewcommand{\bottomfraction}{.85}
\renewcommand{\textfraction}{.06}
\renewcommand{\floatpagefraction}{.75}
\titlespacing*{\section}{0pt}{2.5ex plus 1ex minus .2ex}{1ex}
\titlespacing*{\subsection}{0pt}{2ex plus .5ex minus .2ex}{.7ex}
\newfloat{algorithm}{tbp}{loa}
\floatname{algorithm}{Algorithm}
\lstset{basicstyle=\ttfamily\footnotesize,breaklines=true,columns=fullflexible,
 keepspaces=true,frame=single,framerule=.3pt,rulecolor=\color{black!30},
 xleftmargin=5pt,xrightmargin=5pt,aboveskip=4pt,belowskip=4pt}
\begin{document}
\begin{center}
{\LARGE\bfseries Native Rectangular Linear Assignment\\[3pt]
with Implicit Rejection\par}
\vspace{6pt}
{\large A Jonker--Volgenant Solver without Square Expansion\par}
\vspace{10pt}
{\large Ratha SIV\par}
\vspace{5pt}
{\small 22 October 2026\par}
\end{center}
\vspace{3pt}
"""


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


def convert():
    lines = (PAPER / "manuscript.md").read_text(encoding="utf-8").splitlines()
    out = [PREAMBLE]
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
            out += [r"\FloatBarrier", r"\renewcommand{\bibfont}{\small}", r"\setlength{\bibsep}{3pt plus 1pt}",
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
            out += [
                r"\begin{figure}" + ("[H]" if number[0].isalpha() or number in PINNED else "[!htbp]"), r"\centering",
                r"\includegraphics[width=\linewidth,height=.77\textheight,keepaspectratio]{images/" + image + ".pdf}",
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
    PROJECT.mkdir(exist_ok=True)
    (PROJECT / "main.tex").write_text("\n".join(out), encoding="utf-8")
    shutil.copy2(PAPER / "references.bib", PROJECT / "references.bib")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    images = PROJECT / "images"
    images.mkdir(exist_ok=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex-bin", type=Path, default=Path(r"C:\dev\exc\texlive\2025\bin\windows"))
    parser.add_argument("--sources-only", action="store_true")
    options = parser.parse_args()
    convert()
    figures()
    from plot_scaling import generate as scaling_figures, generate_runs
    scaling_figures()
    generate_runs()
    if options.sources_only:
        print(PROJECT / "main.tex")
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = str(options.tex_bin) + os.pathsep + env["PATH"]
    env["BIBINPUTS"] = str(PROJECT) + os.pathsep + env.get("BIBINPUTS", "")
    engine = options.tex_bin / ("lualatex.exe" if os.name == "nt" else "lualatex")
    bibtex = options.tex_bin / ("bibtex.exe" if os.name == "nt" else "bibtex")
    tex_command = [str(engine), "-interaction=nonstopmode", "-halt-on-error",
                   "-file-line-error", "-output-directory=" + str(BUILD), "main.tex"]
    records = []
    commands = [tex_command, [str(bibtex), "main"], tex_command, tex_command]
    for index, command in enumerate(commands, 1):
        result = subprocess.run(command, cwd=BUILD if index == 2 else PROJECT,
                                env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, errors="replace")
        (BUILD / f"pass_{index}.log").write_text(result.stdout, encoding="utf-8")
        records.append({"command": command, "exit_code": result.returncode})
        if result.returncode:
            print(result.stdout[-14000:])
            raise SystemExit(result.returncode)
        print(f"Typesetting pass {index}/{len(commands)} completed.", flush=True)
    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    issues = [line for line in log.splitlines() if any(term in line for term in (
        "Overfull", "Missing character", "undefined", "Rerun to get", "LaTeX Warning"))]
    target = OUTPUT / "native_rectangular_assignment.pdf"
    shutil.copy2(BUILD / "main.pdf", target)
    metadata = {
        "engine": subprocess.check_output([str(engine), "--version"], text=True).splitlines()[0],
        "manuscript_sha256": hashlib.sha256((PAPER / "manuscript.md").read_bytes()).hexdigest(),
        "tex_sha256": hashlib.sha256((PROJECT / "main.tex").read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "typesetting_passes": records,
        "log_diagnostics": issues,
        "scientific_measurements": "All six figures use the measured summaries in results/.",
    }
    (PAPER / "results" / "pdf_build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Output:", target)
    print("Diagnostics:", json.dumps(issues, ensure_ascii=True))


if __name__ == "__main__":
    main()
