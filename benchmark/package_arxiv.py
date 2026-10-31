"""Rebuild the main paper and package its actual local LaTeX dependencies.

Run after generate_paper_artifacts.py. This does not publish or upload anything.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
JOB = "native_rectangular_assignment"


def submission_files(root: Path) -> dict[str, Path]:
    """Use TeX's recorder, excluding system packages and temporary build files."""
    root = root.resolve()
    latex = root / "latex"
    recorder = root / "output/build" / f"{JOB}.fls"
    sources = {"main.tex": latex / "main.tex",
               "main.bbl": latex / "main.bbl",
               "references.bib": latex / "references.bib"}
    working = latex
    allowed = {".tex", ".bbl", ".bib", ".pdf", ".png", ".jpg", ".jpeg",
               ".eps", ".sty", ".cls", ".bst"}
    for line in recorder.read_text(encoding="utf-8").splitlines():
        if line.startswith("PWD "):
            working = Path(line[4:]).resolve()
        elif line.startswith("INPUT "):
            path = Path(line[6:])
            path = (working / path).resolve() if not path.is_absolute() else path.resolve()
            if path.is_relative_to(latex) and path.suffix.lower() in allowed:
                sources[path.relative_to(latex).as_posix()] = path
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(sorted(sources.items()))


def write_archive(target: Path, sources: dict[str, Path]) -> None:
    """Fixed entry metadata makes identical source inputs produce identical ZIPs."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(sources.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Submission ZIP failed its integrity check")
        for name, path in sources.items():
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"Submission source changed during packaging: {name}")
    temporary.replace(target)


def main() -> None:
    subprocess.run(["latexmk", "-pdf", "latex/main.tex"], cwd=ROOT, check=True)
    shutil.copyfile(ROOT / "output/build" / f"{JOB}.bbl", ROOT / "latex/main.bbl")
    sources = submission_files(ROOT)
    target = ROOT / "output/arxiv/lap-paper-arxiv.zip"
    write_archive(target, sources)
    print(f"Packaged and verified {len(sources)} source files in {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
