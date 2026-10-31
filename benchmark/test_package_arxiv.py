"""Submission packaging retains used sources without bundling unrelated files."""
import zipfile

from .package_arxiv import JOB, submission_files, write_archive


def test_recorder_selects_dependencies_and_archive_is_reproducible(tmp_path):
    latex = tmp_path / "latex"
    build = tmp_path / "output/build"
    build.mkdir(parents=True)
    (latex / "images").mkdir(parents=True)
    (latex / "generated").mkdir()
    expected = {"main.tex": b"paper", "main.bbl": b"bibliography",
                "references.bib": b"sources", "images/used.pdf": b"figure",
                "generated/table.tex": b"table"}
    for name, content in expected.items():
        (latex / name).write_bytes(content)
    (latex / "images/unused.pdf").write_bytes(b"not part of the paper")
    (latex / "main.aux").write_bytes(b"temporary")
    (build / f"{JOB}.fls").write_text(
        f"PWD {latex.as_posix()}\nINPUT main.tex\nINPUT ./images/used.pdf\n"
        f"INPUT {latex.as_posix()}/images/used.pdf\nINPUT generated/table.tex\n"
        "INPUT main.aux\nINPUT ../system-package.sty\n", encoding="utf-8")
    sources = submission_files(tmp_path)
    assert set(sources) == set(expected)
    archive = tmp_path / "submission.zip"
    write_archive(archive, sources)
    first = archive.read_bytes()
    write_archive(archive, sources)
    assert archive.read_bytes() == first
    with zipfile.ZipFile(archive) as packed:
        assert packed.namelist() == sorted(expected)
        assert {name: packed.read(name) for name in packed.namelist()} == expected
