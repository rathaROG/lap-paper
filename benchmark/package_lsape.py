"""Package a complete fresh LSAPE build, verification and three-session study.

Reads immutable observations and independently audits their saved arithmetic.
No solver, compiler, or measurement driver is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import audit_lsape_update


HERE = Path(__file__).resolve().parent
SUMMARY_FILES = ("summary.json", "all_cases.csv", "all_case_methods.csv", "paper_table.csv", "block_counts.csv")


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepare(source, projections, destination):
    source, projections, destination = map(Path, (source, projections, destination))
    if destination.exists():
        raise FileExistsError("Refusing to replace an existing LSAPE archive")
    audit = audit_lsape_update.audit(source)
    directories = ("experiment", "verification", "session-1", "session-2", "session-3", "summary")
    files = {}
    for directory in directories:
        for path in (source / directory).rglob("*"):
            # Compiler intermediates can embed private build directories. The
            # final module, build provenance and source snapshots are retained.
            if (path.is_file() and path.suffix.lower() not in {".obj", ".exp"}
                    and not ({"__pycache__", ".pytest_cache"} & set(path.parts))):
                files["lsape/" + path.relative_to(source).as_posix()] = path
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".json", ".log", ".py"):
            files["lsape/campaign/" + path.name] = path
    for name in SUMMARY_FILES:
        files["lsape/" + name] = source / "summary" / name
    if projections.is_file():
        files["lsape/projections.json"] = projections
    reports = [read(source / f"session-{index}/report.json") for index in (1, 2, 3)]
    with TemporaryDirectory(prefix="lap-paper-lsape-package-") as temporary:
        staging = Path(temporary)
        rounds = {"manifest": reports[0]["manifest"], "sessions": [
            {"session_index": report["session_index"], "cases": [
                {"case": case["case"], "round_average_us": {
                    method: stats["round_average_us"] for method, stats in case["methods"].items()}}
                for case in report["cases"]]} for report in reports]}
        for name, value in (("timing_rounds.json", rounds), ("independent_audit.json", audit)):
            path = staging / name
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            files["lsape/" + name] = path
        manifest = {"scope": "Fresh Windows LSAPE controlled study; three sessions and strict verification",
                    "files": {name: identity(path) for name, path in sorted(files.items())}}
        path = staging / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        files["manifest.json"] = path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x") as archive:
            for name, path in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    return {"archive": str(destination), **identity(destination), "members": len(files), "audit": audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--projections", type=Path, default=HERE / "publication-lsape-20261029-projections.json")
    parser.add_argument("--output", type=Path, default=HERE / "lsape-windows-20261029.zip")
    options = parser.parse_args()
    print(json.dumps(prepare(options.results, options.projections, options.output), indent=2))
