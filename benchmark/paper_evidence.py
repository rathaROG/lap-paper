"""Package or restore the frozen paper evidence; never executes a benchmark.

Reader: python benchmark/paper_evidence.py extract
Author: python benchmark/paper_evidence.py pack [--inputs PATH]
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "benchmark/evidence"
MANIFEST = DESTINATION / "manifest.json"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024*1024): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def pack(input_file):
    source = json.loads(input_file.read_text(encoding="utf-8-sig"))
    required = set(source if isinstance(source, list) else (row["path"] for row in source["files"]))
    assert required and all(path.startswith("benchmark/runs/") for path in required)
    selected = set(required)
    # Complete primary raw timing/validation logs make the central estimates
    # independently auditable, beyond the smaller set read by the plotter.
    roots = sorted({PurePosixPath(path).parts[2] for path in required})
    for run in roots:
        if run == "linux-py3.11.18-20261030-1434": continue
        selected.update(path.relative_to(ROOT).as_posix() for path in
                        (ROOT / "benchmark/runs" / run / "paper-run-light/campaign-session/attempts").glob("*/measurements.jsonl"))
    groups = defaultdict(list)
    files = []
    for relative in sorted(selected):
        path = ROOT / relative
        if not path.is_file(): raise FileNotFoundError(path)
        run = PurePosixPath(relative).parts[2]
        groups[run].append(relative)
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest(path),
                          archive=run + ".zip", role="generator_input" if relative in required else "primary_raw_audit"))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    archives = []
    for run, paths in sorted(groups.items()):
        target = DESTINATION / (run + ".zip")
        temporary = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:
            for relative in sorted(paths):
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 9
                with (ROOT / relative).open("rb") as source_stream, archive.open(info, "w", force_zip64=True) as target_stream:
                    shutil.copyfileobj(source_stream, target_stream, length=1024*1024)
        temporary.replace(target)
        archives.append(dict(path=target.relative_to(ROOT).as_posix(), bytes=target.stat().st_size,
                             sha256=digest(target), files=len(paths)))
    manifest = dict(schema_version=1,
        scope="Every file read by the saved-data generator, plus complete primary campaign measurement/validation logs; no executable binaries or unused source snapshots.",
        selection="generator_input_files.json from an audited complete generation; additional raw primary attempt logs",
        roots=roots, archives=archives, files=files)
    write_json(MANIFEST, manifest)
    print("Packaged", len(files), "files in", len(archives), "archives;", sum(row["bytes"] for row in archives), "compressed bytes.")


def restore(extract):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    indexed = {row["path"]: row for row in manifest["files"]}
    assert len(indexed) == len(manifest["files"])
    restored = existing = 0
    for descriptor in manifest["archives"]:
        archive_path = ROOT / descriptor["path"]
        if digest(archive_path) != descriptor["sha256"]: raise ValueError(f"Archive checksum mismatch: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            expected = {row["path"] for row in indexed.values() if row["archive"] == archive_path.name}
            if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected):
                raise ValueError(f"Unexpected or duplicated archive members: {archive_path}")
            for relative in sorted(expected):
                parts = PurePosixPath(relative).parts
                if len(parts) < 4 or parts[:2] != ("benchmark", "runs") or ".." in parts:
                    raise ValueError(f"Unsafe evidence path: {relative}")
                target = (ROOT / relative).resolve()
                if not target.is_relative_to((ROOT / "benchmark/runs").resolve()):
                    raise ValueError(f"Evidence path escapes runs/: {relative}")
                record = indexed[relative]
                if target.exists():
                    if not target.is_file() or digest(target) != record["sha256"]:
                        raise ValueError(f"Existing evidence differs; left unchanged: {relative}")
                    existing += 1
                    continue
                if not extract: continue
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".extracting")
                with archive.open(relative) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024*1024)
                if temporary.stat().st_size != record["bytes"] or digest(temporary) != record["sha256"]:
                    temporary.unlink()
                    raise ValueError(f"Extracted evidence checksum mismatch: {relative}")
                temporary.replace(target)
                restored += 1
    print("Verified archive checksums;", existing, "matching existing files;", restored, "restored files.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "extract", "verify"))
    parser.add_argument("--inputs", type=Path, default=ROOT / "benchmark/paper-artifacts/generator_input_files.json")
    args = parser.parse_args()
    if args.command == "pack": pack(args.inputs)
    else: restore(args.command == "extract")


if __name__ == "__main__": main()
