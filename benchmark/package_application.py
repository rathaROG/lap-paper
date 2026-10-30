"""Package completed cached tracking/replay observations without measuring again.

The original campaign directory is read-only. Every recorded source-file hash
is checked before packaging, and the resulting nested archive has a manifest
compatible with verify_results.py. No solver or tracker is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile


HERE = Path(__file__).resolve().parent


def identity(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalized(name):
    # The campaign's original Windows file manifest stores backslashes.
    name = name.replace("\\", "/")
    parts = name.split("/")
    if not parts or any(not part or part in (".", "..") or ":" in part for part in parts):
        raise ValueError("Unsafe source manifest path")
    return name


def add_file(archive, source, name, *, stored=False):
    info = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, source.read_bytes())


def prepare(observations, projections, destination, audits=()):
    observations, projections, destination = map(Path, (observations, projections, destination))
    if destination.exists():
        raise FileExistsError("Refusing to replace an existing application archive")
    progress = read(observations / "progress.json")
    if progress.get("phase") != "complete":
        raise ValueError("Application campaign is not complete")
    summary = read(observations / "summary.json")
    if len(summary["tracking"]) != 72 or len(summary["replay"]) != 144:
        raise ValueError("Expected the complete eight-pipeline, nine-method campaign")
    source_manifest = read(observations / "file_manifest.json")
    sources = {}
    for name, expected in source_manifest.items():
        name = normalized(name)
        if name.casefold() in {n.casefold() for n in sources}:
            raise ValueError("Duplicate source manifest path")
        source = observations.joinpath(*name.split("/"))
        if not source.resolve().is_relative_to(observations.resolve()):
            raise ValueError("Source manifest escapes the campaign directory")
        if identity(source) != expected:
            raise ValueError("Source campaign checksum differs: " + name)
        sources[name] = source
    actual = {p.relative_to(observations).as_posix() for p in observations.rglob("*")
              if p.is_file() and p.name != "file_manifest.json"}
    # The portable wrapper writes this final launch record after the campaign's
    # own file_manifest.json. Keep it as an explicitly declared extra member.
    extras = actual - set(sources)
    if extras - {"portable_run.json"} or set(sources) - actual:
        raise ValueError("Source campaign inventory differs from its recorded manifest")
    for name in extras:
        sources[name] = observations / name
    projected = read(projections)
    if set(projected["tables"]) != {"12", "C1", "C2", "C3", "C4"}:
        raise ValueError("Application projection has incomplete paper-table coverage")

    with TemporaryDirectory(prefix="lap-paper-application-package-") as temporary:
        staging = Path(temporary)
        packed = staging / "observations.zip"
        members = {}
        with zipfile.ZipFile(packed, "x") as archive:
            for name, path in sorted(sources.items()):
                add_file(archive, path, name)
                members[name] = identity(path)
            path = observations / "file_manifest.json"
            add_file(archive, path, "file_manifest.json")
            members["file_manifest.json"] = identity(path)
        collection = staging / "collection.json"
        collection.write_text(json.dumps({
            "scope": "Current cached tracking and common-cost replay; detector inference was not repeated",
            "python": summary["environment"]["python"],
            "packages": summary["environment"]["packages"],
            "started_utc": summary["started_utc"], "completed_utc": summary["completed_utc"],
            "source_file_manifest_sha256": members["file_manifest.json"]["sha256"],
        }, indent=2) + "\n", encoding="utf-8")
        files = {"real_world/observations.zip": packed,
                 "real_world/projections.json": projections,
                 "real_world/collection.json": collection}
        for path in map(Path, audits):
            name = "real_world/audits/" + normalized(path.name)
            if name in files:
                raise ValueError("Duplicate application audit filename")
            files[name] = path
        manifest = staging / "manifest.json"
        manifest.write_text(json.dumps({
            "scope": "Current application results for Tables 12 and C1-C4",
            "files": {name: identity(path) for name, path in files.items()},
            "archives": {"real_world/observations.zip": members},
        }, indent=2) + "\n", encoding="utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x") as archive:
            for name, path in files.items():
                add_file(archive, path, name, stored=name.endswith(".zip"))
            add_file(archive, manifest, "manifest.json")
    return {"archive": str(destination), **identity(destination), "observation_files": len(members)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--projections", type=Path,
                        default=HERE / "publication-tracking-20261029-projections.json")
    parser.add_argument("--output", type=Path, default=HERE / "application-windows-20261029.zip")
    parser.add_argument("--audit", type=Path, action="append", default=[],
                        help="Adjacent post-run audit to preserve (may be repeated)")
    options = parser.parse_args()
    print(json.dumps(prepare(options.observations, options.projections, options.output, options.audit), indent=2))


if __name__ == "__main__":
    main()
