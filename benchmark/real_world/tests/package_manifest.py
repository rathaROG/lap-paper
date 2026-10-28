"""Check or refresh provenance/package_manifest.json (size and SHA-256 of every package file).

    python tests/package_manifest.py           # check; exit 1 when stale
    python tests/package_manifest.py --write   # regenerate after editing package files
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE / "provenance" / "package_manifest.json"
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def current_files():
    files = {}
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path == MANIFEST or SKIP_PARTS & set(path.relative_to(PACKAGE).parts):
            continue
        data = path.read_bytes()
        files[path.relative_to(PACKAGE).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    files = current_files()
    if args.write:
        previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {}
        document = {
            "scope": "Every file of this standalone package except this manifest: portable scripts, tests, licences, "
                     "re-frozen inputs, bundled replay matrices and provenance records.",
            "files": files, "bytes": sum(entry["bytes"] for entry in files.values()), "file_count": len(files),
            "verification": previous.get("verification", {}),
        }
        MANIFEST.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {MANIFEST.relative_to(PACKAGE)}: {len(files)} files")
        return 0
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    stale = sorted(name for name in set(files) | set(recorded) if files.get(name) != recorded.get(name))
    for name in stale:
        print("stale:", name)
    print("package manifest is current" if not stale else f"{len(stale)} stale entries; run with --write")
    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
