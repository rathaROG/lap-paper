"""Verify the actual builder retained with an existing LSAPE build."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_builder(build_dir, root, record):
    """Return the recorded builder and any archive evidence to retain.

    The archived builder identifies how this binary was produced; it need not
    match today's command-line plumbing. It is never imported or executed.
    Compiled inputs and the live binary are checked separately by the caller.
    """
    build_dir, root = Path(build_dir).resolve(), Path(root).resolve()
    if record.get("returncode") != 0:
        raise RuntimeError("Build provenance does not describe a successful build")
    current = root / "build_bridge.py"
    expected = record.get("input_sha256", {}).get("build_bridge.py")
    if _digest(current) == expected:
        return current, ()
    snapshot = build_dir / "snapshot"
    archived = snapshot / "sources" / "build_bridge.py"
    archive_record = snapshot / "provenance.json"
    archive_binary = snapshot / "_lsape_native.pyd"
    try:
        if (json.loads(archive_record.read_text(encoding="utf-8")) != record
                or _digest(archived) != expected
                or _digest(archive_binary) != record.get("output_sha256")
                or _digest(build_dir / "_lsape_native.pyd") != record.get("output_sha256")):
            raise RuntimeError("Archived LSAPE builder is not bound to the successful build snapshot")
    except (OSError, ValueError) as error:
        raise RuntimeError("Cannot verify the recorded LSAPE builder snapshot") from error
    return archived, (archive_record, archive_binary)
