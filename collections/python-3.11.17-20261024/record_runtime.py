"""Record the installed packages that the experiments measure.

The paper evaluates the released sources of LAPX 0.11.0, lapcpp 1.0.0 and laprust
1.0.0, installed as ordinary wheels. This script records their versions, wheel and
native-binary hashes, and (when the sibling checkouts exist) the state of each
source tree, so every timed suite can verify that it measured the same binaries.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import run_experiments as base

PAPER = Path(__file__).resolve().parent
PACKAGES = {"lapx": "lap", "lapcpp": "lapcpp", "laprust": "laprust"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_state(checkout):
    if not (checkout / ".git").exists():
        return None
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()
    return {"head": git("rev-parse", "HEAD"), "clean": git("status", "--porcelain") == ""}


def main():
    packages = {}
    for distribution, module_name in PACKAGES.items():
        module = importlib.import_module(module_name)
        direct = metadata.distribution(distribution).read_text("direct_url.json")
        packages[distribution] = {
            "version": metadata.version(distribution),
            "path": str(Path(module.__file__).resolve().parent),
            "installed_from": json.loads(direct) if direct else None,
            "source": source_state(PAPER.parents[2] / distribution),
        }
    native = {}
    for name in base.NATIVE_MODULES:
        path = Path(importlib.import_module(name).__file__).resolve()
        native[name] = {"path": str(path), "sha256": digest(path)}
    report = {
        "python": sys.version, "platform": platform.platform(),
        "packages": packages, "native_modules": native,
        "record_script_sha256": digest(Path(__file__)),
    }
    output = PAPER / "results"
    output.mkdir(exist_ok=True)
    (output / "runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: info["version"] for name, info in packages.items()}))


if __name__ == "__main__":
    main()
