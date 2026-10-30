"""Package the two completed Linux controlled runs without remeasurement.

Every raw observation, suite log, CPU trace, supplied audit and terminal
transcript is retained byte for byte. Existing bundles are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile

import audit_controlled_update

HERE = Path(__file__).resolve().parent
SUITES = ("base", "formulations", "scaling", "extended")


def identity(path):
    data = Path(path).read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def prepare(py311, py314, transcript, supplied_audit, independent_review, output):
    py311, py314, transcript, supplied_audit, independent_review, output = map(
        Path, (py311, py314, transcript, supplied_audit, independent_review, output))
    if output.exists():
        raise FileExistsError("Refusing to replace an existing Linux evidence bundle")
    reports = [audit_controlled_update.audit(folder) for folder in (py311, py314)]
    if reports[0][1] != reports[1][1] or reports[0][2] != reports[1][2]:
        raise ValueError("Linux objectives/cardinalities or measurement schedules differ")
    files = {}
    for tag, folder in (("py311-results", py311), ("py314-results", py314)):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                files[f"{tag}/{path.relative_to(folder).as_posix()}"] = path
        for suite in SUITES:
            environment = json.loads((folder / suite / "environment.json").read_text(encoding="utf-8"))
            if environment["script_sha256"] != identity(HERE / suite / "run.py")["sha256"]:
                raise ValueError("Measured Linux source differs from source snapshot: " + suite)
    for suite in SUITES:
        for path in sorted((HERE / suite).rglob("*.py")):
            files["benchmark-source/" + path.relative_to(HERE).as_posix()] = path
    for name in ("run_all.py", "audit_controlled_update.py", "package_linux.py",
                 "requirements-py311.txt", "requirements-py314.txt"):
        files["benchmark-source/" + name] = HERE / name
    for name, path in (("terminal.md", transcript), ("supplied-linux-audit.json", supplied_audit),
                       ("independent-four-environment-review.json", independent_review)):
        if not path.is_file():
            raise FileNotFoundError(path)
        files["campaign/" + name] = path
    with TemporaryDirectory(prefix="lap-paper-linux-package-") as temporary:
        staging = Path(temporary)
        audit = {"status": "passed", "environments": []}
        for tag, result in zip(("py311-results", "py314-results"), reports):
            report = dict(result[0], result_directory=tag)
            audit["environments"].append(report)
        audit["cross_environment"] = {
            "numeric_objectives_compared": len(reports[0][1]),
            "exact_objective_and_recorded_cardinality_agreements": len(reports[0][1]),
            "same_schedules": True, "same_frozen_input_digests": True}
        provenance = {
            "measurement_date": "2026-10-29", "packages": audit_controlled_update.PACKAGES,
            "source_snapshot": "benchmark-source/", "terminal_transcript": "campaign/terminal.md",
            "scope": "Saved controlled observations; packaging executes no solver or measurement.",
            "affinity_scope": "Recorded single CPU 0 and batch CPUs 0,2,4,6. CPU traces describe load; no Linux core-type/topology mapping was recorded, and these indices do not establish matching physical cores across operating systems.",
            "memory_scope": "Linux additional getrusage(RUSAGE_SELF).ru_maxrss high-water RSS after an already-resident input; zero increment does not mean zero allocation.",
            "binary_scope": "Native module hashes and versions are retained in each suite environment.json; binary payloads are not required for saved-data verification."}
        for name, value in (("packaged-audit.json", audit), ("provenance.json", provenance)):
            path = staging / name
            path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
            files["campaign/" + name] = path
        manifest = {"scope": "Current Linux Python 3.11.18 and 3.14.10 controlled evidence",
                    "files": {name: identity(path) for name, path in sorted(files.items())}}
        path = staging / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        files["manifest.json"] = path
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x") as archive:
            for name, path in sorted(files.items()):
                member = zipfile.ZipInfo(name, date_time=(2026, 10, 29, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o100644 << 16
                archive.writestr(member, path.read_bytes())
    return {"archive": str(output), **identity(output), "members": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--py311", type=Path, required=True)
    parser.add_argument("--py314", type=Path, required=True)
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--supplied-audit", type=Path, required=True)
    parser.add_argument("--independent-review", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "controlled-linux-20261029.zip")
    args = parser.parse_args()
    print(json.dumps(prepare(args.py311, args.py314, args.transcript, args.supplied_audit,
                             args.independent_review, args.output), indent=2))
