"""Small integration smoke in a relocated copy of this package, without a full rerun.

Copies only this directory to a temporary location and runs from an unrelated
working directory, proving that no sibling repository layout or archive is
needed: standard-library input verification, cached assignment replay from the
bundled matrices (optional dependencies blocked), and, when pyppbox is
available, a two-frame Centroid+SORT tracker smoke plus a report from it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PACKAGE = Path(__file__).resolve().parents[1]

GUARD = """import sys,runpy,importlib.abc,json
blocked=set(json.loads(sys.argv[1]));entry=sys.argv[2];args=sys.argv[3:]
class Denied(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in blocked:
   raise ModuleNotFoundError('Deliberately unavailable optional dependency: '+fullname,name=fullname)
sys.meta_path.insert(0,Denied())
sys.path.insert(0,str(__import__('pathlib').Path(entry).parent))
sys.argv=[entry]+args
runpy.run_path(entry,run_name='__main__')
assert not (blocked & set(sys.modules))
"""


def smoke(python, pyppbox_root, output):
    results = []
    with tempfile.TemporaryDirectory(prefix="lap-realworld-portability-") as temporary:
        scratch = Path(temporary).resolve()
        assert scratch.is_relative_to(Path(tempfile.gettempdir()).resolve())
        copied = scratch / "relocated" / "real_world"
        shutil.copytree(PACKAGE, copied, ignore=shutil.ignore_patterns("__pycache__"))
        unrelated = scratch / "unrelated-working-directory"
        unrelated.mkdir()

        def run(label, args, blocked=(), stdlib_only=False):
            entry = copied / "run.py"
            if blocked:
                command = [str(python), "-B", "-c", GUARD, json.dumps(list(blocked)), str(entry), *args]
            else:
                command = [str(python), "-B", *(["-S"] if stdlib_only else []), str(entry), *args]
            completed = subprocess.run(command, cwd=unrelated, capture_output=True, text=True, timeout=180)
            print(f"{label}: exit {completed.returncode}", flush=True)
            if completed.returncode:
                raise RuntimeError(completed.stdout + "\n" + completed.stderr)
            results.append({"check": label, "passed": True, "entry_point": "relocated/real_world/run.py",
                            "working_directory": "unrelated-working-directory",
                            "arguments": [arg.replace(str(scratch), "<temporary-copy>") for arg in args],
                            "blocked_imports": list(blocked), "exit_code": completed.returncode})

        run("standard-library input verification", ["--verify-inputs"], stdlib_only=True)
        replay = scratch / "replay-output"
        run("bundled-matrix replay without detector/pyppbox imports",
            ["--replay-only", "--output", str(replay), "--solvers", "scipy", "lapjvh", "laprust",
             "--cases", "mot16_04_gt_centroid", "--precisions", "float32", "--rounds", "1",
             "--block-seconds", "0.001", "--limit-matrices", "2"], blocked=("torch", "rfdetr", "cv2", "pyppbox"))
        replay_summary = json.loads((replay / "summary.json").read_text(encoding="utf-8"))
        assert len(replay_summary["replay"]) == 3
        assert all(row["objective_mismatch_count"] == 0 for row in replay_summary["replay"])
        # The tracker stage needs requirements-tracking.txt (pyppbox, motmetrics, filterpy, yaml, cv2).
        missing = [name for name in ("motmetrics", "filterpy", "yaml", "cv2")
                   if importlib.util.find_spec(name) is None]
        if not pyppbox_root and importlib.util.find_spec("pyppbox") is None:
            missing.insert(0, "pyppbox")
        tracker_ran = not missing
        if tracker_ran:
            tracker = scratch / "tracker-output"
            args = ["--tracker-only", "--output", str(tracker), "--solvers", "scipy", "lapjvh", "laprust",
                    "--sources", "gt", "--sequences", "mot16_04", "--frames", "2", "--rounds", "1",
                    "--warmup-frames", "1"]
            if pyppbox_root:
                args += ["--pyppbox-root", str(pyppbox_root)]
            run("two-frame actual Centroid+SORT, SciPy/H2/Rust", args, blocked=("torch", "rfdetr"))
            summary = json.loads((tracker / "summary.json").read_text(encoding="utf-8"))
            assert len(summary["tracking"]) == 6
            report = scratch / "report-output"
            run("saved-summary report from relocated code",
                ["--report-only", "--collection", str(tracker), "--output", str(report)],
                blocked=("torch", "rfdetr", "cv2", "pyppbox"))
            assert (report / "report.html").is_file()
        else:
            print(f"tracker/report smoke skipped; missing: {', '.join(missing)} (see requirements-tracking.txt)",
                  flush=True)
        proof = {"scope": "Small wiring/integrity smokes; local short timings are not publication measurements",
                 "passed": True, "relocated_copy_only": True, "tracker_smoke": tracker_ran, "checks": results}
        output.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--pyppbox-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "lap-realworld-portability.json")
    options = parser.parse_args()
    smoke(options.python, options.pyppbox_root, options.output)
