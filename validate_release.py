"""Run the three packages' test suites against the installed releases.

Python suites run from a neutral working directory so they import the installed
wheels, not the sibling source checkouts. Rust unit tests run from the laprust and
LAPX sources; lapcpp's native C++ tests run when CMake is available.
"""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent


def main():
    out = PAPER / "results"
    runtime = json.loads((out / "runtime.json").read_text())
    neutral = Path(tempfile.mkdtemp(prefix="paper-validation-"))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CARGO_TARGET_DIR"] = str(neutral / "cargo-target")
    pytest = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    commands = [
        ("lapx_python", pytest + [str(ROOT / "lapx/tests"),
                                  "--ignore=" + str(ROOT / "lapx/tests/test_setup.py")], neutral),
    ]
    for package in ("lapcpp", "laprust"):
        tests = ROOT / package / "tests"
        # Editable-install and setup tests build from source; they do not test the installed wheel.
        commands.append((package + "_python", pytest + [str(tests), "--run-cross-package",
                         "--ignore=" + str(tests / "test_editable_install.py"),
                         "--ignore=" + str(tests / "test_setup.py")], neutral))
    commands += [
        ("laprust_rust", ["cargo", "test", "--release", "--locked", "--manifest-path",
                          str(ROOT / "laprust/src/_lapjv/Cargo.toml")], ROOT / "laprust"),
        ("lapx_rust", ["cargo", "test", "--release", "--locked", "--all-features", "--manifest-path",
                       str(ROOT / "lapx/src/_lapjvr/Cargo.toml")], ROOT / "lapx"),
    ]
    if shutil.which("cmake") and shutil.which("ctest"):
        build = neutral / "lapcpp-native"
        commands += [
            ("lapcpp_native_configure", ["cmake", "-S", str(ROOT / "lapcpp/tests/native"),
                                         "-B", str(build)], neutral),
            ("lapcpp_native_build", ["cmake", "--build", str(build), "--config", "Release"], neutral),
            ("lapcpp_native", ["ctest", "--test-dir", str(build), "-C", "Release"], neutral),
        ]
    reports = {}
    for name, command, cwd in commands:
        started = time.perf_counter()
        logpath = out / "validation_logs" / (name + ".log")
        logpath.parent.mkdir(exist_ok=True)
        with logpath.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        text = logpath.read_text(encoding="utf-8", errors="replace")
        reports[name] = {
            "exit_code": result.returncode, "seconds": time.perf_counter() - started,
            "command": command,
            "reported_pass_counts": [int(n) for n in re.findall(r"(\d+) passed", text)],
        }
        print(name, "exit", result.returncode, "pass counts", reports[name]["reported_pass_counts"], flush=True)
        if result.returncode:
            print(text[-10000:])
            raise SystemExit(result.returncode)
    report = {"packages": {name: info["version"] for name, info in runtime["packages"].items()},
              "suites": reports}
    (out / "test_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(neutral, ignore_errors=True)


if __name__ == "__main__":
    main()
