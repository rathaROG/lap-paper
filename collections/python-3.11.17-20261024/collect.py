"""Collect the complete Python 3.11.17 replication in this isolated directory."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def preserved():
    files = json.loads((ROOT / "protected_previous_files.json").read_text())
    for name, expected in files.items():
        assert hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest() == expected, name
    return len(files)


def main():
    assert sys.version_info[:3] == (3, 11, 17)
    assert Path(sys.executable).resolve() == Path(r"C:\dev\exc\python\p311\python.exe").resolve()
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    state = dict(status="running", started_utc=now(), executable=sys.executable, stages=[])
    if (ROOT / "collection_run.json").exists():
        previous = json.loads((ROOT / "collection_run.json").read_text())
        assert previous["status"] == "incomplete" and not previous["stages"], "This collector requires a fresh collection."
        save(ROOT / "logs/preflight_attempt.json", previous)
    save(ROOT / "collection_run.json", state)

    def run(name, arguments):
        command = [sys.executable, "-u", *arguments]
        stage = dict(stage=name, command=command, started_utc=now(), status="running")
        state["stages"].append(stage)
        save(ROOT / "collection_run.json", state)
        print(f"Starting {name}", flush=True)
        with (ROOT / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.rstrip(), flush=True)
            code = process.wait()
        stage.update(status="completed" if code == 0 else "failed", exit_code=code, completed_utc=now())
        save(ROOT / "collection_run.json", state)
        if code:
            raise RuntimeError(f"{name} failed with exit code {code}; see logs/{name}.log")

    try:
        preserved()
        for args, filename in ((["list", "--format=json"], "installed_packages.json"),
                               (["inspect"], "pip_inspect.json"), (["freeze"], "requirements-installed.txt")):
            result = subprocess.run([sys.executable, "-m", "pip", *args], check=True,
                                    capture_output=True, text=True, encoding="utf-8")
            (ROOT / filename).write_text(result.stdout, encoding="utf-8")
        run("pip_check", ["-m", "pip", "check"])
        run("runtime", ["record_runtime.py"])
        run("paper_harness", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"])
        run("release_validation", ["validate_release.py"])
        run("batch_validation", ["run_experiments.py", "--validate-batch-only", "--output", "results"])
        run("base_measurements", ["rerun_measurements.py", "--output", "staging"])
        for source, destination in (("main", "results"), ("optional", "results/formulations"),
                                    ("scaling", "results/scaling")):
            target = ROOT / destination
            target.mkdir(exist_ok=True)
            for path in (ROOT / "staging" / source).iterdir():
                assert not (target / path.name).exists(), path.name
                shutil.copy2(path, target / path.name)
        (ROOT / "results/rerun").mkdir()
        for path in (ROOT / "staging").iterdir():
            if path.is_file():
                shutil.copy2(path, ROOT / "results/rerun" / path.name)
        run("extended_measurements", ["run_extended_experiments.py", "--output", "extended"])
        state.update(status="completed", completed_utc=now(), previous_files_unchanged=preserved())
    except BaseException as error:
        state.update(status="incomplete", error=repr(error), interrupted_utc=now())
        raise
    finally:
        save(ROOT / "collection_run.json", state)
    print("All Python 3.11.17 measurements and validation completed.", flush=True)


if __name__ == "__main__":
    main()
