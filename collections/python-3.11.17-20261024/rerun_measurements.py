"""Rerun all timed suites sequentially, recording background CPU utilization.

Each suite keeps its original generator, solver adapters and timing protocol.
Outputs go to a new directory; promotion into the manuscript artifact is separate.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

PAPER = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PAPER / "tmp" / "rerun")
    options = parser.parse_args()
    options.output = options.output.resolve()  # The suites run from the parent directory.
    options.output.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    state = {"stage": "baseline"}
    records = []

    def monitor():
        with (options.output / "cpu_load.csv").open("w", newline="") as stream:
            fields = ["utc", "stage", "mean_cpu_percent", "mean_outside_cpu0_3_percent"] + [
                f"cpu{i}_percent" for i in range(psutil.cpu_count())]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            while not stop.is_set():
                values = psutil.cpu_percent(interval=1, percpu=True)
                row = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       "stage": state["stage"], "mean_cpu_percent": sum(values) / len(values),
                       "mean_outside_cpu0_3_percent": sum(values[4:]) / len(values[4:]),
                       **{f"cpu{i}_percent": value for i, value in enumerate(values)}}
                writer.writerow(row)
                stream.flush()

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    time.sleep(3)
    try:
        for stage, script in (("main", "run_experiments.py"),
                              ("optional", "compare_optional.py"),
                              ("scaling", "benchmark_scaling.py")):
            state["stage"] = stage
            command = [sys.executable, str(PAPER / script), "--output", str(options.output / stage)]
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(f"Starting {stage} measurements", flush=True)
            with (options.output / f"{stage}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=PAPER.parent, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                returncode = process.wait()
            records.append({"stage": stage, "command": command, "started_utc": started,
                            "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "script_sha256": hashlib.sha256((PAPER / script).read_bytes()).hexdigest(),
                            "exit_code": returncode})
            if returncode:
                raise RuntimeError(f"{stage} failed with code {returncode}; see archived log")
    finally:
        stop.set()
        monitor_thread.join(timeout=3)
        (options.output / "rerun.json").write_text(json.dumps({
            "reason": "All timing suites run sequentially on an otherwise idle system.",
            "status": "completed" if len(records) == 3 and all(r["exit_code"] == 0 for r in records) else "incomplete",
            "suites": records,
            "cpu_monitor": "One-second per-logical-CPU utilization snapshots; descriptive evidence, not isolation or frequency control.",
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
