"""Run the four timing suites sequentially, as in the paper's collections.

Each suite is started as its own process from its own folder, so every suite
remains independently runnable; this script only orders them, logs their
output and records per-CPU utilization once a second (descriptive evidence of
background load, not isolation). Order: base, formulations, scaling, extended
(about 25 minutes with the defaults). LSAPE and the real-world benchmark have
their own workflows; see their README files.

Without --output, new measurements go to results/full-YYYYMMDD next to this
script, using the local date. A non-empty output directory is never overwritten.

    python run_all.py
    python run_all.py --output runs/all-20261029
    python run_all.py --output runs/smoke --smoke
    python run_all.py --output runs/all --cpu 2 --batch-cpus 2 4 6 8
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

from lsape.redaction import redact

HERE = Path(__file__).resolve().parent
SUITES = ("base", "formulations", "scaling", "extended")

def scrub(text, output=None):
    """Hide local paths in console output and saved launcher records."""
    roots = [(str(HERE), "<benchmark>")]
    if output is not None:
        roots.append((str(output), "<output>"))
    for root, placeholder in sorted(roots, key=lambda item: -len(item[0])):
        pattern = re.compile(re.escape(root).replace(r"\\", r"[\\/]") + r"(?=$|[\\/\s\"'])",
                             re.IGNORECASE if os.name == "nt" else 0)
        text = pattern.sub(placeholder, text)
    return redact(text)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=HERE / "results" / f"full-{datetime.date.today():%Y%m%d}",
                        help="new directory; default: benchmark/results/full-YYYYMMDD (local date)")
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--cpu", type=int, help="logical CPU for single-problem timing")
    parser.add_argument("--batch-cpus", type=int, nargs=4, help="four logical CPUs for batch timing (base)")
    parser.add_argument("--smoke", action="store_true", help="one round, 1 ms blocks, tiny extended grid; not paper timings")
    args = parser.parse_args()
    import psutil

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(scrub(f"Choose a new output directory: {output}", output))
    output.mkdir(parents=True, exist_ok=True)
    stop, state, records = threading.Event(), {"stage": "starting"}, []

    def monitor():
        with (output / "cpu_load.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["utc", "stage", "mean_cpu_percent"]
                                    + [f"cpu{i}_percent" for i in range(psutil.cpu_count())])
            writer.writeheader()
            while not stop.is_set():
                values = psutil.cpu_percent(interval=1, percpu=True)
                writer.writerow({"utc": utc(), "stage": state["stage"], "mean_cpu_percent": sum(values) / len(values),
                                 **{f"cpu{i}_percent": v for i, v in enumerate(values)}})
                stream.flush()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    time.sleep(2)
    try:
        for suite in args.suites:
            command = [sys.executable, "-u", "run.py", "--output", str(output / suite)]
            if args.cpu is not None:
                command += ["--cpu", str(args.cpu)]
            if suite == "base" and args.batch_cpus:
                command += ["--batch-cpus", *map(str, args.batch_cpus)]
            if args.smoke:
                command += ["--smoke"] if suite == "extended" else ["--rounds", "1", "--block-seconds", "0.001"]
            state["stage"] = suite
            started = utc()
            print(scrub(f"== {suite}: {' '.join(command[2:])}", output), flush=True)
            with (output / f"{suite}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=HERE / suite, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, errors="replace")
                for line in process.stdout:
                    line = scrub(line, output)
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                code = process.wait()
            records.append({"suite": suite, "arguments": [scrub(part, output) for part in command[2:]],
                            "started_utc": started, "completed_utc": utc(), "exit_code": code})
            if code:
                raise SystemExit(scrub(f"{suite} failed with exit code {code}; see {output / (suite + '.log')}", output))
    finally:
        stop.set()
        thread.join(timeout=3)
        (output / "run_all.json").write_text(json.dumps({
            "status": "completed" if len(records) == len(args.suites) and all(r["exit_code"] == 0 for r in records)
            else "incomplete", "smoke": args.smoke, "python": sys.version, "suites": records,
            "cpu_monitor": "One-second per-logical-CPU utilization snapshots; descriptive, not isolation."},
            indent=2) + "\n", encoding="utf-8")
    print(scrub(f"All selected suites completed: {output}", output), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(scrub(traceback.format_exc()), file=sys.stderr, end="")
        raise SystemExit(1)
