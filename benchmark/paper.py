"""Run the paper workflow with automatic paths, placement and preparation."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

from .campaign.common import now, read_json, write_json
from .campaign.resources import HostLock, terminate_tree
from .campaign.topology import discover, validate_selection
from .paths import RUNS, default_output
from .preparation import prepare_campaign, prepare_lsape


class WorkflowError(RuntimeError):
    """A paper workflow stage failed; its evidence remains in the output."""


def check_dependencies():
    required = {"lapx": "0.13.0", "laprust": "1.2.0", "lapcpp": "1.2.0",
                "pyppbox": "4.0.0"}
    for name, expected in required.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            raise WorkflowError(f"Need {name}=={expected}; run the two pip install commands in benchmark/README.md")
    from .application.realbench_paths import pyppbox_module
    pyppbox_module()


def execute(command: list[str], *, log: Path):
    """Stream each fresh-process stage to the console and its complete log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   env=environment)
        try:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            returncode = process.wait()
        except BaseException:
            terminate_tree(process.pid)
            process.wait()
            raise
        finally:
            process.stdout.close()
    if returncode:
        raise WorkflowError(f"Command exited {returncode}; details: {log}")


def run_workflow(mode: str, output: Path, cache: Path, *, cpus: list[int] | None = None,
                 single_cpu: int | None = None, profile: str = "full") -> dict:
    if mode not in ("smoke", "run"):
        raise ValueError("Choose smoke or run")
    if profile not in ("full", "light"):
        raise ValueError("Choose the full or light campaign profile")
    output, cache = Path(output).resolve(), Path(cache).resolve()
    if output.exists() and any(output.iterdir()):
        raise WorkflowError(f"Existing results are preserved; choose a fresh output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "mode": mode, "profile": profile, "status": "running",
              "scope": "paper_smoke_not_publication_evidence" if mode == "smoke" else "paper_one_environment",
              "platform": platform.system(), "python": platform.python_version(),
              "executable": sys.executable, "started_utc": now(), "stages": []}
    started = time.perf_counter()

    def save():
        write_json(output / "status.json", record)

    def command(name, module, *arguments):
        if module == "benchmark.campaign" and profile == "light":
            arguments = (*arguments, "--profile", "light")
        log = output / "logs" / (name + ".log")
        argv = [sys.executable, "-B", "-m", module, *map(str, arguments)]
        stage = {"name": name, "status": "running", "command": argv, "log": str(log)}
        record["stages"].append(stage)
        save()
        print(f"{name}...", flush=True)
        try:
            execute(argv, log=log)
            if module == "benchmark.campaign" and arguments[0] == "diagnose":
                summary_path = Path(arguments[arguments.index("--output") + 1]) / "summary.json"
                summary = read_json(summary_path)
                if summary.get("status") != "finished" or any(
                        job.get("status") not in ("complete", "resource_excluded")
                        for job in summary.get("jobs", [])):
                    raise WorkflowError(f"Diagnostics are incomplete; details: {summary_path}")
        except BaseException as error:
            stage.update(status="failed", error=str(error))
            raise
        stage["status"] = "complete"
        save()

    save()
    try:
        check_dependencies()
        topology = discover()
        automatic = cpus is None
        cpus = list(topology["recommended_batch_cpus"] if automatic else cpus)
        if single_cpu is None:
            single_cpu = topology["recommended_single_cpu"] if automatic else (cpus[0] if cpus else None)
        if len(cpus) != 4:
            if automatic:
                raise WorkflowError("Automatic CPU selection could not identify four eligible physical cores. "
                                    "Inspect 'python -m benchmark.campaign topology'; select verified CPUs with "
                                    "--cpus and --single-cpu. Core-class and affinity checks still apply.")
            raise WorkflowError("Select four physical cores for the paper's batch comparisons")
        cores = validate_selection(topology, cpus, single_cpu)
        if platform.system() == "Windows":
            try:
                validate_selection(topology, [0], 0)
            except ValueError as error:
                raise WorkflowError("The frozen LSAPE protocol requires an available, verified performance core at CPU 0: "
                                    f"{error}") from error
        occupied = {cpu for core in cores for cpu in core["logical_cpus"]}
        if not set(topology["allowed_cpus"]) - occupied:
            raise WorkflowError("Need a supervisor CPU outside the four measured physical cores")
        record.update(cpus=cpus, single_cpu=single_cpu, cache=str(cache))
        write_json(output / "topology.json", topology)
        print(f"Output: {output}\nProfile: {profile}\nCPUs: {cpus}; single CPU: {single_cpu}", flush=True)

        preparation = {"name": "preparation", "status": "running"}
        record["stages"].append(preparation)
        save()
        try:
            with HostLock():
                primary = prepare_campaign(cache)
                diagnostic = prepare_campaign(cache, diagnostic=True)
                lsape = prepare_lsape(cache) if platform.system() == "Windows" else None
            preparation.update(status="complete", build_index=str(primary),
                               diagnostic_build_index=str(diagnostic), lsape_build=str(lsape) if lsape else None)
        except BaseException as error:
            preparation.update(status="failed", error=str(error))
            raise
        if lsape is None:
            record["excluded_suites"] = {"lsape": "Windows/MSVC protocol"}
        save()

        placement = ["--cpus", *map(str, cpus), "--single-cpu", str(single_cpu)]
        command("assignment-validation", "benchmark.campaign", "validate", "--build-index", primary,
                "--output", output / "campaign-validation.json")
        command("application-validation", "benchmark.application", "--validate",
                "--output", output / "application-validation")
        if lsape:
            command("lsape-verification", "benchmark.lsape", "verify", "--build-dir", lsape,
                    "--output-dir", output / "lsape-verification")
        command("assignment-smoke", "benchmark.campaign", "smoke", "--build-index", primary,
                "--output", output / "campaign-smoke", *placement)
        command("assignment-estimate", "benchmark.campaign", "estimate", "--smoke-root", output / "campaign-smoke",
                "--output", output / "campaign-estimate")
        estimate = output / "campaign-estimate" / "estimate.json"
        if estimate.is_file():
            hours = read_json(estimate)["session_hours_low_central_high"]
            record["assignment_session_hours_low_central_high"] = hours
            print(f"Assignment scheduling estimate: {hours[1]:.2f} hours central", flush=True)
        command("diagnostic-smoke", "benchmark.campaign", "diagnose", "--build-index", diagnostic,
                "--output", output / "diagnostic-smoke", *placement)
        command("application-smoke", "benchmark.application", "--smoke", "--cpu", single_cpu,
                "--output", output / "application-smoke")
        if lsape:
            command("lsape-smoke", "benchmark.lsape", "smoke", "--build-dir", lsape,
                    "--output", output / "lsape-smoke")

        if mode == "run":
            command("assignment-session", "benchmark.campaign", "run", "--allow-full-campaign",
                    "--build-index", primary, "--output", output / "campaign-session", *placement)
            command("assignment-summary", "benchmark.campaign", "summarize", output / "campaign-session",
                    "--output", output / "campaign-summary.json")
            command("diagnostics", "benchmark.campaign", "diagnose", "--full", "--allow-full-campaign",
                    "--build-index", diagnostic, "--output", output / "campaign-diagnostics", *placement)
            command("application-session", "benchmark.application", "--campaign", "--allow-full-campaign",
                    "--cpu", single_cpu, "--output", output / "application-session",
                    *(["--profile", "light"] if profile == "light" else []))
            if lsape:
                reports = []
                for index in (1, 2, 3):
                    session = output / f"lsape-session-{index}"
                    command(f"lsape-session-{index}", "benchmark.lsape", "run", "--allow-full-campaign",
                            "--build-dir", lsape, "--session-index", index, "--output", session)
                    reports.append(session / "report.json")
                command("lsape-summary", "benchmark.lsape", "summarize", *reports,
                        "--output", output / "lsape-summary")
        record["status"] = "complete"
        return record
    except BaseException as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        if isinstance(error, KeyboardInterrupt):
            raise
        raise WorkflowError(f"{error}\nRun status: {output / 'status.json'}") from error
    finally:
        record.update(finished_utc=now(), wall_seconds=time.perf_counter() - started)
        save()


def main(argv=None):
    configure_stdout = getattr(sys.stdout, "reconfigure", None)
    if callable(configure_stdout):
        configure_stdout(errors="replace")
    parser = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__)
    parser.add_argument("mode", choices=("smoke", "run"))
    opt_in = parser.add_mutually_exclusive_group()
    opt_in.add_argument("--allow-full-campaign", action="store_true", help="Run the unchanged full paper measurements")
    opt_in.add_argument("--allow-full-campaign-light", action="store_true",
                        help="Run the light paper protocol: five inputs, three rounds and reduced geometry baselines/settings")
    parser.add_argument("--output", type=Path, help="Optional fresh output directory; otherwise selected automatically")
    parser.add_argument("--cpus", nargs=4, type=int,
                        help="Optional four batch CPUs; physical-core and performance-class checks still apply")
    parser.add_argument("--single-cpu", type=int,
                        help="Optional serial CPU in the batch selection; defaults to the first explicit --cpus entry")
    args = parser.parse_args(argv)
    if args.mode == "run" and not (args.allow_full_campaign or args.allow_full_campaign_light):
        parser.error("Paper measurements require --allow-full-campaign or --allow-full-campaign-light; "
                     "use smoke for the bounded check")
    if args.mode != "run" and args.allow_full_campaign_light:
        parser.error("--allow-full-campaign-light requires run; smoke remains the bounded check")
    profile = "light" if args.allow_full_campaign_light else "full"
    if platform.system() not in ("Windows", "Linux"):
        parser.error("Paper measurements support Windows and Linux")
    if args.mode == "run" and platform.python_version() not in ("3.11.18", "3.14.10"):
        parser.error("Paper measurements require Python 3.11.18 or 3.14.10")
    output = args.output
    if output is None:
        base = default_output("paper-" + args.mode + ("-light" if profile == "light" else ""))
        output, suffix = base, 2
        while output.exists():
            output = base.with_name(base.name + f"-{suffix}")
            suffix += 1
    cache = RUNS / ".cache" / f"{platform.system().lower()}-py{platform.python_version()}"
    try:
        run_workflow(args.mode, output, cache, cpus=args.cpus, single_cpu=args.single_cpu, profile=profile)
    except WorkflowError as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"Interrupted; completed evidence remains in {output}", file=sys.stderr)
        return 130
    print(f"{args.mode} complete: {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
