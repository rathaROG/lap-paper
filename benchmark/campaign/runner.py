"""External supervisor for bounded smoke runs and explicitly enabled sessions."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

from .common import digest_file, digest_json, now, read_json, write_json
from .resources import HostLock, R1, admission, terminate_tree, watch
from .topology import discover, validate_selection
from .worker import local_day


class HostHeadroomUnavailable(RuntimeError):
    """Transient host pressure postpones a case; it does not censor a solver."""

    def __init__(self, admission_record):
        super().__init__("Insufficient free host memory to prepare the next measurement")
        self.admission = admission_record


def require_host_headroom(admission_record):
    if (admission_record.get("host_status") == "postponed" or
            any(item.get("status") == "postponed" for item in admission_record["methods"].values())):
        raise HostHeadroomUnavailable(admission_record)


def select_cases(manifest, *, case_start=None, case_stop=None, case_ids=None):
    """Select a fixed shard without changing the full manifest or method sets."""
    cases = manifest["cases"]
    if case_ids is not None:
        if case_start is not None or case_stop is not None:
            raise ValueError("Use --case-id or --case-start/--case-stop, not both")
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("Case IDs must be nonempty and unique")
        known = {case["id"] for case in cases}
        missing = set(case_ids) - known
        if missing:
            raise ValueError(f"Unknown case IDs: {sorted(missing)}")
        return [case for case in cases if case["id"] in set(case_ids)]
    start = 0 if case_start is None else case_start
    stop = len(cases) if case_stop is None else case_stop
    if not 0 <= start < stop <= len(cases):
        raise ValueError(f"Require 0 <= case-start < case-stop <= {len(cases)} (stop is exclusive)")
    return cases[start:stop]


def workers_for(method):
    return int(method.rsplit(":", 1)[1]) if method.startswith(("workspace:", "batch:")) else 1


def split_placements(methods):
    return [(workers, [m for m in methods if workers_for(m) == workers]) for workers in (1, 4)
            if any(workers_for(m) == workers for m in methods)]


def environment():
    packages = {}
    # LAPX imports as ``lap``, but its installed distribution is named ``lapx``.
    for name in ("numpy", "scipy", "psutil", "threadpoolctl", "lapx", "laprust", "lapcpp", "Cython"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import psutil
    artifacts = {}
    for name in ("numpy", "scipy", "lapx", "laprust", "lapcpp"):
        try:
            dist = importlib.metadata.distribution(name)
            artifacts[name] = {str(file): digest_file(dist.locate_file(file)) for file in dist.files or []
                               if str(file).endswith((".pyd", ".so"))}
        except importlib.metadata.PackageNotFoundError:
            pass
    system_settings = {}
    try:
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                system_settings["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            system_settings["power_scheme"] = subprocess.check_output(
                ["powercfg", "/getactivescheme"], text=True, timeout=5).strip()
        else:
            system_settings["cpu_model"] = next(line.split(":", 1)[1].strip() for line in
                                                 Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
            system_settings["governors"] = {str(p): p.read_text().strip() for p in
                                            Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor")}
    except (OSError, StopIteration, subprocess.SubprocessError) as error:
        system_settings["unavailable"] = str(error)
    return {"python": platform.python_version(), "python_executable": sys.executable,
            "platform": platform.platform(), "system": platform.system(), "machine": platform.machine(),
            "hostname": platform.node(), "logical_cpus": psutil.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total, "packages": packages,
            "processor": platform.processor(), "frequency_at_start": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "native_artifact_hashes": artifacts,
            "system_settings": system_settings,
            "background_observation": {"cpu_percent_before_session": psutil.cpu_percent(interval=.1),
                                       "available_memory_bytes": psutil.virtual_memory().available,
                                       "process_count": len(psutil.pids()),
                                       "note": "Observation only; no claim that the host is otherwise idle"},
            "timer": "time.perf_counter_ns", "created_utc": now()}


def attempt_activity(directory, watchdog):
    """Keep the session budget across discarded/restarted paired attempts."""
    totals = {}
    path = Path(directory) / "measurements.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("record_type") == "phase":
                method = row["method"]
                totals[method] = totals.get(method, 0) + row["elapsed_seconds"]
    state = watchdog.get("last_state", {})
    if watchdog.get("status") == "terminated" and state.get("phase") in ("screen", "warmup", "calibration", "primary"):
        method = state.get("method")
        # Phase log contains completed blocks only; account for the interrupted one.
        totals[method] = totals.get(method, 0) + max(0, time.monotonic() - state["monotonic"])
    return totals


def smoke_plan(manifest):
    """Fixed, bounded coverage probes selected by contract, never by winners."""
    cases = manifest["cases"]
    selected = []
    def pick(predicate, method_filter=lambda method: True):
        case = next(c for c in cases if predicate(c))
        selected.append({**case, "methods": [m for m in case["methods"] if method_filter(m)]})
    # Complete small layout/policy sweeps, both required and optional contracts.
    pick(lambda c: c["id"] == "base:near256x320")
    pick(lambda c: c["id"] == "base:optional128x160")
    pick(lambda c: c["id"] == "base:negative128x160")
    pick(lambda c: c["kind"] == "single" and c["family"] == "geometry" and c["rows"] == 256
         and c["cols"] == 320 and c["gate"] is not None and c["gate"] < c["limit"])
    # Wide required input at the padded-baseline limit and a tall optional input.
    pick(lambda c: c["id"] == "extended:020_extreme_rectangle_required_64x4096")
    pick(lambda c: c["id"] == "extended:023_extreme_rectangle_optional_4096x64")
    # The largest square and geometric inputs anchor one-call > target timing.
    pick(lambda c: c["kind"] == "single" and c["rows"] == c["cols"] == 5000
         and c["objective"] == "required")
    pick(lambda c: c["kind"] == "single" and c["rows"] == c["cols"] == 5000
         and c["objective"] == "optional")
    pick(lambda c: c["kind"] == "single" and c["rows"] == 4096 and c["cols"] == 5120
         and c["objective"] == "optional" and c["family"] == "geometry")
    # One old standard batch plus both orientation-buffer sides of the 8 MiB cutoff.
    pick(lambda c: c["id"] == "batch:batch64x64x80")
    pick(lambda c: c["id"] == "batch:8x4096x256:optional")
    pick(lambda c: c["id"] == "batch:8x8192x256:optional")
    pick(lambda c: c["id"] == "memory:extended:optional:256x512",
         lambda m: m in ("jvrust", "scipy:capped", "scipy:square"))
    return selected


def launch(request, module):
    directory = Path(request["directory"])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", request)
    write_json(directory / "state.json", {"phase": "setup", "monotonic": time.monotonic()})
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONHASHSEED="0")
    command = [sys.executable, "-u", "-m", "benchmark.campaign." + module, str(directory / "request.json")]
    with (directory / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=request.get("source_root", Path(__file__).resolve().parents[2]), env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            watchdog = watch(process, directory / "state.json")
        except BaseException:
            terminate_tree(process.pid)
            raise
    write_json(directory / "watchdog.json", watchdog)
    result = read_json(directory / "result.json") if (directory / "result.json").exists() else None
    return result, watchdog


def run_session(manifest, output, build_index=None, *, smoke=True, cpus=None, single_cpu=None,
                selected_cases=None):
    import psutil
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains a session; choose a new directory (attempts are immutable)")
    output.mkdir(parents=True, exist_ok=True)
    topology = discover()
    cpus = list(cpus or topology["recommended_batch_cpus"])
    if not cpus:
        raise ValueError("Physical core class could not be established; supply verified --cpus")
    single_cpu = cpus[0] if single_cpu is None else single_cpu
    measured_cores = validate_selection(topology, cpus, single_cpu)
    original_mask = psutil.Process().cpu_affinity()
    occupied_logical = {cpu for core in measured_cores for cpu in core["logical_cpus"]}
    supervisor_cpus = sorted(set(original_mask) - occupied_logical)
    if not supervisor_cpus:
        raise ValueError("R1 requires a supervisor CPU outside the measured mask")
    env = environment()
    if not smoke and (env["system"] not in ("Windows", "Linux") or env["python"] not in ("3.11.18", "3.14.10")):
        raise ValueError("Primary session requires Windows/Linux Python3.11.18 or3.14.10; this is a smoke-only interpreter")
    scope = "smoke_not_publication_evidence" if smoke else "primary_one_environment_session"
    day = local_day()
    session_id = day + "-" + uuid.uuid4().hex[:12]
    index_path = str(Path(build_index).resolve()) if build_index else None
    build_records = read_json(index_path).get("builds", []) if index_path else []
    if not smoke and any(r.get("python", {}).get("version", "").split(" ")[0] != env["python"] for r in build_records):
        raise ValueError("Primary session requires controlled builds made with the pinned Python patch version")
    stable_env = {k: env[k] for k in ("python", "platform", "system", "machine", "hostname", "logical_cpus",
                                    "physical_memory_bytes", "packages", "native_artifact_hashes", "system_settings")}
    stable_env["controlled_artifacts"] = {f"{r['language']}:{r['arm_id']}": r["artifact_sha256"] for r in build_records}
    stable_env["placement"] = {"single_cpu": single_cpu, "batch_cpus": cpus}
    env_id = env["system"] + "-py" + env["python"] + "-" + digest_json(stable_env)[:16]
    cases = selected_cases if selected_cases is not None else (smoke_plan(manifest) if smoke else manifest["cases"])
    from .catalog import BLOCK_SECONDS, INPUT_COUNT, MEMORY_TRIALS, ROUNDS, WARMUP_CALLS
    config = {"inputs": 2 if smoke else manifest.get("input_count", INPUT_COUNT),
              "rounds": 2 if smoke else manifest.get("rounds", ROUNDS),
              "target_seconds": .005 if smoke else manifest.get("target_block_seconds", BLOCK_SECONDS),
              "warmup_calls": 1 if smoke else manifest.get("warmup_calls", WARMUP_CALLS),
              "memory_trials": 1 if smoke else manifest.get("memory_trials", MEMORY_TRIALS),
              "profile": manifest.get("profile", "full"), "scope": scope, "day": day,
              "environment_id": env_id, "session_id": session_id,
              "single_cpu": single_cpu, "batch_cpus": cpus, "supervisor_cpus": supervisor_cpus,
              "schedule_seed": 20261030}
    write_json(output / "manifest.json", manifest)
    write_json(output / "selection.json", cases)
    write_json(output / "environment.json", {**env, "topology": topology, "stratum": stable_env,
                                             "environment_id": env_id})
    source_root = output / "sources"
    source_dir = source_root / "benchmark" / "campaign"
    source_dir.mkdir(parents=True)
    (source_dir.parent / "__init__.py").write_text("# Frozen campaign source package.\n", encoding="utf-8")
    source_hashes = {}
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source_dir / path.name)
        source_hashes[path.name] = digest_file(source_dir / path.name)
    write_json(output / "source-hashes.json", source_hashes)
    implementation_hashes = {name: value for name, value in source_hashes.items() if not name.startswith("test_")}
    config["campaign_id"] = digest_json({"manifest": manifest["definition_sha256"], "sources": implementation_hashes,
                                        "resources": R1, "scope": scope})
    config["source_root"] = str(source_root)
    write_json(output / "configuration.json", config)
    if build_index:
        shutil.copy2(index_path, output / "build-index.json")
    summary = {"scope": scope, "configuration": config, "manifest_sha256": manifest["definition_sha256"],
               "started_utc": now(), "status": "running", "jobs": [], "memory": [],
               "planned_case_ids": [case["id"] for case in manifest["cases"]],
               "selected_case_ids": [case["id"] for case in cases],
               "unselected_case_ids": [case["id"] for case in manifest["cases"]
                                       if case["id"] not in {chosen["id"] for chosen in cases}],
               "remaining_case_ids": [case["id"] for case in cases]}
    started = time.perf_counter()
    case_number, current_job = 0, None
    with HostLock():
        psutil.Process().cpu_affinity(supervisor_cpus)
        try:
            for case_number, case in enumerate(cases):
                current_job = None
                print(f"{scope}: {case_number+1}/{len(cases)} {case['id']}", flush=True)
                if case["kind"] == "memory":
                    for method in case["methods"]:
                        admit = admission(case, [method], input_count=1)
                        if admit["methods"][method]["status"] != "admitted":
                            summary["memory"].append({"case_id": case["id"], "method": method, "admission": admit})
                            require_host_headroom(admit)
                            continue
                        for trial in range(config["memory_trials"]):
                            if trial:
                                # Every fresh process prepares its own input. A
                                # prior trial's headroom check is no longer enough.
                                admit = admission(case, [method], input_count=1)
                                if admit["methods"][method]["status"] == "postponed":
                                    summary["memory"].append({"case_id": case["id"], "method": method,
                                                              "admission": admit})
                                require_host_headroom(admit)
                            directory = output / "memory" / f"{case_number:04d}-{digest_json(method)[:10]}-{trial}"
                            request = {**config, "case": case, "method": method, "trial": trial,
                                       "directory": str(directory), "cpus": [single_cpu], "build_index": index_path}
                            trial_started = time.perf_counter()
                            result, watchdog = launch(request, "memory_worker")
                            summary["memory"].append({"case_id": case["id"], "method": method, "trial": trial,
                                                       "result": result, "watchdog": watchdog,
                                                       "directory": str(directory),
                                                       "case_seconds": time.perf_counter() - trial_started})
                            if result is None:
                                break  # deterministic failure retained; do not retry its remaining trials
                            write_json(output / "summary.json", summary)
                    summary["remaining_case_ids"] = [item["id"] for item in cases[case_number + 1:]]
                    continue
                for workers, methods in split_placements(case["methods"]):
                    case_started = time.perf_counter()
                    admit = admission(case, methods, config["inputs"])
                    remaining = [m for m in methods if admit["methods"][m]["status"] == "admitted"]
                    job = {"case_id": case["id"], "workers": workers, "methods": methods, "admission": admit,
                           "statuses": {m: v for m, v in admit["methods"].items() if v["status"] != "admitted"},
                           "attempts": [], "status": "incomplete"}
                    current_job = job
                    require_host_headroom(admit)
                    attempt = 0
                    activity = {}
                    while remaining:
                        if attempt:
                            # A paired restart prepares another full corpus after
                            # the failed worker exits; check current host pressure.
                            retry_admit = admission(case, remaining, config["inputs"])
                            job.setdefault("admission_rechecks", []).append(retry_admit)
                            job["statuses"].update({m: detail for m, detail in retry_admit["methods"].items()
                                                    if detail["status"] != "admitted"})
                            require_host_headroom(retry_admit)
                            remaining = [m for m in remaining if retry_admit["methods"][m]["status"] == "admitted"]
                            if not remaining:
                                break
                        attempt_id = f"{case_number:04d}-w{workers}-a{attempt:02d}"
                        directory = output / "attempts" / attempt_id
                        request = {**config, "case": case, "methods": remaining, "attempt_id": attempt_id,
                                   "directory": str(directory), "cpus": cpus if workers == 4 else [single_cpu],
                                   "build_index": index_path, "active_seed": activity}
                        result, watchdog = launch(request, "worker")
                        for method, seconds in attempt_activity(directory, watchdog).items():
                            activity[method] = activity.get(method, 0) + seconds
                        entry = {"attempt_id": attempt_id, "directory": str(directory), "watchdog": watchdog,
                                 "status": result["status"] if result else "failed"}
                        job["attempts"].append(entry)
                        if result is not None:
                            job["statuses"].update(result["statuses"])
                            if result["status"] == "complete" and watchdog["exit_code"] == 0:
                                job.update(status="complete", attempt_id=attempt_id,
                                           measurements_path=result["measurements_path"],
                                           input_hashes=result["input_hashes"])
                                break
                            offender = result.get("offender")
                            remaining = [m for m in remaining if m != offender and
                                         result["statuses"].get(m, {}).get("status") not in
                                         ("unsupported", "correctness_failed", "resource_censored")]
                        else:
                            offender = watchdog["last_state"].get("method")
                            solver_breaches = {"memory_limit", "screen_timeout", "warmup_timeout",
                                               "calibration_timeout", "primary_timeout", "method_active_time_limit"}
                            if watchdog["reason"] not in solver_breaches or offender not in remaining:
                                job["failure"] = read_json(directory / "fatal.json") if (directory / "fatal.json").exists() else watchdog
                                break
                            admitted_path = directory / "admitted.json"
                            if admitted_path.exists():
                                # A primary-block kill can occur after other
                                # methods have already failed validation or
                                # proved unsupported. Preserve those outcomes;
                                # restart only the actual surviving admission.
                                snapshot = read_json(admitted_path)
                                job["statuses"].update(snapshot.get("statuses", {}))
                                survivors = set(snapshot["methods"])
                                remaining = [m for m in remaining if m in survivors]
                            job["statuses"][offender] = {"status": "resource_censored", "reason": watchdog["reason"]}
                            remaining = [m for m in remaining if m != offender]
                        if offender not in methods:
                            break
                        attempt += 1
                    job["case_seconds"] = time.perf_counter() - case_started
                    job["active_seconds"] = activity
                    summary["jobs"].append(job)
                    current_job = None
                    write_json(output / "summary.json", summary)
                summary["remaining_case_ids"] = [item["id"] for item in cases[case_number + 1:]]
            summary["status"] = "finished"
        except HostHeadroomUnavailable as error:
            # Outputs remain immutable. A later invocation retries this entire
            # case in a fresh directory; earlier placements/trials in this case
            # must not compete with that complete retry as another successful estimate.
            summary.update(status="postponed", postponed_case_ids=[case["id"]],
                           postponement={"reason": "host_headroom_unavailable", "case_id": case["id"],
                                         "admission": error.admission,
                                         "note": "Retry the entire affected case in a fresh output directory. "
                                                 "Its partial evidence is retained but unavailable to summaries."},
                           remaining_case_ids=[item["id"] for item in cases[case_number:]])
            if current_job is not None and current_job not in summary["jobs"]:
                current_job.update(case_seconds=time.perf_counter() - case_started)
                summary["jobs"].append(current_job)
            for job in summary["jobs"]:
                if job["case_id"] == case["id"]:
                    job["status"] = "postponed"
        finally:
            psutil.Process().cpu_affinity(original_mask)
            summary.update(finished_utc=now(), wall_seconds=time.perf_counter()-started)
            write_json(output / "summary.json", summary)
    return summary
