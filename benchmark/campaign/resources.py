"""R1 admission and an external process-tree watchdog (never a solver timer)."""
from __future__ import annotations

import contextlib
import os
from pathlib import Path
import sys
import tempfile
import time

GIB = 1024 ** 3
R1 = {
    "id": "R1", "array_bytes": 4 * GIB, "admission_bytes": 12 * GIB,
    "resident_bytes": 16 * GIB, "free_bytes": 24 * GIB,
    "screen_seconds": 60.0, "block_seconds": 120.0,
    "method_active_seconds": 7200.0, "watchdog_interval_seconds": 0.05,
    "metric": "sum of RSS/working-set bytes across measured process and descendants",
}


def allocation_estimate(case, method, input_count=None):
    """Conservative documented allocation model, not a measured memory result.

    Reserve a runtime/uncertainty GiB, original inputs, two live transformed
    cost arrays (adapter and solver copy), one original-size conversion/orientation
    array, and generous linear state. This model must remain versioned with R1.
    """
    n, m = int(case["rows"]), int(case["cols"])
    r, c = min(n, m), max(n, m)
    metadata = case.get("metadata", {})
    if input_count is None:
        from .catalog import INPUT_COUNT
        input_count = 1 if case.get("kind") == "memory" else INPUT_COUNT
    if input_count < 1:
        raise ValueError("Resource estimates require a positive resident input count")
    batch = int(case.get("batch_size", metadata.get("batch_size", metadata.get("count", 1))))
    if case.get("kind") != "batch":
        batch = 1
    original = n * m * 8
    transformed = original
    parts = method.split(":")
    # Batch wrappers invoke these same single-matrix representations per active
    # worker. In particular JVX still expands optional matrices internally.
    backend = parts[1] if parts[0] in ("batch", "loop") else method
    if method.startswith("cost:") or method == "scipy:square":
        transformed = (n + m) ** 2 * 8
    elif method.startswith("shape:"):
        transformed = ((n + m) ** 2 if parts[-1].startswith("sum_") else c * c) * 8
    elif method in ("scipy:dummies", "scipy:diagonal_dummies"):
        transformed = r * (c + r) * 8
    elif backend == "jvx" and case.get("objective") != "required":
        transformed = (n + m) ** 2 * 8
    elif backend in ("jvx", "jvs"):
        transformed = c * c * 8
    elif parts[0] == "sparse":
        # Worst case: every entry admissible, mirrored, with int64 coordinates.
        transformed = 4 * original
    workers = int(parts[-1]) if method.startswith(("workspace:", "batch:")) and parts[-1].isdigit() else 1
    corpus = original * batch * input_count
    # Python assignment keeps the preceding return value alive during the next
    # complete call, so include both output sets in repeated warm blocks.
    output = 2 * batch * r * 16
    linear = 256 * (n + m + 1)
    peak = GIB + corpus + workers * (2 * transformed + original + linear) + output
    return {"model": "R1-v2: runtime_uncertainty+corpus+workers*(2*transform+original+linear)+two_outputs",
            "input_bytes": corpus, "single_input_bytes": original,
            "largest_cost_array_bytes": max(transformed, original * batch),
            "transformed_cost_array_bytes": transformed, "estimated_peak_bytes": peak,
            "output_reserve_bytes": output, "runtime_uncertainty_bytes": GIB,
            "workers": workers, "batch_size": batch, "input_count": input_count}


def admission(case, methods, input_count=None, check_free=True):
    import psutil
    free = psutil.virtual_memory().available
    records = {}
    for method in methods:
        estimate = allocation_estimate(case, method, input_count)
        reasons = []
        if estimate["largest_cost_array_bytes"] > R1["array_bytes"]:
            reasons.append("individual_cost_array_limit")
        if estimate["estimated_peak_bytes"] > R1["admission_bytes"]:
            reasons.append("estimated_working_set_limit")
        status = "resource_excluded" if reasons else "admitted"
        if check_free and free < R1["free_bytes"]:
            reasons.append("host_headroom_unavailable")
            status = "postponed"
        records[method] = {"status": status,
                           "reasons": reasons, "estimate": estimate}
    return {"policy": R1, "available_physical_bytes": free,
            "host_status": "postponed" if check_free and free < R1["free_bytes"] else "ready",
            "methods": records}


def process_tree_rss(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            total += process.memory_info().rss
    return total


def terminate_tree(pid):
    import psutil
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in children + [parent]:
        with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
            process.kill()
    psutil.wait_procs(children + [parent], timeout=5)


def watch(process, state_path, setup_timeout=300.0):
    """Read phase timestamps written OUTSIDE timed blocks by the case worker."""
    from .common import read_json
    started = time.monotonic()
    peak = 0
    last_state = {}
    breach = None
    maximum_poll_interval = 0.0
    previous_poll = None
    while process.poll() is None:
        poll_started = time.monotonic()
        if previous_poll is not None:
            maximum_poll_interval = max(maximum_poll_interval, poll_started - previous_poll)
        previous_poll = poll_started
        try:
            rss = process_tree_rss(process.pid)
        except Exception as error:
            # Failure to inspect the whole tree is not permission to keep
            # measuring with an understated RSS. End and retain this attempt.
            breach = "resource_monitor_error"
            last_state["resource_monitor_error"] = repr(error)
            terminate_tree(process.pid)
            break
        peak = max(peak, rss)
        try:
            last_state = read_json(state_path)
        except (OSError, ValueError):
            pass
        elapsed = time.monotonic() - float(last_state.get("monotonic", started))
        phase = last_state.get("phase", "setup")
        limit = R1["screen_seconds"] if phase == "screen" else R1["block_seconds"]
        if phase in ("setup", "prepare", "validate", "finalize", "affinity"):
            limit = setup_timeout
        if rss > R1["resident_bytes"]:
            breach = "memory_limit"
        elif phase in ("screen", "warmup", "calibration", "primary") and float(last_state.get("active_seconds", 0)) + elapsed > R1["method_active_seconds"]:
            breach = "method_active_time_limit"
        elif elapsed > limit:
            breach = "screen_timeout" if phase == "screen" else f"{phase}_timeout"
        if breach:
            terminate_tree(process.pid)
            break
        # Target a 50-ms start-to-start cadence, not 50 ms plus inspection time.
        # Scheduler/inspection overshoots are recorded, never hidden.
        time.sleep(max(0., poll_started + R1["watchdog_interval_seconds"] - time.monotonic()))
    process.wait()
    return {"status": "terminated" if breach else "exited", "reason": breach,
            "exit_code": process.returncode, "peak_resident_bytes": peak,
            "memory_overshoot_bytes": max(0, peak - R1["resident_bytes"]),
            "last_state": last_state, "enforcement": "external process-tree RSS watchdog",
            "interval_seconds": R1["watchdog_interval_seconds"],
            "maximum_observed_poll_interval_seconds": maximum_poll_interval}


class HostLock:
    """OS-released host-wide lock: a crash cannot leave a permanently stale lock."""
    def __enter__(self):
        path = Path(tempfile.gettempdir()) / "lap-paper-campaign-host.lock"
        self.file = open(path, "a+b")
        try:
            self.file.seek(0)
            if not self.file.read(1):
                self.file.write(b"0")
                self.file.flush()
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("Another campaign measurement owns the host lock") from None
        return self

    def __exit__(self, *args):
        self.file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()

