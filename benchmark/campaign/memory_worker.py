"""One fresh-process memory trial, using native process high-water counters."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

from .common import input_digest, read_json, write_json


def snapshot(process):
    info = process.memory_info()
    if sys.platform == "win32":
        return {"rss": info.rss, "peak": info.peak_wset, "counter": "peak_working_set"}
    if sys.platform.startswith("linux"):
        import resource
        return {"rss": info.rss, "peak": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "counter": "ru_maxrss_bytes"}
    raise RuntimeError("Memory counters are implemented for Windows/Linux only")


def run(request):
    import numpy as np
    import psutil
    from threadpoolctl import threadpool_limits
    from .adapters import prepare_method
    from .datasets import generate_input
    from .topology import apply_and_verify, verify
    from .validation import validate_output, validate_input

    directory = Path(request["directory"])
    case, method = request["case"], request["method"]
    def state(phase):
        write_json(directory / "state.json", {"phase": phase, "method": method, "monotonic": time.monotonic()})
    state("affinity")
    placement = {"before": apply_and_verify(request["cpus"])}
    threadpool_limits(limits=1)
    index = read_json(request["build_index"]) if request.get("build_index") else None
    state("prepare")
    call = prepare_method(method, {**case, "kind": "single"}, index)
    tiny = {**case, "kind": "single", "rows": 4, "cols": 4}
    warm = prepare_method(method, tiny, index)
    state("screen")
    warm(np.ones((4, 4), dtype=np.float64))
    state("prepare")
    value = generate_input(case, request["trial"])
    validate_input(case, value)
    value.flags.writeable = False
    digest = input_digest(value)
    gc.collect()
    process = psutil.Process()
    # Write watchdog state before taking baseline: IPC allocations are excluded.
    state("screen")
    before = snapshot(process)
    start = time.perf_counter_ns()
    output = call(value)
    elapsed = (time.perf_counter_ns() - start) / 1e9
    # The same declared output lifetime in every fresh trial; outside latency.
    hold_seconds = .050
    time.sleep(hold_seconds)
    after = snapshot(process)
    state("validate")
    reference = prepare_method("scipy:capped" if case["objective"] == "optional" else "scipy", case, index)
    expected = reference(value)
    checked = validate_output(case, value, output, expected)
    if input_digest(value) != digest:
        raise ValueError("Memory trial changed input")
    placement["after"] = verify(request["cpus"])
    result = {k: request[k] for k in ("campaign_id", "environment_id", "day", "session_id", "scope", "trial")}
    result.update(status="complete", case_id=case["id"], method=method, input_sha256=digest,
                  process_id=process.pid, process_creation_time=process.create_time(),
                  input_bytes=value.nbytes, before=before, after=after,
                  additional_peak=max(0, after["peak"] - before["peak"]),
                  baseline_peak_headroom=max(0, before["peak"] - before["rss"]),
                  elapsed_seconds=elapsed, hold_seconds=hold_seconds, validation=checked, affinity=placement,
                  caveat="Peak increment can be hidden by earlier import/input-allocation high-water marks")
    state("finalize")
    write_json(directory / "result.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    request = read_json(parser.parse_args(argv).request)
    try:
        run(request)
    except Exception as error:
        write_json(Path(request["directory"]) / "fatal.json",
                   {"status": "worker_failed", "reason": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
