"""One isolated case/placement attempt. Launched only by the supervisor.

All IPC, affinity inspections, validation and accounting are outside the timed
blocks. A failed paired attempt is never combined with a subsequent attempt.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
import random
import time

from .common import append_jsonl, digest_json, input_digest, read_json, write_json


class MethodStopped(RuntimeError):
    pass


def local_day():
    """Session date label; one session per environment needs no calendar guard."""
    return dt.datetime.now().astimezone().date().isoformat()


class CorpusChanged(RuntimeError):
    """The entire attempt is invalid; never continue with contaminated inputs."""


class PlacementFailure(RuntimeError):
    """A measurement contract failed, not a mathematical solver result."""


def calibrated_repetitions(elapsed, repetitions, target):
    """Bound calibration growth; a slow call always remains one repetition."""
    return min(1_000_000, max(1, min(repetitions * 8,
                                   math.ceil(target * repetitions / max(elapsed, 1e-9)))))


def paired_schedule(methods, inputs, rounds, seed):
    input_rng = random.Random(seed)
    method_rng = random.Random(seed ^ 0xA531F20B)
    schedule = []
    for round_id in range(rounds):
        input_order = list(range(inputs))
        input_rng.shuffle(input_order)
        for i in input_order:
            order = list(methods)
            method_rng.shuffle(order)
            schedule.extend((round_id, i, method) for method in order)
    return schedule


def run(request):
    # These imports happen after the parent supplied the BLAS/thread limits.
    import numpy as np
    from threadpoolctl import threadpool_limits, threadpool_info
    from .adapters import prepare_method, UnsupportedMethod, UnsupportedCostDomain
    from .datasets import generate_input
    from .resources import R1
    from .topology import apply_and_verify, verify, verify_ephemeral_jvx
    from .validation import validate_output, objective, validate_input

    directory = Path(request["directory"])
    state_path, result_path = directory / "state.json", directory / "result.json"
    measurements = directory / "measurements.jsonl"
    case = request["case"]
    identity = {k: request[k] for k in ("campaign_id", "environment_id", "day", "session_id", "attempt_id")}
    identity.update(case_id=case["id"], scope=request["scope"])
    active, statuses, calls, metadata, hashes = dict(request.get("active_seed", {})), {}, {}, {}, []
    current_method = None

    def assert_corpus_unchanged():
        if [input_digest(value) for value in corpus] != hashes:
            raise CorpusChanged("Method changed a frozen input; the entire paired attempt is invalid")

    def state(phase, method=None, **extra):
        write_json(state_path, {**identity, "phase": phase, "method": method,
                               "monotonic": time.monotonic(),
                               "active_seconds": active.get(method, 0), **extra})

    def phase_call(method, input_index, count, phase):
        if active.get(method, 0) >= R1["method_active_seconds"]:
            raise MethodStopped("method_active_time_limit")
        state(phase, method, input_index=input_index, repetitions=count)
        call, value = calls[method], corpus[input_index]
        start = time.perf_counter_ns()
        output = None
        for _ in range(count):
            output = call(value)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        active[method] = active.get(method, 0) + elapsed
        append_jsonl(measurements, {**identity, "record_type": "phase", "method": method,
                                   "input_index": input_index, "phase": phase,
                                   "elapsed_seconds": elapsed, "repetitions": count})
        if active[method] > R1["method_active_seconds"]:
            raise MethodStopped("method_active_time_limit")
        return output, elapsed

    state("affinity")
    placement = {"before_imported_pools": apply_and_verify(request["cpus"])}
    threadpool_limits(limits=1)
    state("prepare")
    corpus = [generate_input(case, i) for i in range(request["inputs"])]
    for value in corpus:
        validate_input(case, value)
        # Reject accidental mutation by native wrappers early, then verify hashes
        # after screening and after the complete paired schedule as well.
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    hashes = [input_digest(value) for value in corpus]
    append_jsonl(measurements, {**identity, "record_type": "inputs", "hashes": hashes,
                               "case_definition_sha256": case["definition_sha256"]})
    build_index = read_json(request["build_index"]) if request.get("build_index") else None
    comparator = "scipy:capped" if case["objective"] == "optional" else "scipy"
    # Independent references are excluded from primary timing and are subject
    # to the same single-call screen watchdog as all measured methods.
    calls["__reference__"] = prepare_method(comparator, case, build_index)
    references = []
    for i, value in enumerate(corpus):
        output, _ = phase_call("__reference__", i, 1, "screen")
        state("validate", "__reference__", input_index=i)
        validate_output(case, value, output)
        references.append([objective(a, p, case["limit"]) for a, p in zip(value, output)]
                          if case["kind"] == "batch" else objective(value, output, case["limit"]))
    calls.pop("__reference__")
    for method in request["methods"]:
        state("prepare", method)
        try:
            calls[method] = prepare_method(method, case, build_index)
            metadata[method] = calls[method].method_info
        except (UnsupportedMethod, UnsupportedCostDomain) as error:
            statuses[method] = {"status": "unsupported", "reason": str(error)}
    repetitions = {}
    for method in list(calls):
        current_method = method
        try:
            # Pool creation precedes the warmup phase; screen every frozen input.
            for i, value in enumerate(corpus):
                output, _ = phase_call(method, i, 1, "screen")
                state("validate", method, input_index=i)
                checked = validate_output(case, value, output, references[i])
                items = checked["items"] if case["kind"] == "batch" else [checked]
                for item in items:
                    item["unmatched_endpoint_fraction"] = (1 - 2 * item["cardinality"] /
                        (case["rows"] + case["cols"])) if case["rows"] + case["cols"] else 0.
                append_jsonl(measurements, {**identity, "record_type": "validation", "method": method,
                                           "input_index": i, "input_sha256": hashes[i], "result": checked})
            assert_corpus_unchanged()
            state("affinity", method)
            try:
                if method == "batch:jvx:4":
                    placement["ephemeral_probe:" + method] = verify_ephemeral_jvx(request["cpus"], calls[method], corpus[0])
                placement["before_warmup:" + method] = verify(request["cpus"])
            except Exception as error:
                raise PlacementFailure(str(error)) from error
            for i, value in enumerate(corpus):
                phase_call(method, i, request["warmup_calls"], "warmup")
                count = 1
                for _ in range(12):
                    output, elapsed = phase_call(method, i, count, "calibration")
                    if elapsed >= request["target_seconds"] * .9 or count == 1_000_000:
                        break
                    count = calibrated_repetitions(elapsed, count, request["target_seconds"])
                state("validate", method, input_index=i)
                validate_output(case, value, output, references[i])
                repetitions[(method, i)] = count
            assert_corpus_unchanged()
            statuses[method] = {"status": "ready"}
        except (UnsupportedMethod, UnsupportedCostDomain) as error:
            assert_corpus_unchanged()
            statuses[method] = {"status": "unsupported", "reason": str(error)}
        except MethodStopped as error:
            assert_corpus_unchanged()
            statuses[method] = {"status": "resource_censored", "reason": str(error)}
        except (CorpusChanged, PlacementFailure, OSError):
            raise
        except Exception as error:
            # A solver may corrupt input and then raise or return a bad answer.
            # Do not continue screening other methods on that changed corpus.
            assert_corpus_unchanged()
            statuses[method] = {"status": "correctness_failed", "reason": f"{type(error).__name__}: {error}"}
        if statuses[method]["status"] != "ready":
            calls.pop(method)
    methods = list(calls)
    write_json(directory / "admitted.json", {"methods": methods, "statuses": statuses,
                                             "method_metadata": metadata,
                                             "input_hashes": hashes})
    schedule = paired_schedule(methods, len(corpus), request["rounds"], request["schedule_seed"])
    outcome = "complete"
    offender = None
    try:
        for round_id, i, method in schedule:
            current_method = method
            count = repetitions[(method, i)]
            output, elapsed = phase_call(method, i, count, "primary")
            # Check each block's output outside timing, including after warmup.
            state("validate", method, input_index=i, round=round_id)
            validate_output(case, corpus[i], output, references[i])
            append_jsonl(measurements, {**identity, "record_type": "timing", "method": method,
                                      "input_index": i, "input_sha256": hashes[i], "round": round_id,
                                      "repetitions": count, "elapsed_seconds": elapsed,
                                      "mean_seconds": elapsed / count, "status": "ok"})
        assert_corpus_unchanged()
        state("affinity")
        try:
            placement["after_primary"] = verify(request["cpus"])
        except Exception as error:
            raise PlacementFailure(str(error)) from error
        for method in methods:
            statuses[method] = {"status": "complete"}
    except (CorpusChanged, PlacementFailure, OSError):
        raise
    except Exception as error:
        assert_corpus_unchanged()
        outcome, offender = "restart_required", current_method
        statuses[offender] = {"status": "resource_censored" if isinstance(error, MethodStopped) else "correctness_failed",
                              "reason": f"{type(error).__name__}: {error}"}
    state("finalize")
    result = {**identity, "status": outcome, "offender": offender,
              "statuses": statuses, "active_seconds": active, "input_hashes": hashes,
              "method_metadata": metadata, "affinity": placement,
              "threadpools": threadpool_info(), "inputs": request["inputs"],
              "rounds": request["rounds"], "target_seconds": request["target_seconds"],
              "schedule_sha256": digest_json(schedule),
              "measurements_path": str(measurements.resolve())}
    write_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args(argv)
    request = read_json(args.request)
    try:
        run(request)
    except Exception as error:
        write_json(Path(request["directory"]) / "fatal.json",
                   {"status": "worker_failed", "reason": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    main()
