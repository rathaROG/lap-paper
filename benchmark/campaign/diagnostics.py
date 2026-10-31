"""Separate input, representation and instrumented-native diagnostic records.

Nothing here supplies primary latency samples. Inputs stream one at a time in
fresh case/placement processes under R1. Construction-only timings are three
excluded repetitions per input/layout, not complete-call speed comparisons.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

import numpy as np

from .catalog import INPUT_COUNT
from .common import append_jsonl, digest_file, digest_json, input_digest, now, read_json, write_json
from .datasets import generate_input, geometry_points


class FrozenInputMutation(RuntimeError):
    """A diagnostic attempt cannot continue after its common input changes."""


def _degree_summary(degrees):
    return dict(minimum=int(degrees.min()) if len(degrees) else None,
                maximum=int(degrees.max()) if len(degrees) else None,
                mean=float(degrees.mean()) if len(degrees) else None,
                median=float(np.median(degrees)) if len(degrees) else None,
                isolated=int(np.count_nonzero(degrees == 0)))


def input_diagnostics(case, value, input_index=None):
    """Describe the actual admissible graph without allocating a full mask.

    ``finite_edge_density`` is measured after the independent gate. Raw finite
    costs are also counted, making the gate's effect explicit. Latent geometric
    correspondences are generator diagnostics, never tracking-accuracy labels.
    """
    def one(a):
        a = np.asarray(a)
        n, m = a.shape
        rows, cols = np.zeros(n, dtype=np.int64), np.zeros(m, dtype=np.int64)
        raw_finite, edges = 0, 0
        minimum, maximum = math.inf, -math.inf
        for start in range(0, n, 64):
            block = a[start:start + 64]
            finite = np.isfinite(block)
            raw_finite += int(np.count_nonzero(finite))
            values = block[finite]
            if values.size:
                minimum, maximum = min(minimum, float(values.min())), max(maximum, float(values.max()))
            allowed = finite
            if case.get("gate") is not None:
                allowed = finite & (block <= case["gate"])
            rows[start:start + len(block)] = allowed.sum(axis=1)
            cols += allowed.sum(axis=0)
            edges += int(np.count_nonzero(allowed))
        return dict(rows=n, cols=m, dtype=a.dtype.str, cost_buffer_bytes=a.nbytes,
                    raw_finite_edges=raw_finite, admissible_edges=edges,
                    raw_finite_edge_density=raw_finite / (n*m) if n*m else None,
                    finite_edge_density=edges / (n*m) if n*m else None,
                    row_degree=_degree_summary(rows), column_degree=_degree_summary(cols),
                    finite_cost_minimum=minimum if raw_finite else None,
                    finite_cost_maximum=maximum if raw_finite else None)

    batch = case.get("kind") == "batch"
    result = dict(case_id=case["id"], input_index=input_index, input_sha256=input_digest(value),
                  objective=case["objective"], limit=case.get("limit"), gate=case.get("gate"),
                  definition_sha256=case.get("definition_sha256"),
                  input_group_sha256=case.get("input_group", {}).get("definition_sha256"),
                  transpose=case.get("transpose", False), batch=batch,
                  matrices=[one(a) for a in value] if batch else [one(value)])
    if case.get("family") == "geometry":
        metadata = case["metadata"]
        result["geometry"] = {key: metadata[key] for key in (
            "q", "motion", "delta", "survivors", "miss_fraction", "false_fraction",
            "limit_over_delta", "gate_over_delta", "redundant_gate")}
        result["geometry"].update(
            canonical_rows=case["input_group"]["rows"], canonical_cols=case["input_group"]["cols"],
            boundary="surviving points displaced without clipping",
            interpretation="latent correspondence diagnostics, not tracking accuracy")
    return result


def save_geometry_latent(directory, case, input_index):
    """Store small point/correspondence arrays, never the dense cost matrix."""
    if case.get("family") != "geometry":
        return None
    sources, observations, pairs = geometry_points(case["input_group"], input_index)
    name = f"latent-{input_index}.npz"
    path = Path(directory) / name
    np.savez_compressed(path, canonical_source_points=sources,
                        canonical_observation_points=observations, canonical_pairs=pairs,
                        oriented_pairs=pairs[:, ::-1] if case["transpose"] else pairs)
    return dict(path=name, sha256=digest_file(path), shared_point_count=len(pairs),
                orientation="transpose" if case["transpose"] else "canonical",
                interpretation="generator correspondences only; no accuracy score inferred")


def layout_diagnostics(case, value, kind, layout, repetitions=3):
    """Time only gate application and layout construction, with actual bytes."""
    from .adapters import cost_layout, shape_layout, masked_cost
    if case.get("kind") == "batch" or kind not in ("cost", "shape"):
        raise ValueError("Layout diagnostics require one matrix and cost/shape layout")
    if repetitions < 1:
        raise ValueError("Positive construction repetition count required")
    samples, info = [], None
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        work = masked_cost(value, case.get("gate"))
        transformed, info = (cost_layout(work, case["limit"], layout) if kind == "cost"
                             else shape_layout(work, layout))
        elapsed = (time.perf_counter_ns() - start) / 1e9
        samples.append(elapsed)
        dimensions, buffer_bytes = list(transformed.shape), int(transformed.nbytes)
        del transformed, work
    return dict(kind=kind, layout=layout, repetitions=repetitions,
                construction_seconds=samples, median_construction_seconds=statistics.median(samples),
                construction_elapsed_seconds=sum(samples), diagnostic_timing_not_primary=True,
                transformed_dimensions=dimensions, cost_buffer_bytes=buffer_bytes,
                objective_offset=info["objective_offset"], sentinel=info.get("sentinel"),
                timing_boundary="gate application and layout construction only; no solver/output decoding",
                scope="secondary_diagnostic_never_primary_latency")


def _jobs(case):
    """One layout construction per representation, not four duplicate backends."""
    methods = [m for m in case["methods"] if m.startswith(("policy:", "workspace:"))]
    layouts = sorted({(m.split(":")[0], m.split(":")[2]) for m in case["methods"]
                      if m.startswith(("cost:", "shape:"))})
    return methods, layouts


def _bounded_cases(manifest):
    cases = manifest["cases"]
    names = {"base:near256x320", "base:optional128x160", "base:negative128x160",
             "batch:batch1024x8x10", "batch:8x4096x256:optional", "batch:8x8192x256:optional"}
    result = [c for c in cases if c["id"] in names]
    result.append(next(c for c in cases if c["family"] == "geometry" and c["rows"] == 256
                       and c["cols"] == 320 and c["gate"] is not None and c["gate"] < c["limit"]))
    return result


def _build_record(index, language, arm):
    arm = {"w0": "reference", "w1": "fresh_state", "w2": "fresh_worker"}.get(arm, arm)
    builds = index.get("builds", [])
    if isinstance(builds, dict):
        builds = list(builds.values())
    matches = [record for record in builds if record.get("language") == language and record.get("arm_id") == arm]
    if len(matches) != 1:
        raise ValueError(f"Need one separate diagnostic build for {language}/{arm}")
    return matches[0]


def run_worker(request):
    """Fresh-process excluded diagnostics, supervised by the existing watchdog."""
    from threadpoolctl import threadpool_limits
    from .adapters import masked_cost, prepare_method
    from .topology import apply_and_verify, verify
    from .validation import validate_input, validate_output

    directory = Path(request["directory"])
    case = request["case"]
    identity = {key: request[key] for key in (
        "campaign_id", "environment_id", "day", "session_id", "attempt_id", "scope")}
    identity["case_id"] = case["id"]
    output_path = directory / "diagnostics.jsonl"

    def state(phase, method=None, **extra):
        write_json(directory / "state.json", {**identity, "phase": phase, "method": method,
                                               "monotonic": time.monotonic(), **extra})

    def emit(record):
        append_jsonl(output_path, {**identity, "record_type": "diagnostic", **record})

    state("affinity")
    placement = {"before": apply_and_verify(request["cpus"])}
    threadpool_limits(limits=1)
    index = read_json(request["build_index"]) if request.get("build_index") else None
    solvers = {}
    statuses = {}
    layout_statuses = {f"{kind}:{layout}": {"status": "planned", "completed_inputs": []}
                       for kind, layout in request["layouts"]}
    for method in request["methods"]:
        state("prepare", method)
        try:
            if index is None:
                raise ValueError("No diagnostic build index supplied; primary builds are never substituted")
            from .diagnostic_build import load_diagnostic
            parts = method.split(":")
            solvers[method] = load_diagnostic(_build_record(index, parts[1], parts[2]))
        except Exception as error:
            statuses[method] = dict(status="unsupported", reason=f"{type(error).__name__}: {error}")
            emit(dict(study="native_counters", method=method, **statuses[method]))
    reference = prepare_method("scipy:capped" if case["objective"] == "optional" else "scipy", case)
    for i in range(request["inputs"]):
        state("prepare", input_index=i)
        value = generate_input(case, i)
        validate_input(case, value)
        value.flags.writeable = False
        diagnostic = input_diagnostics(case, value, i)
        if request["layouts"] or request["workers"] == 1:
            diagnostic["latent_data"] = save_geometry_latent(directory, case, i)
        emit(dict(study="input_graph", **diagnostic))
        expected = None
        if solvers:
            state("screen", "__diagnostic_reference__", input_index=i)
            expected = reference(value)
        for kind, layout in request["layouts"]:
            method = f"{kind}:scipy:{layout}"
            state("screen", method, input_index=i)
            try:
                record = layout_diagnostics(case, value, kind, layout, repetitions=3)
                emit(dict(study="layout_construction", input_index=i,
                          input_sha256=diagnostic["input_sha256"], status="complete", **record))
                layout_statuses[f"{kind}:{layout}"]["completed_inputs"].append(i)
                if layout_statuses[f"{kind}:{layout}"]["status"] != "failed":
                    layout_statuses[f"{kind}:{layout}"]["status"] = "complete"
            except OSError:
                raise
            except Exception as error:
                layout_statuses[f"{kind}:{layout}"].update(status="failed", reason=f"{type(error).__name__}: {error}")
                emit(dict(study="layout_construction", input_index=i, kind=kind, layout=layout,
                          status="failed", reason=f"{type(error).__name__}: {error}"))
        for method, solver in solvers.items():
            state("prepare", method, input_index=i)
            work = masked_cost(value, case.get("gate"))
            kwargs = dict(limit=math.inf if case["limit"] is None else case["limit"],
                          batch=case["kind"] == "batch", workers=request["workers"])
            try:
                # Warm exactly this input once, outside the recorded counter
                # sample. Retained workspace then follows the arm's own rules.
                state("screen", method, input_index=i, diagnostic_stage="warmup")
                warmup_started = time.perf_counter()
                solver.solve(work, **kwargs)
                warmup_seconds = time.perf_counter() - warmup_started
                solver.reset()
                state("screen", method, input_index=i, diagnostic_stage="counted_call")
                diagnostic_started = time.perf_counter()
                output = solver.solve(work, **kwargs)
                counters = solver.snapshot()
                diagnostic_seconds = time.perf_counter() - diagnostic_started
                state("validate", method, input_index=i)
                validation = validate_output(case, value, output, expected)
                checks = validation["items"] if case["kind"] == "batch" else [validation]
                for checked in checks:
                    endpoints = case["rows"] + case["cols"]
                    checked["unmatched_endpoints"] = endpoints - 2 * checked["cardinality"]
                    checked["unmatched_fraction"] = checked["unmatched_endpoints"] / endpoints if endpoints else 0.
                if input_digest(value) != diagnostic["input_sha256"]:
                    raise FrozenInputMutation("Diagnostic solver changed the frozen input")
                statuses.setdefault(method, dict(status="complete", completed_inputs=[]))
                statuses[method].setdefault("completed_inputs", []).append(i)
                emit(dict(study="native_counters", input_index=i,
                          input_sha256=diagnostic["input_sha256"], method=method, status="complete",
                          counters=counters, output_validation=validation,
                          build=solver.build_record, warmup_calls=1, counted_calls=1,
                          diagnostic_wall_seconds=diagnostic_seconds,
                          diagnostic_warmup_seconds=warmup_seconds,
                          diagnostic_timing_not_primary=True,
                          primary_timing_eligible=False))
                del output
            except (FrozenInputMutation, OSError):
                raise
            except Exception as error:
                if input_digest(value) != diagnostic["input_sha256"]:
                    raise FrozenInputMutation("Diagnostic failure also changed the frozen input") from error
                statuses.setdefault(method, dict(completed_inputs=[]))
                statuses[method].update(status="failed", reason=f"{type(error).__name__}: {error}")
                emit(dict(study="native_counters", input_index=i, method=method, **statuses[method]))
            del work
        del value
        del expected
    state("affinity")
    placement["after"] = verify(request["cpus"])
    state("finalize")
    records = list(statuses.values()) + list(layout_statuses.values())
    complete = all(record["status"] == "complete" and len(record["completed_inputs"]) == request["inputs"]
                   for record in records)
    outcome = "complete" if complete else "incomplete"
    result = {**identity, "status": outcome, "method_statuses": statuses,
              "layout_statuses": layout_statuses,
              "records_path": str(output_path), "placement": placement,
              "resident_inputs": 1, "frozen_inputs_examined": request["inputs"],
              "interpretation": "separate explanatory diagnostics; no primary timing observations"}
    write_json(directory / "result.json", result)
    return result


def run_diagnostics(manifest, output, build_index=None, *, full=False, cpus=None, single_cpu=None):
    """Collect bounded diagnostics by default; full mode is explicit at the CLI."""
    import psutil
    from .resources import HostLock, admission
    from .runner import environment, launch
    from .topology import discover, validate_selection

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a fresh diagnostic directory; existing records are immutable")
    output.mkdir(parents=True, exist_ok=True)
    topology = discover()
    cpus = list(cpus or topology["recommended_batch_cpus"])
    single_cpu = cpus[0] if single_cpu is None and cpus else single_cpu
    cores = validate_selection(topology, cpus, single_cpu)
    original_mask = psutil.Process().cpu_affinity()
    occupied = {cpu for core in cores for cpu in core["logical_cpus"]}
    supervisor = sorted(set(original_mask) - occupied)
    if not supervisor:
        raise ValueError("Need a supervisor hardware thread outside all measured physical cores")
    selected = [c for c in manifest["cases"] if c["kind"] != "memory"] if full else _bounded_cases(manifest)
    day = dt.datetime.now().astimezone().date().isoformat()
    index_path = str(Path(build_index).resolve()) if build_index else None
    env = environment()
    scope = "full_excluded_diagnostics" if full else "bounded_excluded_diagnostics"
    source_hashes = {path.name: digest_file(path) for path in Path(__file__).parent.glob("*.py")}
    source_root = output / "sources"
    source_directory = source_root / "benchmark" / "campaign"
    source_directory.mkdir(parents=True)
    (source_root / "benchmark" / "__init__.py").write_text("", encoding="utf-8")
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, source_directory / path.name)
    config = dict(scope=scope, inputs=manifest.get("input_count", INPUT_COUNT) if full else 2,
                  profile=manifest.get("profile", "full"), resident_inputs=1,
                  day=day, session_id=f"diagnostic-{day}-{uuid.uuid4().hex[:12]}",
                  campaign_id=digest_json(dict(manifest=manifest["definition_sha256"], sources=source_hashes,
                                               diagnostic_build_index=digest_file(index_path) if index_path else None)),
                  environment_id=digest_json({key: env.get(key) for key in (
                      "system", "python", "hostname", "machine", "processor", "packages", "native_artifact_hashes")}),
                  cpus=cpus, single_cpu=single_cpu, supervisor_cpus=supervisor,
                  source_root=str(source_root))
    write_json(output / "configuration.json", config)
    write_json(output / "manifest.json", manifest)
    write_json(output / "environment.json", {**env, "topology": topology})
    write_json(output / "selection.json", selected)
    write_json(output / "source-hashes.json", source_hashes)
    if index_path:
        shutil.copy2(index_path, output / "diagnostic-build-index.json")
    summary = dict(scope=scope, status="running", jobs=[], started_utc=now())
    with HostLock():
        psutil.Process().cpu_affinity(supervisor)
        try:
            for number, case in enumerate(selected):
                methods, layouts = _jobs(case)
                for workers in (1, 4):
                    chosen = [m for m in methods if (int(m.rsplit(":", 1)[1]) if m.startswith("workspace:") else 1) == workers]
                    constructions = layouts if workers == 1 else []
                    if workers == 4 and not chosen:
                        continue
                    estimated_methods = chosen + [f"{kind}:scipy:{layout}" for kind, layout in constructions]
                    admit = admission(case, estimated_methods or ["jvrust"], input_count=1)
                    allowed = {method for method, record in admit["methods"].items() if record["status"] == "admitted"}
                    chosen = [m for m in chosen if m in allowed]
                    constructions = [(kind, layout) for kind, layout in constructions if f"{kind}:scipy:{layout}" in allowed]
                    entry = dict(case_id=case["id"], workers=workers, admission=admit,
                                 planned_methods=methods, planned_layouts=layouts)
                    if admit["host_status"] == "postponed" or not allowed:
                        entry["status"] = "postponed" if admit["host_status"] == "postponed" else "resource_excluded"
                    else:
                        attempt = f"{number:04d}-w{workers}"
                        request = {**config, "case": case, "methods": chosen, "layouts": constructions,
                                   "workers": workers, "cpus": cpus if workers == 4 else [single_cpu],
                                   "attempt_id": attempt, "directory": str(output / "attempts" / attempt),
                                   "build_index": index_path}
                        result, watchdog = launch(request, "diagnostics")
                        entry.update(status=result["status"] if result else "incomplete", result=result,
                                     watchdog=watchdog, directory=request["directory"])
                    summary["jobs"].append(entry)
                    write_json(output / "summary.json", summary)
            summary["status"] = "finished"
        finally:
            psutil.Process().cpu_affinity(original_mask)
            summary["finished_utc"] = now()
            write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    if len(arguments) == 1 and not arguments[0].startswith("-"):
        request = read_json(arguments[0])
        try:
            run_worker(request)
        except Exception as error:
            write_json(Path(request["directory"]) / "fatal.json",
                       {"status": "diagnostic_worker_failed", "reason": f"{type(error).__name__}: {error}"})
            raise
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-index", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-full-campaign", action="store_true")
    parser.add_argument("--cpus", nargs=4, type=int)
    parser.add_argument("--single-cpu", type=int)
    args = parser.parse_args(arguments)
    if args.full and not args.allow_full_campaign:
        parser.error("Full diagnostic collection requires --full --allow-full-campaign")
    from .catalog import build_manifest
    run_diagnostics(build_manifest(), args.output, args.build_index, full=args.full,
                    cpus=args.cpus, single_cpu=args.single_cpu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
