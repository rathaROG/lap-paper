"""Compose retained GT and a later RF-DETR campaign without running measurements.

Only complete, matching per-pipeline summaries are selected. Original raw files
are copied byte for byte, with their original campaign context stored separately.
Round indexes remain local to their originating session; no observations or
statistics are pooled between sessions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from realbench_campaign import SOLVERS, write_json
from realbench_data import SEQUENCES, sha256_file


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def select_summary(run, source):
    """Select the four complete pipelines for one source, without recomputation."""
    run = Path(run).resolve(strict=True)
    summary = _read(run / "summary.json")
    protocol = summary["protocol"]
    expected_cases = {f"{sequence}_{source}_{tracker}"
                      for sequence in SEQUENCES for tracker in ("centroid", "sort")}
    methods = tuple(protocol.get("solvers", ()))
    _require(set(methods) == set(SOLVERS) and len(methods) == len(SOLVERS),
             f"{run}: solver coverage differs from the declared nine methods")
    tracking = [row for row in summary["tracking"] if row.get("source") == source]
    replay = [row for row in summary["replay"] if row.get("source") == source]
    _require(Counter((row["case"], row["solver"]) for row in tracking) ==
             Counter((case, solver) for case in expected_cases for solver in SOLVERS),
             f"{run}: incomplete or duplicate {source} tracking settings")
    _require(Counter((row["case"], row["precision"], row["solver"]) for row in replay) ==
             Counter((case, precision, solver) for case in expected_cases
                     for precision in ("float32", "float64") for solver in SOLVERS),
             f"{run}: incomplete or duplicate {source} replay settings")
    for row in (*tracking, *replay):
        _require(row["case"] == f"{row['sequence']}_{source}_{row['tracker']}" and
                 row["sequence"] in SEQUENCES and row["tracker"] in ("centroid", "sort"),
                 f"{run}: case metadata disagrees with its name")
        _require(row["rounds"] == protocol["rounds"], f"{run}: inconsistent round count")
    for row in tracking:
        _require(row["frames"] == protocol["frames_per_clip"], f"{run}: inconsistent frame coverage")
    inputs = [item for item in summary["inputs"] if item.get("source") == source]
    _require(Counter(item["sequence"] for item in inputs) == Counter(SEQUENCES),
             f"{run}: incomplete or duplicate {source} input identities")
    for item in inputs:
        _require(item["frames"] == protocol["frames_per_clip"], f"{run}: input frame count differs")
    _require(set(summary["validation"]) >= expected_cases, f"{run}: missing validation records")
    return {"run": run, "source": source, "summary": summary, "protocol": protocol,
            "tracking": tracking, "replay": replay, "inputs": inputs, "cases": expected_cases}


def _native_identity(environment):
    return {package: sorted((Path(item["path"]).name, item["sha256"])
                             for item in data.get("native_files", []))
            for package, data in environment.get("solver_binaries", {}).items()}


def validate_compatibility(gt, rf):
    """Check comparability while retaining both complete environment records."""
    left, right = gt["protocol"], rf["protocol"]
    keys = ("frames_per_clip", "rounds", "cpu", "library_threads", "tracker_input_precision",
            "replay_precisions", "warmup_frames", "minimum_replay_block_seconds", "adapter_scope",
            "stock_tracker_semantics", "evaluation")
    for key in keys:
        _require(key in left and left.get(key) == right.get(key), f"Campaign protocols differ for {key}")
    env_gt, env_rf = gt["summary"]["environment"], rf["summary"]["environment"]
    for key in ("cpu", "platform", "python", "executable", "packages"):
        _require(key in env_gt and env_gt.get(key) == env_rf.get(key), f"Campaign environments differ for {key}")
    native_gt, native_rf = _native_identity(env_gt), _native_identity(env_rf)
    _require(native_gt and native_gt == native_rf, "Campaign native solver binaries differ")
    for name in ("realbench_tracking.py", "realbench_evaluate.py"):
        a = env_gt.get("measurement_sources", {}).get(name, {}).get("sha256")
        b = env_rf.get("measurement_sources", {}).get(name, {}).get("sha256")
        _require(a and a == b, f"Campaigns differ in {name}")
    for name, data in env_gt.get("pyppbox_sources", {}).items():
        if name.startswith("modules/trackers/") or name == "utils/persontools.py":
            _require(data.get("sha256") and data.get("sha256") ==
                     env_rf.get("pyppbox_sources", {}).get(name, {}).get("sha256"),
                     f"Campaign tracker source differs: {name}")
    for sequence in SEQUENCES:
        a = next(item for item in gt["inputs"] if item["sequence"] == sequence)
        b = next(item for item in rf["inputs"] if item["sequence"] == sequence)
        for name in ("video", "gt", "gt_map"):
            digest = a.get("inputs", {}).get(name, {}).get("sha256")
            _require(digest and digest == b.get("inputs", {}).get(name, {}).get("sha256"),
                     f"Campaign input identity differs for {sequence}/{name}")
        _require(a.get("frame_mapping") == b.get("frame_mapping"), f"Frame mapping differs for {sequence}")
    return {"recorded_cpu_platform_interpreter_packages_match": True,
            "native_solver_binary_hashes_match": True,
            "tracker_and_evaluator_source_hashes_match": True,
            "video_and_reference_hashes_match": True,
            "qualification": "Matching recorded identities do not establish an isolated session or independent hardware replication."}


def combine(gt_run, rf_run, output):
    """Copy selected saved evidence and generate a derived report; never measure."""
    gt, rf = select_summary(gt_run, "gt"), select_summary(rf_run, "rfdetr_nano")
    checks = validate_compatibility(gt, rf)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Combined output must be a fresh directory")
    for selected in (gt, rf):
        if output.is_relative_to(selected["run"]) or selected["run"].is_relative_to(output):
            raise ValueError("Combined output must be separate from both original campaigns")
    # Validate all expected artifact paths before writing a partial composition.
    planned = []
    campaign_info = {}
    combined_samples = []
    for label, selected in (("gt_v1", gt), ("rf_v2", rf)):
        run, cases = selected["run"], selected["cases"]
        root = Path("provenance") / "campaigns" / label
        for name in ("summary.json", "protocol.json", "environment.json", "file_manifest.json", "tracking_samples.json"):
            planned.append((run / name, root / name, label))
        if (run / "input_validation.json").is_file():
            planned.append((run / "input_validation.json", root / "input_validation.json", label))
        for path in sorted((run / "provenance").rglob("*")):
            if path.is_file():
                planned.append((path, root / "original_provenance" / path.relative_to(run / "provenance"), label))
        for item in selected["inputs"]:
            name = f"{item['sequence']}_{selected['source']}.json"
            cache = _read(run / "inputs" / name)
            _require(cache.get("metadata") == item, f"{run}: input cache metadata differs from summary: {name}")
            planned.append((run / "inputs" / name, Path("inputs") / name, label))
        for case in sorted(cases):
            planned.append((run / "matrices" / f"{case}.npz", Path("matrices") / f"{case}.npz", label))
            for precision in ("float32", "float64"):
                name = f"{case}_{precision}.json"
                planned.append((run / "replay" / name, Path("replay") / name, label))
            for solver in SOLVERS:
                name = f"{case}__{solver}.json"
                for folder in ("predictions", "metrics", "profiles"):
                    planned.append((run / folder / name, Path(folder) / name, label))
                if solver == "scipy":
                    profile = f"{case}__scipy_without_capture.json"
                    planned.append((run / "profiles" / profile, Path("profiles") / profile, label))
                for index in range(selected["protocol"]["rounds"]):
                    relative = Path("timing") / f"round_{index:02d}" / name
                    planned.append((run / relative, relative, label))
        raw_samples = _read(run / "tracking_samples.json")
        selected_samples = [row for row in raw_samples if row.get("case") in cases]
        expected = {(case, solver, index) for case in cases for solver in SOLVERS
                    for index in range(selected["protocol"]["rounds"])}
        _require(Counter((row["case"], row["solver"], row["round"]) for row in selected_samples) == Counter(expected),
                 f"{run}: incomplete/duplicate selected tracking samples")
        combined_samples.extend(selected_samples)
        summary = selected["summary"]
        campaign_info[label] = {"source": selected["source"], "original_directory": run.name,
                               "summary_sha256": sha256_file(run / "summary.json"),
                               "started_utc": summary.get("started_utc"), "completed_utc": summary.get("completed_utc"),
                               "cases": sorted(cases), "copied_context": root.as_posix()}
    destinations = [relative.as_posix() for _, relative, _ in planned]
    _require(len(destinations) == len(set(destinations)), "Selected evidence destination collision")
    from realbench_paths import redact
    for original, relative, _ in planned:
        _require(original.is_file(), f"Required saved evidence is missing: {redact(str(original))}")
        # Preserve original evidence byte-for-byte, but do not publish records
        # from older runs if they still contain local paths.
        if original.suffix.lower() == ".json":
            value = _read(original)
            _require(redact(value) == value,
                     f"Saved evidence contains local paths: {relative.as_posix()}")
        elif original.suffix.lower() in {".log", ".txt"}:
            value = original.read_text(encoding="utf-8")
            _require(redact(value) == value,
                     f"Saved evidence contains local paths: {relative.as_posix()}")

    output.mkdir(parents=True)
    copied = {}
    runs = {"gt_v1": gt["run"], "rf_v2": rf["run"]}
    for original, relative, label in planned:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(original)
        shutil.copy2(original, target)
        _require(sha256_file(target) == digest, f"Copied evidence hash differs: {relative}")
        # Relative to its source campaign directory: no local absolute path is recorded.
        copied[relative.as_posix()] = {"origin_campaign": label,
                                      "original_path": original.relative_to(runs[label]).as_posix(),
                                      "bytes": target.stat().st_size, "sha256": digest}
    for label, selected in (("gt_v1", gt), ("rf_v2", rf)):
        _require(sha256_file(selected["run"] / "summary.json") == campaign_info[label]["summary_sha256"],
                 f"Source campaign changed during composition: {label}")

    note = ("Derived comparison from two saved sessions: GT results are retained byte-for-byte from gt_v1; "
            "RF-DETR results come from rf_v2. GT was not rerun. No new measurements, statistical pooling, "
            "cross-session round pairing, or claim that all eight pipelines were measured together is made. "
            "Every ratio remains within its originating pipeline and session.")
    protocol = dict(gt["protocol"])
    protocol.update({"name": "Derived eight-pipeline comparison: retained GT v1 and RF-DETR v2",
                     "collection_mode": "derived_composition_no_measurements", "sources": ["gt", "rfdetr_nano"],
                     "sequences": list(SEQUENCES), "clips": len(SEQUENCES), "pipeline_configurations": 8,
                     "composition_note": note, "measurement_sessions_by_source": campaign_info,
                     "round_index_scope": "Round indexes are local to each original session; they are not paired across sources."})
    shared_environment = {key: gt["summary"]["environment"][key]
                          for key in ("cpu", "platform", "python", "executable", "packages")}
    shared_environment.update({"kind": "shared recorded identities; complete per-session environments retained separately",
                               "composition_checks": checks})
    tracking, replay, inputs, validation = [], [], [], {}
    environments = {}
    for label, selected in (("gt_v1", gt), ("rf_v2", rf)):
        origin = {"source_campaign": label, "source_campaign_summary_sha256": campaign_info[label]["summary_sha256"]}
        tracking.extend({**row, **origin} for row in selected["tracking"])
        replay.extend({**row, **origin} for row in selected["replay"])
        inputs.extend(selected["inputs"])
        validation.update({case: selected["summary"]["validation"][case] for case in selected["cases"]})
        environments[selected["source"]] = selected["summary"]["environment"]
    reproducibility = {
        "all_repeated_prediction_hashes_identical": all(item["summary"]["reproducibility"]["all_repeated_prediction_hashes_identical"] for item in (gt, rf)),
        "all_instrumented_predictions_match": all(item["summary"]["reproducibility"]["all_instrumented_predictions_match"] for item in (gt, rf)),
        "unique_evaluated_trajectories": len({(row["case"], row["prediction_sha256"]) for row in tracking}),
        "composition_note": note,
    }
    summary = {"protocol": protocol, "environment": shared_environment,
               "source_environment_by_source": environments,
               "source_protocol_by_source": {item["source"]: item["protocol"] for item in (gt, rf)},
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "timestamp_scope": "Original session timestamps are recorded by source; composition time is not measurement time.",
               "inputs": inputs, "tracking": tracking, "replay": replay, "validation": validation,
               "reproducibility": reproducibility}
    write_json(output / "summary.json", summary)
    write_json(output / "protocol.json", protocol)
    write_json(output / "environment.json", shared_environment)
    write_json(output / "source_environment_by_source.json", environments)
    write_json(output / "tracking_samples.json", combined_samples)
    write_json(output / "composition.json", {"note": note, "campaigns": campaign_info,
               "compatibility_checks": checks, "copied_files": copied,
               "path_policy": "Copied JSON and text evidence is checked for local paths; mappings locate byte-identical copies relative to their source campaigns."})
    own_source = Path(__file__).resolve()
    frozen_source = output / "provenance" / "composition" / own_source.name
    frozen_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(own_source, frozen_source)
    from realbench_report import generate_report
    generate_report(output)
    # Keep the session distinction visible without changing the frozen reporter.
    html_path = output / "report.html"
    html = html_path.read_text(encoding="utf-8")
    from html import escape
    html = html.replace("<div class='scope'>", "<div class='scope'><p><strong>Separate source sessions.</strong> " + escape(note) + "</p>", 1)
    html_path.write_text(html, encoding="utf-8")
    text_path = output / "report.txt"
    text_path.write_text(note + "\n\n" + text_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_json(output / "progress.json", {"phase": "complete_derived_composition", "new_measurements": 0})
    manifest = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(output.rglob("*")) if path.is_file() and path != output / "file_manifest.json"}
    write_json(output / "file_manifest.json", manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-run", type=Path, required=True)
    parser.add_argument("--rf-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    from realbench_paths import redact
    print(f"Derived comparison complete: {redact(str(combine(options.gt_run, options.rf_run, options.output)))}")


if __name__ == "__main__":
    main()
