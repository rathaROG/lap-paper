"""Strict paired summaries and explicitly heuristic campaign budget estimates.

Publication summaries require every declared input and round from one
session per environment. Smoke observations never fill missing cells.
Runtime scenarios describe a scheduling exercise, not solver performance claims
or confidence intervals. This module runs no solvers and allocates no matrices.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from .common import digest_json, write_json
from .catalog import BLOCK_SECONDS, INPUT_COUNT, MEMORY_TRIALS, ROUNDS, WARMUP_CALLS
from .resources import R1, allocation_estimate


def _iqr(values):
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def _read_method_statuses(root):
    """Retain supervisor outcomes even when a method produced no timing file.

    Job statuses describe the final scheduling outcome; attempt snapshots retain
    the history leading to it. Neither is a latency observation or a substitute
    for a successful paired attempt.
    """
    root = Path(root)
    if root.is_file():
        return []
    records = []
    identity_keys = ("campaign_id", "environment_id", "day", "session_id", "scope")
    for summary_path in sorted(root.rglob("summary.json")):
        session = summary_path.parent
        configuration_path = session / "configuration.json"
        if not configuration_path.exists():
            continue
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = {key: configuration.get(key) for key in identity_keys}
        identity_valid = all(isinstance(value, str) and value for value in identity.values())

        def emit(job, method, detail, level, source, attempt=None, valid=True):
            records.append({**identity, "record_type": "method_status", "case_id": job["case_id"],
                            "method": method, "workers": job.get("workers"),
                            "status": detail.get("status", "unavailable"), "detail": detail,
                            "evidence_level": level, "job_status": job.get("status"),
                            "case_host_postponed": job["case_id"] in summary.get("postponed_case_ids", []),
                            "attempt_id": attempt.get("attempt_id") if attempt else None,
                            "attempt_status": attempt.get("status") if attempt else None,
                            "attempt_ids": [item["attempt_id"] for item in job.get("attempts", [])],
                            "_session_metadata_valid": bool(identity_valid and valid),
                            "_record_source": str(source.resolve())})

        for job in summary.get("jobs", []):
            statuses = job.get("statuses", {})
            for method in job.get("methods", statuses):
                detail = statuses.get(method)
                if detail is None:
                    detail = {"status": "unavailable", "reason": "job_has_no_final_method_outcome"}
                else:
                    detail = dict(detail)
                if job.get("failure") is not None:
                    detail["job_failure"] = job["failure"]
                admission_record = job.get("admission", {}).get("methods", {}).get(method)
                if admission_record is not None:
                    detail["admission"] = admission_record
                emit(job, method, detail, "session_job", summary_path)
            for attempt in job.get("attempts", []):
                # Resolve within the supplied session, including relocated archives.
                directory = session / "attempts" / attempt["attempt_id"]
                for filename, level in (("admitted.json", "attempt_admission"),
                                        ("result.json", "attempt_result")):
                    source = directory / filename
                    if not source.exists():
                        continue
                    snapshot = json.loads(source.read_text(encoding="utf-8"))
                    valid = (filename == "admitted.json" or
                             all(snapshot.get(key) == value for key, value in identity.items()) and
                             snapshot.get("attempt_id") == attempt["attempt_id"] and
                             snapshot.get("case_id") == job["case_id"])
                    for method, detail in snapshot.get("statuses", {}).items():
                        emit(job, method, detail, level, source, attempt, valid)
                watchdog = attempt.get("watchdog", {})
                if watchdog.get("reason"):
                    method = watchdog.get("last_state", {}).get("method")
                    if method in job.get("methods", []):
                        emit(job, method, {"status": "attempt_terminated", "watchdog": watchdog},
                             "attempt_watchdog", summary_path, attempt)
    return records


def read_records(root):
    """Read every JSONL once, retaining failed/partial attempts for accounting."""
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    records = []
    for path in paths:
        outcome_path, watchdog_path = path.parent / "result.json", path.parent / "watchdog.json"
        outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else None
        watchdog = json.loads(watchdog_path.read_text(encoding="utf-8")) if watchdog_path.exists() else None
        is_attempt = path.name == "measurements.jsonl"
        accepted = (outcome is not None and outcome.get("status") == "complete"
                    and watchdog is not None and watchdog.get("exit_code") == 0 and not watchdog.get("reason"))
        configuration = None
        postponed_cases = set()
        for ancestor in path.parents:
            config_path = ancestor / "configuration.json"
            if config_path.exists():
                configuration = json.loads(config_path.read_text(encoding="utf-8"))
                summary_path = ancestor / "summary.json"
                if summary_path.exists():
                    postponed_cases = set(json.loads(summary_path.read_text(encoding="utf-8"))
                                          .get("postponed_case_ids", []))
                break
        request_path = path.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8")) if request_path.exists() else None
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Malformed JSONL {path}:{line_number}: {error}") from error
                    item["_record_source"] = str(path.resolve())
                    if is_attempt:
                        item["_case_host_postponed"] = item.get("case_id") in postponed_cases
                        item["_attempt_accepted"] = accepted and not item["_case_host_postponed"]
                        identity_keys = ("campaign_id", "environment_id", "day", "session_id", "scope")
                        identity_valid = (configuration is not None and
                                          all(configuration.get(key) is not None and item.get(key) == configuration[key]
                                              for key in identity_keys))
                        if request is not None:
                            identity_valid = identity_valid and all(item.get(key) == request.get(key)
                                for key in (*identity_keys, "attempt_id"))
                            identity_valid = identity_valid and item.get("case_id") == request.get("case", {}).get("id")
                        if outcome is not None:
                            identity_valid = identity_valid and all(item.get(key) == outcome.get(key)
                                for key in (*identity_keys, "attempt_id", "case_id"))
                        item["_session_metadata_valid"] = identity_valid
                    records.append(item)
    records.extend(_read_method_statuses(root))
    return records


def _reference(method, case, selected):
    if isinstance(selected, str):
        return selected
    if isinstance(selected, dict):
        chosen = selected.get((case.get("id"), method), selected.get(method, selected.get(case.get("id"))))
        if chosen:
            return chosen
    parts = method.split(":")
    if parts[0] == "policy":
        return f"policy:{parts[1]}:reference"
    if parts[0] == "workspace":
        baseline = "w1" if parts[2] == "w2" else "w0"
        return f"workspace:{parts[1]}:{baseline}:{parts[-1]}"
    if parts[0] == "batch":
        if parts[1] == "jvrust":
            return "batch:jvrust:1" if parts[-1] == "4" else "loop:jvrust"
        return f"batch:jvrust:{parts[-1]}"
    if parts[0] == "loop":
        return "loop:jvrust"
    return case.get("metadata", {}).get("primary_comparator", "scipy")


def _placement_workers(method):
    parts = method.split(":")
    if parts[0] in ("batch", "workspace") and parts[-1] in ("1", "4"):
        return int(parts[-1])
    return 1 if parts[0] == "loop" else None


def _pair_estimate(row, control, reference, case):
    keys = ("campaign_id", "environment_id", "case_id", "method")
    result = {key: row[key] for key in keys}
    result["reference"] = reference
    reasons = []
    if row["status"] != "complete":
        reasons.append("method_estimate_unavailable")
    if control is None or control["status"] != "complete":
        reasons.append("reference_estimate_unavailable")
    workers, reference_workers = _placement_workers(row["method"]), _placement_workers(reference)
    cross_placement = (case.get("kind") == "batch" and workers is not None
                       and reference_workers is not None and workers != reference_workers)
    result["pairing"] = "same_session_distinct_placement_processes" if cross_placement else "same_paired_attempt"
    if not reasons:
        if row["session_id"] != control["session_id"] or (
                not cross_placement and row["attempt_id"] != control["attempt_id"]):
            reasons.append("reference_and_method_different_session_or_attempt")
        if cross_placement and row["attempt_id"] == control["attempt_id"]:
            reasons.append("different_placements_share_attempt")
        if row["input_hashes"] != control["input_hashes"]:
            reasons.append("reference_and_method_input_hashes_differ")
    result.update(status="unavailable" if reasons else "complete", reasons=reasons)
    if not reasons:
        result.update(ratio=control["seconds"] / row["seconds"], seconds=row["seconds"],
                      reference_seconds=control["seconds"], input_hashes=row["input_hashes"],
                      session_id=row["session_id"], attempt_id=row["attempt_id"],
                      reference_attempt_id=control["attempt_id"], workers=workers,
                      reference_workers=reference_workers, headline_win=control["seconds"] > row["seconds"])
    return result


def _batch_contrasts(estimates, cases):
    groups = defaultdict(dict)
    keys = ("campaign_id", "environment_id", "case_id")
    for row in estimates:
        if cases.get(row["case_id"], {}).get("kind") == "batch":
            groups[tuple(row[key] for key in keys)][row["method"]] = row
    results = []
    for key, observations in sorted(groups.items()):
        case = cases[key[-1]]
        planned = set(case["methods"])
        specifications = [("batch_overhead", "batch:jvrust:1", "loop:jvrust")]
        specifications += [("worker_count", f"batch:{backend}:4", f"batch:{backend}:1")
                           for backend in ("jvrust", "jvhybrid", "jvcpp", "jvx")]
        for language in ("rust", "cpp"):
            for workers in (1, 4):
                prefix = f"workspace:{language}:"
                specifications += [
                    ("solver_state_lifetime", f"{prefix}w1:{workers}", f"{prefix}w0:{workers}"),
                    ("orientation_buffer_lifetime", f"{prefix}w2:{workers}", f"{prefix}w1:{workers}"),
                    ("total_worker_lifetime", f"{prefix}w2:{workers}", f"{prefix}w0:{workers}"),
                ]
            specifications += [("workspace_worker_count", f"workspace:{language}:{arm}:4",
                                f"workspace:{language}:{arm}:1") for arm in ("w0", "w1", "w2")]
        for label, method, reference in specifications:
            if not {method, reference} <= planned:
                continue
            row = observations.get(method, dict(zip(keys, key)) | {"method": method, "status": "unavailable"})
            result = _pair_estimate(row, observations.get(reference), reference, case)
            result.update(contrast=label, ratio_definition="reference_seconds / method_seconds")
            results.append(result)
    return results


def summarize(records, reference=None, *, expected_inputs=None, expected_rounds=None,
              manifest=None):
    """Use one complete paired session for each environment/case/method.

    Ratios require the same successful session/attempt and input hashes.
    Prespecified one/four-worker contrasts use distinct successful placement
    processes within the same session and retain both attempt identities.
    Two complete attempts for the same condition are ambiguous and unavailable:
    the analyst must declare an attempt selection outside this function, not pick a
    favorable result. Earlier incomplete attempts remain explicitly recorded.
    """
    expected_inputs = (manifest or {}).get("input_count", INPUT_COUNT) if expected_inputs is None else expected_inputs
    expected_rounds = (manifest or {}).get("rounds", ROUNDS) if expected_rounds is None else expected_rounds
    if min(expected_inputs, expected_rounds) < 1:
        raise ValueError("Summary replication requirements must be positive")
    cases = {c["id"]: c for c in (manifest or {}).get("cases", [])}
    grouped = defaultdict(list)
    malformed = []
    method_statuses = []
    keys = ("campaign_id", "environment_id", "session_id", "attempt_id", "case_id", "method")
    for row in records:
        if row.get("record_type") == "method_status":
            method_statuses.append(row)
            continue
        if row.get("record_type") != "timing":
            continue
        if any(not isinstance(row.get(key), str) or not row[key] for key in (*keys, "day")):
            malformed.append({"status": "unavailable", "reason": "missing_timing_identity", "record": row})
            continue
        grouped[tuple(row[key] for key in keys)].append(row)
    attempts = []
    for identity, rows in sorted(grouped.items()):
        item = dict(zip(keys, identity))
        reasons = set()
        day_labels = {row.get("day") for row in rows}
        if len(day_labels) != 1:
            reasons.add("session_day_label_changed")
        else:
            item["day"] = next(iter(day_labels))
        by_cell = {}
        for row in rows:
            if row.get("_attempt_accepted") is False:
                reasons.add("whole_attempt_failed_or_not_finalized")
            if row.get("_case_host_postponed") is True:
                reasons.add("case_host_postponed")
            if row.get("_session_metadata_valid") is False:
                reasons.add("timing_identity_differs_from_session_metadata")
            if row.get("scope") is not None and row["scope"] != "primary_one_environment_session":
                reasons.add("non_primary_scope")
            index, round_number = row.get("input_index"), row.get("round")
            if type(index) is not int or type(round_number) is not int:
                reasons.add("noninteger_input_or_round")
                continue
            cell = (index, round_number)
            if cell in by_cell:
                reasons.add("duplicate_input_round")
            by_cell[cell] = row
            rep, elapsed, mean = row.get("repetitions"), row.get("elapsed_seconds"), row.get("mean_seconds")
            if row.get("status") != "ok":
                reasons.add("unsuccessful_timing_record")
            if type(rep) is not int or rep < 1:
                reasons.add("invalid_repetitions")
            elif not all(isinstance(x, (float, int)) and math.isfinite(x) and x > 0 for x in (elapsed, mean)):
                reasons.add("invalid_elapsed_or_mean")
            elif not math.isclose(mean, elapsed / rep, rel_tol=1e-9, abs_tol=1e-15):
                reasons.add("mean_differs_from_elapsed_over_repetitions")
        expected = {(i, r) for i in range(expected_inputs) for r in range(expected_rounds)}
        if set(by_cell) != expected:
            reasons.add("incomplete_or_unexpected_input_round_grid")
        hashes, repetitions = {}, {}
        for index in range(expected_inputs):
            subset = [row for (i, _), row in by_cell.items() if i == index]
            identities = {row.get("input_sha256") for row in subset}
            counts = {row.get("repetitions") for row in subset}
            if len(identities) != 1 or not all(isinstance(h, str) and h for h in identities):
                reasons.add("input_hash_missing_or_changed")
            else:
                hashes[str(index)] = next(iter(identities))
            if len(counts) != 1:
                reasons.add("repetitions_changed_between_rounds")
            else:
                repetitions[str(index)] = next(iter(counts))
        item.update(status="unavailable" if reasons else "complete", reasons=sorted(reasons),
                    observed_blocks=len(rows), expected_blocks=len(expected),
                    input_hashes=hashes, repetitions=repetitions)
        if not reasons:
            round_means = [statistics.fmean(by_cell[i, r]["mean_seconds"] for i in range(expected_inputs))
                           for r in range(expected_rounds)]
            item.update(seconds=statistics.median(round_means), round_means_seconds=round_means,
                        round_iqr_seconds=_iqr(round_means),
                        inputs=[{"input_index": i, "input_sha256": hashes[str(i)],
                                 "median_seconds": statistics.median(by_cell[i, r]["mean_seconds"] for r in range(expected_rounds)),
                                 "iqr_seconds": _iqr([by_cell[i, r]["mean_seconds"] for r in range(expected_rounds)])}
                                for i in range(expected_inputs)])
        attempts.append(item)
    condition_groups = defaultdict(list)
    condition_keys = ("campaign_id", "environment_id", "case_id", "method")
    for item in attempts:
        condition_groups[tuple(item[key] for key in condition_keys)].append(item)
    estimates = []
    for key, group in sorted(condition_groups.items()):
        complete = [item for item in group if item["status"] == "complete"]
        if len(complete) == 1:
            result = dict(complete[0])
            result["excluded_incomplete_attempts"] = [
                {k: item[k] for k in ("session_id", "attempt_id", "reasons")}
                for item in group if item["status"] != "complete"]
        else:
            result = dict(zip(condition_keys, key))
            result.update(status="unavailable", reasons=["ambiguous_complete_attempts" if complete else "no_complete_attempt"],
                          attempts=[{k: item[k] for k in ("session_id", "attempt_id", "status", "reasons")} for item in group])
        estimates.append(result)
    lookup = {tuple(row[k] for k in condition_keys): row for row in estimates}
    paired = []
    for row in estimates:
        case = cases.get(row["case_id"], {"id": row["case_id"]})
        ref = _reference(row["method"], case, reference)
        control = lookup.get(tuple(ref if key == "method" else row[key] for key in condition_keys))
        paired.append(_pair_estimate(row, control, ref, case))
    contrasts = _batch_contrasts(estimates, cases)
    observed_conditions = {(row["case_id"], row["method"]) for row in attempts}
    planned_conditions = {(case["id"], method) for case in cases.values()
                          if case["kind"] != "memory" for method in case["methods"]}
    outcome_lookup = defaultdict(list)
    for row in method_statuses:
        outcome_lookup[row.get("case_id"), row.get("method")].append(row)
    missing = []
    for case, method in sorted(planned_conditions - observed_conditions):
        outcomes = outcome_lookup[case, method]
        primary_outcomes = [row for row in outcomes if row.get("scope") == "primary_one_environment_session"
                            and row.get("_session_metadata_valid") is True]
        item = {"case_id": case, "method": method, "status": "unavailable",
                "reason": "recorded_outcome_without_timing" if primary_outcomes else "no_timing_records"}
        if outcomes:
            item["recorded_outcomes"] = outcomes
        missing.append(item)
    return {"schema_version": 2,
            "profile": (manifest or {}).get("profile", "full"),
            "manifest_sha256": (manifest or {}).get("definition_sha256"),
            "estimator": "equal-input round mean; median across rounds; paired reference/method ratio from one successful session",
            "requirements": dict(inputs=expected_inputs, rounds=expected_rounds, successful_sessions=1),
            "attempts": attempts, "malformed_records": malformed, "estimates": estimates,
            "method_status_records": method_statuses,
            "paired": paired, "batch_contrasts": contrasts,
            "unobserved_planned_conditions": missing,
            "coverage": {"attempts": len(attempts), "complete_attempts": sum(x["status"] == "complete" for x in attempts),
                         "planned_case_methods": len(planned_conditions), "unobserved_planned_case_methods": len(missing),
                         "comparisons": len(paired),
                         "available_comparisons": sum(x["status"] == "complete" for x in paired)}}


def summarize_memory(session_roots, required_trials=MEMORY_TRIALS):
    """Strict fresh-process memory summaries, separate from latency records.

    Each condition uses one complete session containing each trial exactly once.
    Windows working-set peaks and Linux ru_maxrss remain separate environment strata;
    no peak-counter increment is relabelled as exact allocation size.
    """
    if required_trials < 1:
        raise ValueError("Memory replication requirements must be positive")
    if isinstance(session_roots, (str, Path)):
        session_roots = [session_roots]
    roots = sorted({Path(root).resolve() for root in session_roots})
    identity_keys = ("campaign_id", "environment_id", "day", "session_id", "scope")
    attempts, source_errors = [], []
    for root in roots:
        try:
            configuration = json.loads((root / "configuration.json").read_text(encoding="utf-8"))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            source_errors.append({"session_root": str(root), "reason": f"missing_or_invalid_session_metadata: {error}"})
            continue
        common_reasons = set()
        if any(not isinstance(configuration.get(key), str) or not configuration[key] for key in identity_keys):
            source_errors.append({"session_root": str(root), "reason": "missing_memory_session_identity"})
            continue
        if configuration["scope"] != "primary_one_environment_session":
            common_reasons.add("non_primary_scope")
        if configuration.get("memory_trials") != required_trials:
            common_reasons.add("session_memory_replication_differs")
        stratum = environment.get("stratum")
        expected_environment = (f"{environment.get('system')}-py{environment.get('python')}-"
                                f"{digest_json(stratum)[:16]}") if isinstance(stratum, dict) else None
        if expected_environment != configuration["environment_id"] or environment.get("environment_id") != expected_environment:
            common_reasons.add("environment_stratum_identity_mismatch")
        expected_counter = {"Windows": "peak_working_set", "Linux": "ru_maxrss_bytes"}.get(environment.get("system"))
        if expected_counter is None:
            common_reasons.add("unsupported_memory_counter_OS")
        grouped = defaultdict(list)
        for entry in summary.get("memory", []):
            grouped[entry.get("case_id"), entry.get("method")].append(entry)
        for case in manifest.get("cases", []):
            if case.get("kind") != "memory":
                continue
            for method in case["methods"]:
                reasons = set(common_reasons)
                if case["id"] in summary.get("postponed_case_ids", []):
                    reasons.add("case_host_postponed")
                entries = grouped.get((case["id"], method), [])
                trials, events = [], []
                for entry in entries:
                    if "trial" not in entry:
                        reasons.add("memory_condition_not_admitted")
                        events.append(entry)
                        continue
                    trial = entry["trial"]
                    row = dict(trial=trial, status="unavailable")
                    trial_reasons = []
                    directory_name = str(entry.get("directory", "")).replace("\\", "/").rstrip("/").split("/")[-1]
                    directory = root / "memory" / directory_name
                    row["source_directory"] = str(directory)
                    try:
                        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
                        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
                        watchdog = json.loads((directory / "watchdog.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError) as error:
                        trial_reasons.append("missing_or_invalid_fresh_process_record")
                        row["error"] = str(error)
                        row["watchdog"] = entry.get("watchdog")
                    else:
                        if result.get("status") != "complete" or watchdog.get("exit_code") != 0 or watchdog.get("reason"):
                            trial_reasons.append("fresh_process_failed_or_censored")
                        if any(result.get(key) != configuration[key] or request.get(key) != configuration[key]
                               for key in identity_keys):
                            trial_reasons.append("fresh_process_session_identity_mismatch")
                        if (result.get("case_id"), result.get("method"), result.get("trial")) != (case["id"], method, trial):
                            trial_reasons.append("fresh_process_condition_identity_mismatch")
                        if (request.get("case", {}).get("id"), request.get("method"), request.get("trial")) != (case["id"], method, trial):
                            trial_reasons.append("fresh_process_request_identity_mismatch")
                        if case.get("definition_sha256") != request.get("case", {}).get("definition_sha256"):
                            trial_reasons.append("case_definition_differs_from_manifest")
                        before, after = result.get("before", {}), result.get("after", {})
                        if before.get("counter") != expected_counter or after.get("counter") != expected_counter:
                            trial_reasons.append("counter_definition_differs_from_OS")
                        numeric = [before.get("rss"), before.get("peak"), after.get("rss"), after.get("peak"),
                                   result.get("additional_peak"), result.get("baseline_peak_headroom")]
                        if any(type(value) is not int or value < 0 for value in numeric):
                            trial_reasons.append("invalid_memory_counter_values")
                        elif result["additional_peak"] != max(0, after["peak"] - before["peak"]) or result["baseline_peak_headroom"] != max(0, before["peak"] - before["rss"]):
                            trial_reasons.append("reported_increment_or_headroom_mismatch")
                        elif after["peak"] < before["peak"]:
                            trial_reasons.append("process_peak_counter_decreased")
                        if result.get("hold_seconds") != .050:
                            trial_reasons.append("output_hold_contract_differs")
                        if result.get("validation", {}).get("status") != "passed":
                            trial_reasons.append("memory_output_validation_missing_or_failed")
                        if result.get("input_bytes") != case["rows"] * case["cols"] * 8:
                            trial_reasons.append("resident_input_size_differs")
                        if not isinstance(result.get("input_sha256"), str) or not result["input_sha256"]:
                            trial_reasons.append("input_hash_missing")
                        pid, created = result.get("process_id"), result.get("process_creation_time")
                        if type(pid) is not int or pid < 1 or type(created) not in (int, float) or not math.isfinite(created) or created <= 0:
                            trial_reasons.append("fresh_process_identity_missing_or_invalid")
                        row.update(counter=expected_counter, input_sha256=result.get("input_sha256"),
                                   additional_peak=result.get("additional_peak"), before=before, after=after,
                                   baseline_peak_headroom=result.get("baseline_peak_headroom"), watchdog=watchdog,
                                   process_id=pid, process_creation_time=created)
                    row.update(status="unavailable" if trial_reasons else "complete", reasons=trial_reasons)
                    trials.append(row)
                indices = [row["trial"] for row in trials]
                valid_indices = [index for index in indices if type(index) is int]
                if len(valid_indices) != len(indices) or set(valid_indices) != set(range(required_trials)):
                    reasons.add("incomplete_or_unexpected_fresh_trial_grid")
                if len(valid_indices) != len(set(valid_indices)):
                    reasons.add("duplicate_fresh_trial")
                if any(row["status"] != "complete" for row in trials):
                    reasons.add("unavailable_fresh_trial")
                process_identities = [(row["process_id"], row["process_creation_time"])
                                      for row in trials if row["status"] == "complete"]
                if len(process_identities) != len(set(process_identities)):
                    reasons.add("fresh_trials_reuse_same_process_identity")
                item = {key: configuration[key] for key in identity_keys}
                item.update(case_id=case["id"], method=method, session_root=str(root), counter=expected_counter,
                            status="unavailable" if reasons else "complete", reasons=sorted(reasons),
                            required_trials=required_trials, observed_trials=len(trials), trials=trials,
                            admission_events=events)
                if not reasons:
                    values = [row["additional_peak"] for row in trials]
                    item.update(median_additional_peak_bytes=statistics.median(values),
                                min_additional_peak_bytes=min(values), max_additional_peak_bytes=max(values),
                                input_hashes={str(row["trial"]): row["input_sha256"] for row in trials},
                                baseline_peak_headroom_bytes=[row["baseline_peak_headroom"] for row in trials])
                attempts.append(item)
    condition_groups = defaultdict(list)
    condition_keys = ("campaign_id", "environment_id", "case_id", "method")
    for attempt in attempts:
        condition_groups[tuple(attempt[key] for key in condition_keys)].append(attempt)
    conditions = []
    for key, values in sorted(condition_groups.items()):
        complete = [row for row in values if row["status"] == "complete"]
        if len(complete) == 1:
            item = dict(complete[0])
            item["excluded_incomplete_sessions"] = [row["session_id"] for row in values if row["status"] != "complete"]
        else:
            item = dict(zip(condition_keys, key))
            item.update(status="unavailable", reasons=["ambiguous_complete_memory_sessions" if complete else "no_complete_memory_session"],
                        sessions=[{"session_id": row["session_id"], "status": row["status"], "reasons": row["reasons"]} for row in values])
        conditions.append(item)
    return dict(schema_version=2, requirements=dict(fresh_trials=required_trials, successful_sessions=1),
                estimator="median and observed range of fresh-process increments from one successful session",
                interpretation="OS peak-counter increments, not exact allocated bytes; Windows and Linux remain separate environment strata",
                attempts=attempts, conditions=conditions, source_errors=source_errors,
                coverage=dict(planned_session_conditions=len(attempts),
                              complete_session_conditions=sum(row["status"] == "complete" for row in attempts),
                              planned_environment_conditions=len(conditions),
                              available_environment_conditions=sum(row["status"] == "complete" for row in conditions)))


def method_class(method):
    """Coarse fallback classes only; exact method matches have priority."""
    parts = method.split(":")
    if parts[0] == "policy":
        return "native:" + parts[1]
    if parts[0] == "workspace":
        return f"native_batch:{parts[1]}:{parts[-1]}"
    if parts[0] == "batch":
        lang = {"jvrust": "rust", "jvcpp": "cpp"}.get(parts[1], parts[1])
        return f"native_batch:{lang}:{parts[-1]}"
    if parts[0] == "loop":
        return "loop:" + method_class(":".join(parts[1:]))
    if parts[0] in ("cost", "shape"):
        return f"{parts[0]}:{parts[1]}"
    if parts[0] in ("jvrust", "jvcpp"):
        return "native:" + {"jvrust": "rust", "jvcpp": "cpp"}[parts[0]]
    return method


def _work_dimensions(case, method):
    n, m = case["rows"], case["cols"]
    r, c = sorted((n, m))
    parts = method.split(":")
    backend = parts[1] if parts[0] in ("cost", "shape", "batch", "loop") else parts[0]
    if parts[0] == "cost" or method == "scipy:square":
        r = c = n + m
    elif parts[0] == "shape":
        r = c = n + m if parts[-1].startswith("sum_") else c
    elif method in ("scipy:dummies", "scipy:diagonal_dummies"):
        c += r
    elif backend in ("jvx", "jvs"):
        r = c = n + m if case["objective"] == "optional" else c
    batch = case.get("metadata", {}).get("batch_size", 1) if case["kind"] == "batch" else 1
    workers = int(parts[-1]) if parts[0] in ("workspace", "batch") else 1
    return max(r, 1), max(c, 1), max(batch, 1), workers


def _work_units(case, method, exponent):
    r, c, batch, workers = _work_dimensions(case, method)
    return r ** exponent * c * batch / workers


def _anchors(records, cases):
    grouped = defaultdict(list)
    for row in records:
        value = row.get("mean_seconds")
        if (row.get("record_type") == "timing" and row.get("status") == "ok"
                and row.get("_attempt_accepted") is not False
                and row.get("_session_metadata_valid") is not False
                and row.get("case_id") in cases and isinstance(value, (int, float))
                and math.isfinite(value) and value > 0):
            grouped[row["case_id"], row["method"]].append(row)
    anchors = []
    for (case_id, method), rows in grouped.items():
        by_input = defaultdict(list)
        for row in rows:
            by_input[row["input_index"]].append(row["mean_seconds"])
        means = [statistics.median(values) for values in by_input.values()]
        anchors.append({"case_id": case_id, "method": method,
                        "method_class": method_class(method), "case": cases[case_id],
                        "seconds": statistics.fmean(means), "min_input_seconds": min(means),
                        "max_input_seconds": max(means), "input_count": len(means),
                        "observed_blocks": len(rows),
                        "median_elapsed_block_seconds": statistics.median(row["elapsed_seconds"] for row in rows)})
    return anchors


def _select_anchor(case, method, anchors):
    if not anchors:
        return None, "unanchored"
    def score(anchor):
        other = anchor["case"]
        distance = abs(math.log2(_work_units(case, method, 1.5) /
                                  _work_units(other, anchor["method"], 1.5)))
        return (not (other["id"] == case["id"] and anchor["method"] == method),
                other["kind"] != case["kind"], other["objective"] != case["objective"],
                anchor["method"] != method, anchor["method_class"] != method_class(method),
                other["family"] != case["family"], distance,
                (other["rows"] > other["cols"]) != (case["rows"] > case["cols"]))
    anchor = min(anchors, key=score)
    other = anchor["case"]
    if case["id"] == other["id"] and method == anchor["method"]:
        quality = "observed_case_method"
    elif (case["kind"], case["objective"], method, case["family"]) == (
            other["kind"], other["objective"], anchor["method"], other["family"]):
        quality = "same_method_family"
    elif (case["kind"], case["objective"], method_class(method)) == (
            other["kind"], other["objective"], anchor["method_class"]):
        quality = "same_method_class_objective"
    else:
        quality = "weak_cross_class_or_objective"
    return anchor, quality


def _runtime_scenarios(case, method, anchor, quality):
    if anchor is None:
        return [.001, .03, 1.0]  # Explicit unanchored planning assumptions.
    if quality == "observed_case_method":
        return [anchor["min_input_seconds"], anchor["seconds"], anchor["max_input_seconds"]]
    central = anchor["seconds"]
    predictions = [central * _work_units(case, method, exponent) /
                   _work_units(anchor["case"], anchor["method"], exponent)
                   for exponent in (1.0, 1.5, 2.0)]
    uncertainty = {"same_method_family": 2., "same_method_class_objective": 4.,
                   "weak_cross_class_or_objective": 10.}[quality]
    return [min(predictions) / uncertainty, predictions[1], max(predictions) * uncertainty]


def _summary_jobs(root):
    result = []
    root = Path(root)
    for path in sorted(root.rglob("summary.json")) if root.is_dir() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        result.extend(data.get("jobs", []))
    return result


def _memory_observations(root):
    observations = []
    root = Path(root)
    for path in sorted(root.rglob("summary.json")) if root.is_dir() else []:
        summary = json.loads(path.read_text(encoding="utf-8"))
        for trial in summary.get("memory", []):
            result = trial.get("result") or {}
            elapsed = trial.get("case_seconds")
            if result.get("status") == "complete" and isinstance(elapsed, (int, float)) and elapsed > 0:
                observations.append({**trial, "elapsed_seconds": elapsed,
                                     "solve_seconds": result.get("elapsed_seconds", 0.)})
    return observations


def estimate(manifest, smoke_root, *, environments=1, block_seconds=None,
             input_count=None, rounds=None, memory_trials=None, warmup_calls=None):
    """Produce scheduling scenarios, preserving unmeasured/censored coverage.

    The default is one environment session. Current smoke hardware is an
    anchor only; low/high factors are sensitivity scenarios, not statistical
    or guaranteed bounds. Full solver calls can exceed the target block.
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    input_count = manifest.get("input_count", INPUT_COUNT) if input_count is None else input_count
    rounds = manifest.get("rounds", ROUNDS) if rounds is None else rounds
    block_seconds = manifest.get("target_block_seconds", BLOCK_SECONDS) if block_seconds is None else block_seconds
    memory_trials = manifest.get("memory_trials", MEMORY_TRIALS) if memory_trials is None else memory_trials
    warmup_calls = manifest.get("warmup_calls", WARMUP_CALLS) if warmup_calls is None else warmup_calls
    if min(environments, input_count, rounds, memory_trials) < 1 or block_seconds <= 0:
        raise ValueError("Invalid runtime-estimate replication settings")
    cases = {case["id"]: case for case in manifest["cases"]}
    records = read_records(smoke_root)
    anchors = _anchors(records, cases)
    jobs = _summary_jobs(smoke_root)
    sessions = environments
    phase_records = [row for row in records if row.get("record_type") == "phase"
                     and isinstance(row.get("elapsed_seconds"), (int, float))
                     and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0]
    phase_totals = defaultdict(float)
    for row in phase_records:
        phase_totals[row.get("phase", "unspecified")] += row["elapsed_seconds"]
    # Primary phase records duplicate timing elapsed values by design. They are
    # counted only here, never added to the timing elapsed sum.
    timed_total = sum(row["elapsed_seconds"] for row in records
                      if row.get("record_type") == "timing" and row.get("status") == "ok")
    all_phase_total = sum(phase_totals.values())
    calibration_ratios = []
    for anchor in anchors:
        matching = [row for row in phase_records if row.get("phase") == "calibration"
                    and row.get("case_id") == anchor["case_id"] and row.get("method") == anchor["method"]]
        if matching:
            per_input = sum(row["elapsed_seconds"] for row in matching) / max(1, len({row.get("input_index") for row in matching}))
            calibration_ratios.append(per_input / max(anchor["median_elapsed_block_seconds"], 1e-9))
    calibration_blocks = max(1., statistics.median(calibration_ratios)) if calibration_ratios else 2.
    elapsed_jobs = [job["case_seconds"] for job in jobs
                    if isinstance(job.get("case_seconds"), (float, int)) and job["case_seconds"] >= 0]
    overhead_per_job = (max(0., sum(elapsed_jobs) - all_phase_total) / len(elapsed_jobs)) if elapsed_jobs else 1.
    # Missing phase capture must not turn measured compute into alleged startup.
    overhead_basis = "observed_case_elapsed_minus_all_phases" if elapsed_jobs and phase_records else "unanchored_1_second_per_job"
    if overhead_basis.startswith("unanchored"):
        overhead_per_job = 1.
    memory_conditions = sum(len(case["methods"]) for case in cases.values() if case["kind"] == "memory")
    planned = sum(len(case["methods"]) for case in cases.values() if case["kind"] != "memory")
    packages = defaultdict(lambda: {"planned_jobs_per_session": 0, "model_admitted_jobs_per_session": 0,
                                    "model_excluded_jobs_per_session": 0,
                                    "low_seconds": 0., "central_seconds": 0., "high_seconds": 0.})
    totals = [0., 0., 0.]
    excluded, predictions, predicted_screen_censors = [], [], [0, 0, 0]
    anchor_counts = Counter()
    case_workers = set()
    admitted = 0
    for case in cases.values():
        if case["kind"] == "memory":
            continue
        for method in case["methods"]:
            package = ("workspace" if method.startswith("workspace:") else
                       "policy" if method.startswith("policy:") else
                       "sparse" if method.startswith("sparse:") else
                       "layout" if method.startswith(("cost:", "shape:")) else
                       "geometry" if case["family"] == "geometry" else
                       "batch" if case["kind"] == "batch" else "legacy")
            group = packages[package]
            group["planned_jobs_per_session"] += 1
            resources = allocation_estimate(case, method, input_count)
            reasons = []
            if resources["largest_cost_array_bytes"] > R1["array_bytes"]:
                reasons.append("individual_array_limit")
            if resources["estimated_peak_bytes"] > R1["admission_bytes"]:
                reasons.append("working_set_admission_limit")
            if reasons:
                group["model_excluded_jobs_per_session"] += 1
                excluded.append({"case_id": case["id"], "method": method, "reasons": reasons})
                continue
            group["model_admitted_jobs_per_session"] += 1
            admitted += 1
            case_workers.add((case["id"], resources["workers"]))
            anchor, quality = _select_anchor(case, method, anchors)
            anchor_counts[quality] += 1
            per_call = _runtime_scenarios(case, method, anchor, quality)
            active = []
            calibration_factors = [max(1., calibration_blocks / 2), calibration_blocks, max(4., 2 * calibration_blocks)]
            for scenario, seconds in enumerate(per_call):
                if seconds > R1["screen_seconds"]:
                    # A deterministic first-input screening timeout prevents
                    # primary measurement; do not fill its cells with a cap.
                    cost = R1["screen_seconds"]
                    predicted_screen_censors[scenario] += 1
                else:
                    block = max(1, math.ceil(block_seconds / seconds)) * seconds
                    cost = input_count * ((1 + warmup_calls) * seconds + calibration_factors[scenario] * block + rounds * block)
                    cost = min(cost, R1["method_active_seconds"])
                active.append(cost * sessions)
                totals[scenario] += cost * sessions
                group[("low_seconds", "central_seconds", "high_seconds")[scenario]] += cost * sessions
            predictions.append({"case_id": case["id"], "method": method, "package": package,
                                "anchor_quality": quality,
                                "anchor_case_id": anchor["case_id"] if anchor else None,
                                "anchor_method": anchor["method"] if anchor else None,
                                "call_seconds_low_central_high": per_call,
                                "scheduled_seconds_low_central_high": active})
    # Setup/validation/input preparation and process launch are outside primary
    # blocks. These deliberately broad factors are not claimed as measurements.
    observed_block_count = sum(row.get("record_type") == "timing" and row.get("status") == "ok" for row in records)
    observed_blocks_per_job = observed_block_count / len(elapsed_jobs) if elapsed_jobs else 0.
    full_blocks_per_job = admitted * input_count * rounds / max(1, len(case_workers))
    residual_work_ratio = max(1., full_blocks_per_job / observed_blocks_per_job) if observed_blocks_per_job else 1.
    # Residual elapsed contains both fixed interpreter startup and per-block
    # validation/JSON overhead. Smoke cannot identify their split reliably.
    overhead = [len(case_workers) * sessions * overhead_per_job * factor *
                ((1 - fraction) + fraction * residual_work_ratio)
                for factor, fraction in zip((.5, 1., 4.), (0., .25, 1.))]
    reference_seconds = [0., 0., 0.]
    for case_id, _ in case_workers:
        case = cases[case_id]
        comparator = "scipy:capped" if case["objective"] == "optional" else "scipy"
        anchor, quality = _select_anchor(case, comparator, anchors)
        calls = _runtime_scenarios(case, comparator, anchor, quality)
        for i in range(3):
            reference_seconds[i] += min(calls[i], R1["screen_seconds"]) * input_count * sessions
    memory_rows = _memory_observations(smoke_root)
    if memory_rows:
        overheads = [max(0., row["elapsed_seconds"] - row["solve_seconds"]) for row in memory_rows]
        memory_per_trial = [min(overheads) / 2, statistics.median(overheads), max(overheads) * 4]
        memory_basis = "observed_fresh_process_elapsed_minus_solve_plus_extrapolated_solve"
    else:
        # Startup, input preparation, solve, output holding and teardown together.
        memory_per_trial = [.25, 2., 10.]
        memory_basis = "unanchored_declared_0.25_2_10_seconds_process_overhead_plus_extrapolated_solve"
    memory_processes = memory_conditions * memory_trials * sessions
    memory_seconds = [0., 0., 0.]
    for case in cases.values():
        if case["kind"] != "memory":
            continue
        proxy = {**case, "kind": "single"}
        for method in case["methods"]:
            anchor, quality = _select_anchor(proxy, method, anchors)
            call_seconds = _runtime_scenarios(proxy, method, anchor, quality)
            for i in range(3):
                memory_seconds[i] += (memory_per_trial[i] + min(call_seconds[i], R1["screen_seconds"])) * memory_trials * sessions
    # Explanatory diagnostics are a single pass per environment. Their event
    # counts cannot become primary timing samples; their elapsed budget is
    # separately modelled.
    prediction_lookup = {(item["case_id"], item["method"]): item for item in predictions}
    diagnostic_packages = {}
    for category, prefix in (("policy_counters", "policy:"), ("workspace_capacity", "workspace:")):
        conditions = [(case, method) for case in cases.values() for method in case["methods"]
                      if method.startswith(prefix)]
        active = [0., 0., 0.]
        charged = 0
        for case, method in conditions:
            prediction = prediction_lookup.get((case["id"], method))
            if prediction is None:
                continue  # The allocation model already excluded this condition.
            charged += 1
            for i, multiplier in enumerate((1., 2., 8.)):
                seconds = min(prediction["call_seconds_low_central_high"][i] * multiplier, R1["screen_seconds"])
                active[i] += 2 * seconds * input_count * environments
        diagnostic_packages[category] = {
            "planned_conditions": len(conditions), "model_admitted_conditions": charged,
            "planned_counted_calls": len(conditions) * input_count * environments,
            "planned_calls": 2 * len(conditions) * input_count * environments,
            "warmup_calls_per_input": 1, "counted_calls_per_input": 1,
            "passes_per_environment": 1, "environments": environments,
            "seconds_low_central_high": active,
            "basis": "One warmup plus one counted call/input; heuristic 1x/2x/8x instrumentation overhead on projected complete calls, not measured diagnostic latency."}
    layouts = {}
    for case in cases.values():
        for method in case["methods"]:
            parts = method.split(":")
            if parts[0] in ("cost", "shape"):
                layouts.setdefault((case["id"], parts[0], parts[-1]), (case, method))
    construction = [0., 0., 0.]
    construction_admitted = 0
    for case, method in layouts.values():
        prediction = prediction_lookup.get((case["id"], method))
        if prediction is None:
            continue
        construction_admitted += 1
        rows, cols, _, _ = _work_dimensions(case, method)
        bytes_written = rows * cols * 8
        for i, (fraction, assumed_gib_per_second) in enumerate(zip((.01, .1, 1.), (20., 5., 1.))):
            seconds = max(prediction["call_seconds_low_central_high"][i] * fraction,
                          bytes_written / (assumed_gib_per_second * 1024 ** 3))
            construction[i] += min(seconds, R1["screen_seconds"]) * 3 * input_count * environments
    diagnostic_packages["layout_construction"] = {
        "planned_conditions": len(layouts), "model_admitted_conditions": construction_admitted,
        "planned_calls": len(layouts) * 3 * input_count * environments,
        "repeats_per_input": 3, "passes_per_environment": 1, "environments": environments,
        "seconds_low_central_high": construction,
        "basis": "Backend-deduplicated layouts; heuristic 1%/10%/100% of complete-call projection, at least one output write at assumed 20/5/1 GiB/s. These rates are assumptions, not measurements."}
    diagnostic_reference = [0., 0., 0.]
    diagnostic_case_workers = {(case["id"], allocation_estimate(case, method, input_count)["workers"])
                               for case in cases.values() for method in case["methods"]
                               if method.startswith(("policy:", "workspace:", "cost:", "shape:"))}
    for case_id, _ in diagnostic_case_workers:
        case = cases[case_id]
        method = "scipy:capped" if case["objective"] == "optional" else "scipy"
        anchor, quality = _select_anchor(case, method, anchors)
        times = _runtime_scenarios(case, method, anchor, quality)
        for i, validation_factor in enumerate((1., 1.5, 3.)):
            diagnostic_reference[i] += min(times[i] * validation_factor, R1["screen_seconds"]) * input_count * environments
    diagnostic_packages["reference_and_validation"] = {
        "planned_conditions": len(diagnostic_case_workers),
        "planned_calls": len(diagnostic_case_workers) * input_count * environments,
        "passes_per_environment": 1, "environments": environments,
        "seconds_low_central_high": diagnostic_reference,
        "basis": "One independent comparator/input/diagnostic process plus heuristic 1x/1.5x/3x allowance for objective checks; setup also charged separately."}
    diagnostic_seconds = [sum(item["seconds_low_central_high"][i] for item in diagnostic_packages.values()) for i in range(3)]
    # Diagnostic child startup/input preparation is additional to the one-call
    # event pass; reuse the explicitly uncertain primary setup anchor.
    diagnostic_processes = len(diagnostic_case_workers) * environments
    diagnostic_setup = [diagnostic_processes * overhead_per_job * factor for factor in (.5, 1., 4.)]
    diagnostic_seconds = [diagnostic_seconds[i] + diagnostic_setup[i] for i in range(3)]
    schedule = [totals[i] + overhead[i] + reference_seconds[i] + memory_seconds[i] + diagnostic_seconds[i] for i in range(3)]
    return {"schema_version": 2, "purpose": "runtime planning only; smoke is not publication evidence",
            "profile": manifest.get("profile", "full"),
            "manifest_sha256": manifest.get("definition_sha256"),
            "design": dict(environments=environments, sessions=sessions,
                           inputs=input_count, rounds=rounds, target_block_seconds=block_seconds,
                           memory_trials=memory_trials),
            "coverage": dict(planned_timing_case_methods=planned, model_admitted_timing_case_methods=admitted,
                             model_excluded_timing_case_methods=len(excluded),
                             timing_case_worker_processes_per_session=len(case_workers),
                             memory_conditions=memory_conditions, memory_fresh_processes=memory_processes),
            "nominal_primary_target_seconds_before_censoring": planned * input_count * rounds * block_seconds * sessions,
            "nominal_primary_target_seconds_after_model_admission": admitted * input_count * rounds * block_seconds * sessions,
            "nominal_floor_note": "Calibration-target accounting, not a hard elapsed-time lower bound: observed blocks vary and complete calls may be longer. No censoring is hidden in the before-censoring figure.",
            "observed_smoke": {"anchored_case_methods": len(anchors), "timing_block_elapsed_seconds": timed_total,
                               "phase_elapsed_seconds": dict(phase_totals), "case_elapsed_seconds": sum(elapsed_jobs),
                               "environment_ids": sorted({row.get("environment_id", "unknown") for row in records}),
                               "primary_phase_and_timing_not_double_counted": True},
            "assumptions": {"scaling": "same-method/family anchors first; reading r*c, intermediate r^1.5*c, and bound r^2*c scenarios use transformed solver dimensions and batch/worker counts",
                            "multiplicative_shape_uncertainty": {"same_method_family": 2, "same_method_class_objective": 4, "weak_cross_class_or_objective": 10},
                            "unanchored_call_seconds": [.001, .03, 1.],
                            "screen_and_warmup": f"one screening call and {warmup_calls} warmup calls per input plus separately estimated calibration and {rounds} primary blocks",
                            "calibration_equivalent_blocks": calibration_blocks,
                            "setup_overhead_basis": overhead_basis, "setup_seconds_per_job_anchor": overhead_per_job,
                            "residual_smoke_to_full_block_ratio": residual_work_ratio,
                            "residual_dynamic_fraction_low_central_high": [0., .25, 1.],
                            "primary_reference_calls": "One excluded independent comparator call per input/case/placement; modelled separately from primary-method screens",
                            "memory_basis": memory_basis, "memory_overhead_seconds_per_process_low_central_high": memory_per_trial,
                            "cross_environment": "the smoke host's scale applied to every configuration; environment differences remain unmeasured",
                            "limits": "Scenarios are not confidence intervals or guaranteed bounds. Large sizes, hard inputs, cold memory trials, validation, calibration and unobserved method classes can depart substantially."},
            "anchor_quality_counts": dict(anchor_counts), "work_packages": dict(packages),
            "timing_active_seconds_low_central_high": totals,
            "setup_validation_seconds_low_central_high": overhead,
            "primary_reference_seconds_low_central_high": reference_seconds,
            "memory_seconds_low_central_high": memory_seconds,
            "diagnostic_work_packages": diagnostic_packages,
            "diagnostic_processes": diagnostic_processes,
            "diagnostic_setup_seconds_low_central_high": diagnostic_setup,
            "diagnostic_seconds_low_central_high": diagnostic_seconds,
            "total_scheduled_seconds_low_central_high": schedule,
            "session_hours_low_central_high": [x / 3600 / sessions for x in schedule],
            "session_note": "One session per environment; sessions run sequentially on one host.",
            "predicted_screen_censors_low_central_high": predicted_screen_censors,
            "censoring_note": "Predicted timeouts are scheduling scenarios, not observed solver failures. High per-call scenarios may finish the schedule sooner because R1 screens them out. Host headroom and runtime failures remain additional exclusions.",
            "model_resource_exclusions": excluded, "predictions": predictions}


def estimate_markdown(report):
    coverage = report["coverage"]
    low, central, high = report["total_scheduled_seconds_low_central_high"]
    lines = ["# Campaign runtime budget", "", "Planning estimate only; no full campaign has run.", "",
             f"The manifest plans {coverage['planned_timing_case_methods']:,} timing case/method conditions per session. "
             f"The deterministic allocation model admits {coverage['model_admitted_timing_case_methods']:,} and excludes "
             f"{coverage['model_excluded_timing_case_methods']:,}; actual host headroom and screening remain to be checked.", "",
             f"Across {report['design']['sessions']} session(s), the nominal {1000*report['design']['target_block_seconds']:g} ms primary-block budget before censoring is "
             f"{report['nominal_primary_target_seconds_before_censoring']/3600:.2f} hours. "
             "This target accounting excludes screening, warmup, calibration, setup and memory, and is not a guaranteed elapsed-time lower bound.", "",
             f"Low / central / high scheduling scenarios: **{low/3600:.2f} / {central/3600:.2f} / {high/3600:.2f} active hours**. "
             "These are heuristic sensitivity scenarios, not confidence intervals. A high-cost scenario can screen out more methods and therefore need less total scheduling time.", "",
             "| Work package | Planned conditions/session | Admitted | Low hours | Central hours | High hours |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, group in sorted(report["work_packages"].items()):
        lines.append(f"| {name} | {group['planned_jobs_per_session']:,} | {group['model_admitted_jobs_per_session']:,} | "
                     f"{group['low_seconds']/3600:.2f} | {group['central_seconds']/3600:.2f} | {group['high_seconds']/3600:.2f} |")
    lines += ["", f"Memory adds {coverage['memory_conditions']} conditions × {report['design']['memory_trials']} fresh processes × {report['design']['sessions']} sessions = "
              f"{coverage['memory_fresh_processes']:,} processes. Basis: {report['assumptions']['memory_basis']}.", "",
              "Diagnostics are a separate one-pass-per-environment work package:", "",
              "| Diagnostic work | Conditions | Calls across environments | Low hours | Central hours | High hours |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, group in report["diagnostic_work_packages"].items():
        times = group["seconds_low_central_high"]
        lines.append(f"| {name} | {group['planned_conditions']:,} | {group['planned_calls']:,} | "
                     f"{times[0]/3600:.2f} | {times[1]/3600:.2f} | {times[2]/3600:.2f} |")
    lines += ["", "Counter overhead and construction rates are explicit planning assumptions, not benchmark data. "
              "Layout construction is counted once per case/layout, with three repeats per input, deduplicating identical backend construction. "
              "Additional diagnostic setup is included in total scheduling time.", "",
              "Observed smoke time and extrapolation are separate in estimate.json. Anchor coverage: " +
              ", ".join(f"{name}={count:,}" for name, count in sorted(report["anchor_quality_counts"].items())) + ".", "",
              "The model uses transformed solver dimensions, equal input weighting, complete-call blocks, screening and warmup calls per input, "
              "calibration, process/setup allowance and memory trials. Large inputs, weakly matched methods and other environments remain uncertain. "
              "The low/high shape scenarios scale dense reading and the cubic assignment bound; they do not predict the realized algorithmic path.", "",
              "Hours per environment session (low / central / high): " + " / ".join(
                  f"{x:.2f}" for x in report["session_hours_low_central_high"]) + ".", ""]
    return "\n".join(lines)


def write_estimate(manifest, smoke_root, output=None, **kwargs):
    report = estimate(manifest, smoke_root, **kwargs)
    from .common import digest_file
    report["estimator_source_sha256"] = digest_file(Path(__file__))
    output = Path(output or smoke_root)
    write_json(output / "estimate.json", report)
    (output / "estimate.md").write_text(estimate_markdown(report), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("summarize", "estimate"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.action == "estimate":
        report = write_estimate(manifest, args.input, args.output)
        print(json.dumps({"active_hours_low_central_high": [x / 3600 for x in report["total_scheduled_seconds_low_central_high"]],
                          "coverage": report["coverage"]}, indent=2))
    else:
        report = summarize(read_records(args.input), args.reference, manifest=manifest)
        write_json(args.output / "analysis.json", report)
        print(json.dumps(report["coverage"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
