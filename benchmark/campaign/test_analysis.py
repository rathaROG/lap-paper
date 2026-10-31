import json
import math

import pytest

from .analysis import estimate, estimate_markdown, read_records
from .analysis import summarize, summarize_memory
from .catalog import BLOCK_SECONDS, INPUT_COUNT, MEMORY_TRIALS, ROUNDS, WARMUP_CALLS
from .common import digest_json


def _records(day="2026-10-30", attempt="a0", session="session", ratio=2.):
    result = []
    for method, multiplier in [("reference", ratio), ("candidate", 1.)]:
        for i in range(INPUT_COUNT):
            for r in range(ROUNDS):
                mean = (i + 1) * multiplier * (1 if r < ROUNDS // 2 + 1 else 3)
                repetitions = 1000 if i == 0 else 1
                result.append(dict(record_type="timing", campaign_id="test-campaign", environment_id="test-env", day=day,
                                   session_id=session, attempt_id=attempt, case_id="test", method=method,
                                   input_index=i, input_sha256=f"hash-{i}", round=r,
                                   repetitions=repetitions, elapsed_seconds=mean * repetitions,
                                   mean_seconds=mean, status="ok"))
    return result


def test_equal_input_round_means_then_median_are_not_repetition_weighted():
    result = summarize(_records(), "reference")
    candidate = next(x for x in result["estimates"] if x["method"] == "candidate")
    mean = (INPUT_COUNT + 1) / 2
    majority = ROUNDS // 2 + 1
    assert candidate["seconds"] == mean
    assert candidate["round_means_seconds"] == [mean] * majority + [3 * mean] * (ROUNDS - majority)
    assert candidate["inputs"][0]["median_seconds"] == 1.
    assert candidate["inputs"][0]["iqr_seconds"] == 2.
    paired = next(row for row in result["paired"] if row["method"] == "candidate")
    assert paired["status"] == "complete" and paired["ratio"] == 2.
    assert result["requirements"] == dict(inputs=INPUT_COUNT, rounds=ROUNDS, successful_sessions=1)


@pytest.mark.parametrize("second_day", ["2026-10-30", "2026-10-31"])
def test_complete_sessions_are_ambiguous_regardless_of_date(second_day):
    records = _records(session="first", ratio=.5) + _records(day=second_day, session="second", ratio=3.)
    result = summarize(records, "reference")
    assert all(row["reasons"] == ["ambiguous_complete_attempts"] for row in result["estimates"])
    assert all(row["status"] == "unavailable" and "ratio" not in row for row in result["paired"])


@pytest.mark.parametrize("defect", ["duplicate", "missing", "hash", "repetitions", "mean", "scope", "failed_attempt"])
def test_invalid_attempt_is_unavailable(defect):
    rows = _records()
    if defect == "duplicate":
        rows.append(rows[-1].copy())
    elif defect == "missing":
        rows.pop()
    elif defect == "hash":
        rows[-1]["input_sha256"] = "changed"
    elif defect == "repetitions":
        rows[-1]["repetitions"] *= 2
        rows[-1]["elapsed_seconds"] *= 2
    elif defect == "mean":
        rows[-1]["mean_seconds"] *= 2
    elif defect == "scope":
        for row in rows:
            row["scope"] = "smoke_not_publication_evidence"
    else:
        for row in rows:
            row["_attempt_accepted"] = False
    result = summarize(rows, "reference")
    candidate = next(x for x in result["estimates"] if x["method"] == "candidate")
    assert candidate["status"] == "unavailable"
    assert "seconds" not in candidate


def test_attempts_never_stitched_and_duplicate_complete_attempts_are_ambiguous():
    first, second = _records(attempt="a0"), _records(attempt="a1")
    midpoint = ROUNDS // 2
    partial = [row for row in first if row["round"] < midpoint] + [row for row in second if row["round"] >= midpoint]
    result = summarize(partial, "reference")
    assert all(x["status"] == "unavailable" for x in result["paired"])
    result = summarize(first + second, "reference")
    assert all(x["reasons"] == ["ambiguous_complete_attempts"] for x in result["estimates"])


def test_pair_requires_same_attempt_and_shared_hashes():
    rows = _records()
    for row in rows:
        if row["method"] == "candidate":
            row["attempt_id"] = "different"
    paired = summarize(rows, "reference")["paired"]
    candidate = next(x for x in paired if x["method"] == "candidate")
    assert candidate["reasons"] == ["reference_and_method_different_session_or_attempt"]


def _batch_records():
    methods = {"loop:jvrust": (1, 12.), "batch:jvrust:1": (1, 8.),
               "batch:jvrust:4": (4, 2.),
               "workspace:rust:w0:1": (1, 3.), "workspace:rust:w1:1": (1, 6.),
               "workspace:rust:w2:1": (1, 12.)}
    result = []
    for method, (workers, seconds) in methods.items():
        for i in range(INPUT_COUNT):
            for r in range(ROUNDS):
                result.append(dict(record_type="timing", campaign_id="c", environment_id="e",
                                   day="2026-10-30", session_id="session", attempt_id=f"case-w{workers}-a0", case_id="b",
                                   method=method, input_index=i, round=r, input_sha256=f"hash{i}",
                                   repetitions=1, elapsed_seconds=seconds, mean_seconds=seconds, status="ok"))
    manifest = {"cases": [dict(id="b", kind="batch", methods=list(methods), metadata={})]}
    return result, manifest


def test_batch_cross_placement_ratios_keep_both_successful_attempts():
    rows, manifest = _batch_records()
    result = summarize(rows, manifest=manifest)
    worker = next(row for row in result["batch_contrasts"] if row["contrast"] == "worker_count")
    assert worker["status"] == "complete"
    assert worker["ratio"] == 4.
    assert worker["attempt_id"] == "case-w4-a0"
    assert worker["reference_attempt_id"] == "case-w1-a0"
    overhead = next(row for row in result["batch_contrasts"] if row["contrast"] == "batch_overhead")
    assert overhead["ratio"] == 1.5
    lifetime = {row["contrast"]: row for row in result["batch_contrasts"] if "lifetime" in row["contrast"]}
    assert lifetime["solver_state_lifetime"]["ratio"] == .5
    assert lifetime["orientation_buffer_lifetime"]["ratio"] == .5
    assert lifetime["total_worker_lifetime"]["ratio"] == .25


@pytest.mark.parametrize("defect", ["session", "hash", "same_attempt", "incomplete"])
def test_cross_placement_requires_same_session_inputs_and_complete_processes(defect):
    rows, manifest = _batch_records()
    for row in rows:
        if row["method"] != "batch:jvrust:4":
            continue
        if defect == "session":
            row["session_id"] += "-other"
        elif defect == "hash":
            row["input_sha256"] += "-other"
        elif defect == "same_attempt":
            row["attempt_id"] = "case-w1-a0"
        elif row["round"] == ROUNDS - 1:
            row["status"] = "failed"
    result = summarize(rows, manifest=manifest)
    comparisons = [row for row in result["batch_contrasts"] if row["contrast"] == "worker_count"]
    assert len(comparisons) == 1
    assert all(row["status"] == "unavailable" for row in comparisons)


def test_read_records_checks_whole_attempt_outcome(tmp_path):
    measurements = tmp_path / "measurements.jsonl"
    rows = _records()
    for row in rows:
        row["scope"] = "primary_one_environment_session"
    identity = {key: rows[0][key] for key in ("campaign_id", "environment_id", "day", "session_id", "scope", "attempt_id", "case_id")}
    (tmp_path / "configuration.json").write_text(json.dumps(identity), encoding="utf-8")
    measurements.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps({**identity, "status": "restart_required"}), encoding="utf-8")
    (tmp_path / "watchdog.json").write_text(json.dumps({"exit_code": 0, "reason": None}), encoding="utf-8")
    report = summarize(read_records(tmp_path), "reference")
    assert all(x["status"] == "unavailable" for x in report["estimates"])
    (tmp_path / "result.json").write_text(json.dumps({**identity, "status": "complete"}), encoding="utf-8")
    report = summarize(read_records(tmp_path), "reference")
    assert all(x["status"] == "complete" for x in report["estimates"])


def _status_session(tmp_path, name, statuses, *, scope="primary_one_environment_session", timing=False,
                    day="2026-10-30", case_id="test"):
    root = tmp_path / name
    directory = root / "attempts" / "a0"
    directory.mkdir(parents=True)
    identity = dict(campaign_id="test-campaign", environment_id="test-env", day=day,
                    session_id=name, scope=scope)
    outcome = {**identity, "attempt_id": "a0", "case_id": case_id, "status": "complete",
               "statuses": statuses}
    watchdog = dict(exit_code=0, reason=None)
    job = dict(case_id=case_id, workers=1, methods=list(statuses), statuses=statuses, status="complete",
               attempts=[dict(attempt_id="a0", status="complete", watchdog=watchdog)])
    (root / "configuration.json").write_text(json.dumps(identity), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(dict(jobs=[job])), encoding="utf-8")
    (directory / "result.json").write_text(json.dumps(outcome), encoding="utf-8")
    (directory / "watchdog.json").write_text(json.dumps(watchdog), encoding="utf-8")
    if timing:
        rows = _records(day=day, session=name)
        for row in rows:
            row["scope"] = scope
            row["case_id"] = case_id
        (directory / "measurements.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return root


@pytest.mark.parametrize("status", ["unsupported", "correctness_failed", "resource_censored", "resource_excluded"])
def test_pre_timing_outcomes_survive_without_any_jsonl(tmp_path, status):
    detail = dict(status=status, reason="retained specific reason")
    root = _status_session(tmp_path, status, {"candidate": detail})
    manifest = {"cases": [dict(id="test", kind="single", methods=["candidate", "unattempted"])]}
    report = summarize(read_records(root), manifest=manifest)
    assert report["attempts"] == [] and report["paired"] == []
    missing = {row["method"]: row for row in report["unobserved_planned_conditions"]}
    assert missing["candidate"]["reason"] == "recorded_outcome_without_timing"
    assert missing["unattempted"]["reason"] == "no_timing_records"
    outcomes = missing["candidate"]["recorded_outcomes"]
    assert all(row["status"] == status and row["detail"] == detail for row in outcomes)
    assert {row["attempt_id"] for row in outcomes} == {None, "a0"}
    assert all(row["environment_id"] == "test-env" and row["day"] == "2026-10-30" for row in outcomes)


def test_admission_and_infrastructure_details_survive_without_attempt(tmp_path):
    root = _status_session(tmp_path, "admission", {"candidate": {"status": "resource_excluded"}})
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    job = summary["jobs"][0]
    job.update(attempts=[], status="incomplete", statuses={},
               admission={"methods": {"candidate": {"status": "postponed", "reasons": ["host_headroom_unavailable"]}}},
               failure={"error_kind": "infrastructure", "reason": "cannot inspect thread"})
    path.write_text(json.dumps(summary), encoding="utf-8")
    report = summarize(read_records(root))
    row, = report["method_status_records"]
    assert row["status"] == "unavailable"
    assert row["detail"]["admission"]["reasons"] == ["host_headroom_unavailable"]
    assert row["detail"]["job_failure"]["reason"] == "cannot inspect thread"
    assert row["attempt_id"] is None


def test_status_history_preserves_discarded_attempt_and_successful_retry(tmp_path):
    statuses = {"reference": {"status": "complete"}, "candidate": {"status": "complete"},
                "rejected": {"status": "resource_censored", "reason": "primary_timeout"}}
    root = _status_session(tmp_path, "retry", statuses, timing=True)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    discarded = root / "attempts" / "discarded"
    discarded.mkdir()
    watchdog = dict(exit_code=-9, reason="primary_timeout", last_state={"method": "rejected"})
    (discarded / "watchdog.json").write_text(json.dumps(watchdog), encoding="utf-8")
    (discarded / "admitted.json").write_text(json.dumps({"statuses": {
        "reference": {"status": "ready"}, "candidate": {"status": "ready"}, "rejected": {"status": "ready"}}}), encoding="utf-8")
    rows = _records(session="retry", attempt="discarded")[:4]
    for row in rows:
        row["scope"] = "primary_one_environment_session"
    (discarded / "measurements.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    summary["jobs"][0]["attempts"].insert(0, dict(attempt_id="discarded", status="failed", watchdog=watchdog))
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = summarize(read_records(root), "reference")
    assert all(row["status"] == "complete" and row["attempt_id"] == "a0" for row in report["estimates"])
    assert any(row["status"] == "unavailable" and row["attempt_id"] == "discarded" for row in report["attempts"])
    terminal = next(row for row in report["method_status_records"] if row["evidence_level"] == "attempt_watchdog")
    assert terminal["method"] == "rejected" and terminal["attempt_id"] == "discarded"
    assert terminal["detail"]["watchdog"]["reason"] == "primary_timeout"


def test_smoke_statuses_are_retained_without_becoming_primary_failures(tmp_path):
    root = _status_session(tmp_path, "smoke", {"candidate": {"status": "correctness_failed", "reason": "smoke only"}},
                           scope="smoke_not_publication_evidence")
    manifest = {"cases": [dict(id="test", kind="single", methods=["candidate"])]}
    report = summarize(read_records(root), manifest=manifest)
    assert report["unobserved_planned_conditions"][0]["reason"] == "no_timing_records"
    assert all(row["scope"] == "smoke_not_publication_evidence" for row in report["method_status_records"])
    assert report["estimates"] == []


@pytest.mark.parametrize("retry_day", ["2026-10-30", "2026-10-31"])
def test_postponed_case_invalidates_earlier_placement_and_allows_retry(tmp_path, retry_day):
    statuses = {"reference": {"status": "complete"}, "candidate": {"status": "complete"}}
    first = _status_session(tmp_path, "postponed", statuses, timing=True)
    path = first / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(status="postponed", postponed_case_ids=["test"])
    path.write_text(json.dumps(summary), encoding="utf-8")
    retry = _status_session(tmp_path, "clean-retry", statuses, timing=True, day=retry_day)
    report = summarize(read_records(first) + read_records(retry), "reference")
    assert all(row["status"] == "complete" and row["session_id"] == "clean-retry" for row in report["estimates"])
    excluded = [row for row in report["attempts"] if row["session_id"] == "postponed"]
    assert all(row["status"] == "unavailable" and "case_host_postponed" in row["reasons"] for row in excluded)


def test_disjoint_case_shards_can_use_different_sessions_and_dates(tmp_path):
    statuses = {"reference": {"status": "complete"}, "candidate": {"status": "complete"}}
    first = _status_session(tmp_path, "first-shard", statuses, timing=True, case_id="first")
    second = _status_session(tmp_path, "second-shard", statuses, timing=True,
                             day="2026-10-31", case_id="second")
    report = summarize(read_records(first) + read_records(second), "reference")
    comparisons = [row for row in report["paired"] if row["method"] == "candidate"]
    assert {row["case_id"] for row in comparisons} == {"first", "second"}
    assert all(row["status"] == "complete" and row["ratio"] == 2. for row in comparisons)


def test_postponement_does_not_discard_an_earlier_completed_case(tmp_path):
    statuses = {"reference": {"status": "complete"}, "candidate": {"status": "complete"}}
    root = _status_session(tmp_path, "earlier-case", statuses, timing=True)
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(status="postponed", postponed_case_ids=["later-case"])
    path.write_text(json.dumps(summary), encoding="utf-8")
    report = summarize(read_records(root), "reference")
    assert all(row["status"] == "complete" for row in report["estimates"])


def test_campaign_identity_required_and_different_generations_never_pool():
    rows = _records()
    for row in rows:
        del row["campaign_id"]
    result = summarize(rows, "reference")
    assert len(result["malformed_records"]) == len(rows)
    rows = []
    for number, day in enumerate(("2026-10-28", "2026-10-29", "2026-10-30")):
        current = _records(day=day, session=day)
        for row in current:
            row["campaign_id"] = f"campaign-{number}"
        rows += current
    result = summarize(rows, "reference")
    assert len(result["paired"]) == 6
    assert {row["campaign_id"] for row in result["paired"]} == {"campaign-0", "campaign-1", "campaign-2"}
    assert all(row["status"] == "complete" for row in result["paired"])


def test_summary_cli_and_estimate_cli_use_actual_apis(tmp_path, monkeypatch):
    from . import __main__ as cli
    monkeypatch.setattr(cli, "build_manifest", _manifest)
    smoke = tmp_path / "smoke"
    smoke.mkdir()
    estimate_output = tmp_path / "estimate"
    assert cli.main(["estimate", "--smoke-root", str(smoke), "--output", str(estimate_output)]) == 0
    assert (estimate_output / "estimate.json").exists()
    assert (estimate_output / "estimate.md").exists()
    summary_output = tmp_path / "analysis.json"
    assert cli.main(["summarize", str(smoke), "--output", str(summary_output)]) == 0
    assert json.loads(summary_output.read_text())["coverage"]["unobserved_planned_case_methods"] == 2


def test_light_summary_uses_three_rounds_and_still_rejects_missing_cells():
    manifest = {**_manifest(), "profile": "light", "input_count": 5, "rounds": 3}
    records = [row for row in _records() if row["round"] < 3]
    report = summarize(records, "reference", manifest=manifest)
    assert report["profile"] == "light"
    assert report["requirements"]["rounds"] == 3 and report["requirements"]["inputs"] == 5
    assert all(row["status"] == "complete" for row in report["paired"])
    assert next(row for row in report["paired"] if row["method"] == "candidate")["ratio"] == 2.
    incomplete = summarize(records[:-1], "reference", manifest=manifest)
    assert next(row for row in incomplete["paired"] if row["method"] == "candidate")["status"] == "unavailable"


def test_derived_cli_reports_inherit_saved_light_manifest(tmp_path):
    from . import __main__ as cli
    session = tmp_path / "session"
    session.mkdir()
    manifest = {**_manifest(), "profile": "light", "input_count": 5, "rounds": 3}
    (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (session / "records.jsonl").write_text("\n".join(json.dumps(row) for row in _records() if row["round"] < 3))
    output = tmp_path / "summary.json"
    assert cli.main(["summarize", str(session), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["requirements"]["rounds"] == 3
    assert all(row["status"] == "complete" for row in report["estimates"])
    estimate_output = tmp_path / "estimate"
    assert cli.main(["estimate", "--smoke-root", str(session), "--output", str(estimate_output)]) == 0
    projection = json.loads((estimate_output / "estimate.json").read_text())
    assert projection["profile"] == "light"
    assert projection["design"]["rounds"] == 3
    assert projection["nominal_primary_target_seconds_before_censoring"] == pytest.approx(2 * 5 * 3 * BLOCK_SECONDS)


def test_summary_cli_rejects_mixed_profiles_before_writing_output(tmp_path):
    from . import __main__ as cli
    sessions = [tmp_path / name for name in ("full", "light")]
    for path in sessions:
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps({**_manifest(), "profile": path.name}))
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(SystemExit):
        cli.main(["summarize", *map(str, sessions), "--output", str(output)])
    with pytest.raises(SystemExit):
        cli.main(["summarize", str(sessions[1]), "--profile", "full", "--output", str(output)])
    assert not output.exists()


def _memory_session(tmp_path, day, *, host="host-a", trials=MEMORY_TRIALS,
                    scope="primary_one_environment_session", suffix="", case_id="mem"):
    root = tmp_path / (day + "-" + host + suffix)
    root.mkdir()
    stratum = dict(hostname=host, system="Windows", python="3.11.18", binaries="same")
    env_id = "Windows-py3.11.18-" + digest_json(stratum)[:16]
    environment = dict(system="Windows", python="3.11.18", environment_id=env_id, stratum=stratum)
    identity = dict(campaign_id="memory-generation", environment_id=env_id, day=day,
                    session_id=root.name, scope=scope)
    configuration = {**identity, "memory_trials": MEMORY_TRIALS}
    case = dict(id=case_id, kind="memory", rows=2, cols=3, objective="required", limit=None,
                methods=["scipy", "jvrust"], definition_sha256="frozen-case")
    (root / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"cases": [case]}), encoding="utf-8")
    (root / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    entries = []
    for trial in range(trials):
        directory = root / "memory" / f"condition-{trial}"
        directory.mkdir(parents=True)
        result = {**identity, "trial": trial, "case_id": case_id, "method": "scipy", "status": "complete",
                  "process_id": 1000+trial, "process_creation_time": 10000.+trial,
                  "input_sha256": f"frozen-{trial}", "input_bytes": 48,
                  "before": dict(rss=90, peak=100, counter="peak_working_set"),
                  "after": dict(rss=100+trial, peak=110+trial, counter="peak_working_set"),
                  "additional_peak": 10+trial, "baseline_peak_headroom": 10,
                  "hold_seconds": .050, "validation": {"status": "passed"}}
        request = {**identity, "trial": trial, "case": case, "method": "scipy"}
        watchdog = dict(exit_code=0, reason=None)
        for name, payload in (("result", result), ("request", request), ("watchdog", watchdog)):
            (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
        entries.append(dict(case_id=case_id, method="scipy", trial=trial, directory=str(directory), result=result, watchdog=watchdog))
    (root / "summary.json").write_text(json.dumps({"memory": entries}), encoding="utf-8")
    return root


def test_memory_uses_declared_fresh_trials_from_one_session(tmp_path):
    root = _memory_session(tmp_path, "2026-10-30")
    result = summarize_memory([root])
    scipy = next(row for row in result["conditions"] if row["method"] == "scipy")
    assert scipy["status"] == "complete"
    assert scipy["median_additional_peak_bytes"] == 10 + (MEMORY_TRIALS - 1) / 2
    assert scipy["min_additional_peak_bytes"] == 10
    assert scipy["max_additional_peak_bytes"] == 10 + MEMORY_TRIALS - 1
    assert scipy["counter"] == "peak_working_set"
    assert next(row for row in result["conditions"] if row["method"] == "jvrust")["status"] == "unavailable"
    assert result["requirements"] == dict(fresh_trials=MEMORY_TRIALS, successful_sessions=1)
    assert result["coverage"]["planned_session_conditions"] == 2


@pytest.mark.parametrize("defect", ["missing", "duplicate", "censored", "counter", "identity", "smoke", "input_hash", "same_process"])
def test_memory_invalid_fresh_process_or_scope_stays_unavailable(tmp_path, defect):
    root = _memory_session(tmp_path, "2026-10-30", trials=MEMORY_TRIALS - 1 if defect == "missing" else MEMORY_TRIALS,
                           scope="smoke_not_publication_evidence" if defect == "smoke" else "primary_one_environment_session")
    if defect == "duplicate":
        path = root / "summary.json"
        summary = json.loads(path.read_text())
        summary["memory"].append(summary["memory"][0])
        path.write_text(json.dumps(summary))
    elif defect == "censored":
        (root / "memory/condition-0/watchdog.json").write_text(json.dumps(dict(exit_code=-1, reason="memory_limit")))
    elif defect in ("counter", "identity", "input_hash", "same_process"):
        path = root / "memory/condition-0/result.json"
        record = json.loads(path.read_text())
        if defect == "counter":
            record["after"]["counter"] = "ru_maxrss_bytes"
        elif defect == "identity":
            record["environment_id"] = "different-host"
        elif defect == "same_process":
            record["process_id"] = 1001
            record["process_creation_time"] = 10001.
        else:
            del record["input_sha256"]
        path.write_text(json.dumps(record))
    result = summarize_memory([root])
    scipy = next(row for row in result["attempts"] if row["method"] == "scipy")
    assert scipy["status"] == "unavailable"
    assert "median_additional_peak_bytes" not in scipy


def test_memory_PID_reuse_with_distinct_creation_times_is_still_fresh(tmp_path):
    root = _memory_session(tmp_path, "2026-10-30")
    for directory in (root / "memory").iterdir():
        path = directory / "result.json"
        record = json.loads(path.read_text())
        record["process_id"] = 123
        path.write_text(json.dumps(record))
    result = summarize_memory([root])
    assert next(row for row in result["attempts"] if row["method"] == "scipy")["status"] == "complete"


def test_memory_keeps_different_hosts_as_separate_environments(tmp_path):
    first = _memory_session(tmp_path, "2026-10-28")
    other = _memory_session(tmp_path, "2026-10-29", host="host-b")
    result = summarize_memory([first, other])
    complete = [row for row in result["conditions"] if row["status"] == "complete"]
    assert len(complete) == 2
    assert len({row["environment_id"] for row in complete}) == 2


@pytest.mark.parametrize("second_day", ["2026-10-30", "2026-10-31"])
def test_memory_complete_sessions_are_ambiguous_regardless_of_date(tmp_path, second_day):
    first = _memory_session(tmp_path, "2026-10-30")
    duplicate = _memory_session(tmp_path, second_day, suffix="-retry")
    result = summarize_memory([first, duplicate])
    scipy = next(row for row in result["conditions"] if row["method"] == "scipy")
    assert scipy["reasons"] == ["ambiguous_complete_memory_sessions"]
    assert "median_additional_peak_bytes" not in scipy


@pytest.mark.parametrize("retry_day", ["2026-10-30", "2026-10-31"])
def test_postponed_memory_case_keeps_trials_but_does_not_conflict_with_clean_retry(tmp_path, retry_day):
    first = _memory_session(tmp_path, "2026-10-30")
    path = first / "summary.json"
    summary = json.loads(path.read_text())
    summary.update(status="postponed", postponed_case_ids=["mem"])
    path.write_text(json.dumps(summary), encoding="utf-8")
    retry = _memory_session(tmp_path, retry_day, suffix="-clean-retry")
    report = summarize_memory([first, retry])
    initial = [row for row in report["attempts"] if row["session_root"] == str(first.resolve())]
    assert initial and all(row["status"] == "unavailable" and "case_host_postponed" in row["reasons"] for row in initial)
    assert next(row for row in initial if row["method"] == "scipy")["observed_trials"] == MEMORY_TRIALS
    recovered = next(row for row in report["conditions"] if row["method"] == "scipy")
    assert recovered["status"] == "complete" and recovered["session_root"] == str(retry.resolve())


def test_memory_disjoint_case_shards_keep_the_complete_session_for_each_case(tmp_path):
    first = _memory_session(tmp_path, "2026-10-30", case_id="first")
    second = _memory_session(tmp_path, "2026-10-31", case_id="second")
    cases = [json.loads((root / "manifest.json").read_text())["cases"][0] for root in (first, second)]
    for root in (first, second):
        (root / "manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    report = summarize_memory([first, second])
    complete = [row for row in report["conditions"] if row["status"] == "complete"]
    assert {row["case_id"] for row in complete} == {"first", "second"}
    assert all(row["method"] == "scipy" and len(row["excluded_incomplete_sessions"]) == 1 for row in complete)


def _manifest():
    single = dict(id="test", rows=16, cols=20, kind="single", family="test",
                  objective="required", methods=["scipy", "policy:rust:reference"],
                  metadata={"primary_comparator": "scipy"})
    memory = dict(id="memory", rows=16, cols=20, kind="memory", family="memory",
                  objective="required", methods=["scipy"], metadata={})
    return {"cases": [single, memory], "definition_sha256": "frozen"}


def test_estimator_counts_replication_and_never_double_counts_primary_phase(tmp_path):
    rows = []
    for i in range(2):
        for r in range(2):
            rows += [dict(record_type="timing", case_id="test", method="scipy", input_index=i,
                          round=r, status="ok", mean_seconds=.001, elapsed_seconds=.005),
                     dict(record_type="phase", case_id="test", method="scipy", input_index=i,
                          phase="primary", elapsed_seconds=.005)]
    (tmp_path / "smoke.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    report = estimate(_manifest(), tmp_path)
    assert report["schema_version"] == 2
    assert report["design"]["sessions"] == 1
    assert "distinct_days_per_environment" not in report["design"]
    assert report["coverage"]["planned_timing_case_methods"] == 2
    assert report["coverage"]["memory_fresh_processes"] == MEMORY_TRIALS
    assert report["nominal_primary_target_seconds_before_censoring"] == pytest.approx(2 * INPUT_COUNT * ROUNDS * BLOCK_SECONDS)
    assert report["observed_smoke"]["timing_block_elapsed_seconds"] == pytest.approx(.02)
    assert report["observed_smoke"]["phase_elapsed_seconds"]["primary"] == pytest.approx(.02)
    assert len(report["predictions"]) == 2
    assert report["anchor_quality_counts"]["observed_case_method"] == 1
    assert report["diagnostic_work_packages"]["policy_counters"]["planned_calls"] == 2 * INPUT_COUNT
    assert report["diagnostic_work_packages"]["policy_counters"]["planned_counted_calls"] == INPUT_COUNT
    assert report["diagnostic_work_packages"]["policy_counters"]["passes_per_environment"] == 1
    assert f"{MEMORY_TRIALS} processes" in estimate_markdown(report)


def test_estimator_empty_smoke_is_explicitly_unanchored(tmp_path):
    report = estimate(_manifest(), tmp_path)
    assert report["observed_smoke"]["anchored_case_methods"] == 0
    assert report["anchor_quality_counts"] == {"unanchored": 2}
    assert report["assumptions"]["memory_basis"].startswith("unanchored")
    assert all(math.isfinite(x) and x > 0 for x in report["total_scheduled_seconds_low_central_high"])


def test_missing_planned_conditions_remain_explicit():
    report = summarize([], manifest=_manifest())
    assert report["coverage"]["planned_case_methods"] == 2
    assert len(report["unobserved_planned_conditions"]) == 2
