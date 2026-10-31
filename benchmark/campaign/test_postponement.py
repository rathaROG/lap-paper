"""Transient admission failures retain pending work without taking measurements."""
from copy import deepcopy
import json

import pytest

from . import __main__ as cli, native_build, runner, worker
from .test_failures import supervisor


def _case(name="case", kind="single", methods=None):
    return {"id": name, "kind": kind, "methods": methods or ["jvrust"],
            "rows": 2, "cols": 3, "limit": None, "gate": None, "objective": "required"}


def _run(monkeypatch, cases, headroom, responses=None):
    """Use the real supervisor loop with the in-memory supervisor fixture."""
    available = iter(headroom)
    launches = []
    environment = runner.environment
    monkeypatch.setattr(runner, "environment", lambda: {**environment(), "python": "3.11.18"})
    monkeypatch.setattr(runner, "local_day", lambda: "2026-10-30")
    monkeypatch.setattr(worker, "local_day", lambda: "2026-10-30")

    def admission(case, methods, input_count):
        enough = next(available)
        return {"host_status": "ready" if enough else "postponed",
                "methods": {method: {"status": "admitted" if enough else "postponed",
                                     "reasons": [] if enough else ["host_headroom_unavailable"]}
                            for method in methods}}

    def launch(request, module):
        launches.append((module, deepcopy(request)))
        if responses is not None:
            return deepcopy(responses[len(launches) - 1])
        result = {"status": "complete"}
        if module == "worker":
            result.update(statuses={method: {"status": "complete"} for method in request["methods"]},
                          measurements_path="successful.jsonl", input_hashes=["same-input"])
        return result, {"status": "exited", "exit_code": 0, "reason": None}

    monkeypatch.setattr(runner, "admission", admission)
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "attempt_activity", lambda *args: {})
    manifest = {"definition_sha256": "manifest", "cases": cases}
    return runner.run_session(manifest, "virtual-postponed", smoke=False, selected_cases=cases), launches


@pytest.mark.parametrize("kind", ["single", "memory"])
def test_host_pressure_stops_shard_and_retains_current_and_later_cases(supervisor, monkeypatch, kind):
    summary, launches = _run(monkeypatch, [_case(kind=kind), _case("later")], [False])
    assert summary["status"] == "postponed"
    assert summary["remaining_case_ids"] == ["case", "later"]
    assert summary["postponed_case_ids"] == ["case"]
    assert summary["postponement"]["reason"] == "host_headroom_unavailable"
    assert launches == []


def test_postponement_keeps_earlier_completed_cases_valid(supervisor, monkeypatch):
    cases = [_case("earlier"), _case("interrupted"), _case("later")]
    summary, launches = _run(monkeypatch, cases, [True, False])
    assert len(launches) == 1
    assert summary["remaining_case_ids"] == ["interrupted", "later"]
    assert summary["postponed_case_ids"] == ["interrupted"]
    assert [job["status"] for job in summary["jobs"]] == ["complete", "postponed"]


def test_postponement_between_placements_invalidates_entire_case(supervisor, monkeypatch):
    case = _case(kind="batch", methods=["batch:jvrust:1", "batch:jvrust:4"])
    summary, launches = _run(monkeypatch, [case, _case("later")], [True, False])
    assert [request["methods"] for _, request in launches] == [["batch:jvrust:1"]]
    assert [job["status"] for job in summary["jobs"]] == ["postponed", "postponed"]
    assert summary["jobs"][0]["attempts"][0]["status"] == "complete"  # Raw outcome retained.
    assert summary["remaining_case_ids"] == ["case", "later"]
    assert summary["postponed_case_ids"] == ["case"]


def test_headroom_is_rechecked_between_fresh_memory_trials(supervisor, monkeypatch):
    summary, launches = _run(monkeypatch, [_case(kind="memory"), _case("later")], [True, True, False])
    assert [request["trial"] for _, request in launches] == [0, 1]
    assert len(summary["memory"]) == 3  # Two retained trials plus the admission failure.
    assert all(entry["result"]["status"] == "complete" for entry in summary["memory"][:2])
    assert summary["memory"][-1]["admission"]["host_status"] == "postponed"
    assert summary["status"] == "postponed"
    assert summary["remaining_case_ids"] == ["case", "later"]


def test_headroom_is_rechecked_before_restarting_a_paired_attempt(supervisor, monkeypatch):
    failure = {"status": "restart_required", "offender": "jvrust", "statuses": {
        "jvrust": {"status": "correctness_failed"}, "jvcpp": {"status": "ready"}}}
    watchdog = {"status": "exited", "exit_code": 0, "reason": None}
    case = _case(methods=["jvrust", "jvcpp"])
    summary, launches = _run(monkeypatch, [case], [True, False], [(failure, watchdog)])
    assert len(launches) == 1
    assert summary["status"] == "postponed"
    assert summary["postponed_case_ids"] == summary["remaining_case_ids"] == ["case"]
    job = summary["jobs"][0]
    assert job["statuses"]["jvrust"]["status"] == "correctness_failed"
    assert job["statuses"]["jvcpp"]["status"] == "postponed"
    assert job["admission_rechecks"][0]["host_status"] == "postponed"


@pytest.mark.parametrize("status,expected", [("postponed", 2), ("finished", 1)])
def test_memory_only_postponement_never_exits_success(tmp_path, monkeypatch, status, expected):
    monkeypatch.setattr(cli, "build_manifest", lambda: {"cases": [_case(kind="memory")]})
    monkeypatch.setattr(cli, "validation_report", lambda *args: {"correctness_failed": 0, "unsupported": 0})
    monkeypatch.setattr(native_build, "ALL_ARMS", ())
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"builds": []}), encoding="utf-8")
    summary = {"scope": "primary_one_environment_session", "wall_seconds": 0., "status": status,
               "remaining_case_ids": ["case"] if status == "postponed" else [], "jobs": [],
               "memory": [{"case_id": "case", "method": "jvrust", "admission": {
                   "methods": {"jvrust": {"status": "postponed"}}}}]}
    monkeypatch.setattr(runner, "run_session", lambda *args, **kwargs: summary)
    assert cli.main(["run", "--allow-full-campaign", "--build-index", str(index),
                     "--output", str(tmp_path / "output")]) == expected
