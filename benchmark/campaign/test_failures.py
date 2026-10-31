"""Harness failures are injected in memory: no solvers, affinity, or processes.

These tests execute the supervisor/worker control flow with fake calls and
clocks, including failures occurring after useful partial observations exist.
"""
from copy import deepcopy
from pathlib import PurePosixPath
from types import SimpleNamespace

import numpy as np
import pytest

from . import adapters, datasets, runner, topology, worker
from .resources import R1


class MemoryPath(PurePosixPath):
    files = {}
    directories = set()

    def resolve(self):
        return self

    def exists(self):
        return str(self) in self.files or str(self) in self.directories

    def mkdir(self, *args, **kwargs):
        self.directories.add(str(self))

    def iterdir(self):
        return iter(())

    def glob(self, pattern):
        return iter(())

    def write_text(self, value, **kwargs):
        self.files[str(self)] = value
        return len(value)

    def read_text(self, **kwargs):
        return self.files[str(self)]


@pytest.fixture
def supervisor(monkeypatch):
    """Run the real retry loop against in-memory process results."""
    MemoryPath.files = {}
    MemoryPath.directories = set()
    storage = MemoryPath.files
    affinity_calls = []
    fake_process = SimpleNamespace(cpu_affinity=lambda mask=None:
                                   list(range(9)) if mask is None else affinity_calls.append(list(mask)))
    import psutil
    monkeypatch.setattr(psutil, "Process", lambda: fake_process)
    monkeypatch.setattr(runner, "Path", MemoryPath)
    monkeypatch.setattr(runner, "write_json", lambda path, value: storage.__setitem__(str(path), deepcopy(value)))
    monkeypatch.setattr(runner, "read_json", lambda path: deepcopy(storage[str(path)]))
    monkeypatch.setattr(runner, "discover", lambda: {"recommended_batch_cpus": [0, 2, 4, 6]})
    monkeypatch.setattr(runner, "validate_selection", lambda *args:
                        [{"logical_cpus": [i, i+1]} for i in (0, 2, 4, 6)])
    monkeypatch.setattr(runner, "environment", lambda: {
        "system": "Windows", "python": "3.11.17", "platform": "mock", "machine": "mock", "hostname": "mock",
        "logical_cpus": 9, "physical_memory_bytes": 64*1024**3, "packages": {},
        "native_artifact_hashes": {}, "system_settings": {}})
    class Lock:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(runner, "HostLock", Lock)
    calls, activity = [], []

    def run(responses, *, methods=("bad", "good", "unsupported"), excluded=()):
        def admit(case, methods, input_count):
            return {"methods": {m: {"status": "resource_excluded" if m in excluded else "admitted"}
                                for m in methods}}
        monkeypatch.setattr(runner, "admission", admit)
        def launch(request, module):
            calls.append(deepcopy(request))
            assert module == "worker"
            response = responses[len(calls)-1]
            directory = MemoryPath(request["directory"])
            if "admitted" in response:
                storage[str(directory / "admitted.json")] = deepcopy(response["admitted"])
            if "fatal" in response:
                storage[str(directory / "fatal.json")] = deepcopy(response["fatal"])
            activity.append(response.get("activity", {}))
            return deepcopy(response.get("result")), deepcopy(response["watchdog"])
        monkeypatch.setattr(runner, "launch", launch)
        monkeypatch.setattr(runner, "attempt_activity", lambda *args: activity[-1])
        case = {"id": "case", "kind": "single", "methods": list(methods), "rows": 2, "cols": 3,
                "limit": None, "gate": None, "objective": "required"}
        manifest = {"campaign_id": "test-campaign", "definition_sha256": "manifest", "cases": [case]}
        result = runner.run_session(manifest, "virtual-output", smoke=True, selected_cases=[case])
        return result
    return SimpleNamespace(run=run, calls=calls, storage=storage, affinity_calls=affinity_calls)


def terminated(reason="memory_limit", method="bad"):
    return {"status": "terminated", "reason": reason, "exit_code": -9,
            "last_state": {"method": method, "phase": "primary", "monotonic": 0}}


def completed(methods=("good",), path="attempt-1-only.jsonl"):
    return {"result": {"status": "complete", "offender": None,
                       "statuses": {m: {"status": "complete"} for m in methods},
                       "measurements_path": path, "input_hashes": ["input0", "input1"]},
            "watchdog": {"status": "exited", "exit_code": 0, "reason": None, "last_state": {}}}


def test_memory_offender_is_never_retried_and_only_worker_admitted_methods_restart(supervisor):
    first = {"watchdog": terminated(), "activity": {"bad": 2., "good": 3.},
             "admitted": {"methods": ["bad", "good"],
                          "statuses": {"bad": {"status": "ready"}, "good": {"status": "ready"},
                                       "unsupported": {"status": "unsupported", "reason": "missing backend"}}}}
    second = {**completed(), "activity": {"good": 4.}}
    summary = supervisor.run([first, second], methods=("bad", "good", "unsupported", "excluded"), excluded=("excluded",))
    assert [r["methods"] for r in supervisor.calls] == [["bad", "good", "unsupported"], ["good"]]
    assert supervisor.calls[0]["attempt_id"] != supervisor.calls[1]["attempt_id"]
    assert supervisor.calls[1]["active_seed"] == {"bad": 2., "good": 3.}
    job = summary["jobs"][0]
    assert job["active_seconds"] == {"bad": 2., "good": 7.}
    assert job["statuses"]["bad"] == {"status": "resource_censored", "reason": "memory_limit"}
    assert job["statuses"]["unsupported"]["status"] == "unsupported"
    assert job["statuses"]["excluded"]["status"] == "resource_excluded"
    assert job["measurements_path"] == "attempt-1-only.jsonl"
    assert len(job["attempts"]) == 2
    # Supervisor avoids the measured cores' SMT siblings and restores its mask.
    assert supervisor.affinity_calls == [[8], list(range(9))]


def test_completed_partial_attempt_is_not_stitched_into_restart(supervisor):
    failed = {"result": {"status": "restart_required", "offender": "bad",
                          "statuses": {"bad": {"status": "correctness_failed"},
                                       "good": {"status": "ready"}, "unsupported": {"status": "unsupported"}},
                          "measurements_path": "invalid-partial.jsonl"},
              "watchdog": {"status": "exited", "reason": None, "exit_code": 0},
              "activity": {"good": 5., "bad": 1.}}
    result = supervisor.run([failed, completed()])
    assert supervisor.calls[1]["methods"] == ["good"]
    job = result["jobs"][0]
    assert job["measurements_path"] != "invalid-partial.jsonl"
    assert job["statuses"]["bad"]["status"] == "correctness_failed"
    assert job["attempts"][0]["status"] == "restart_required"


@pytest.mark.parametrize("reason", [None, "resource_monitor_error", "prepare_timeout", "validate_timeout"])
def test_infrastructure_failure_is_not_a_solver_failure_or_a_retry(supervisor, reason):
    first = {"watchdog": terminated(reason),
             "fatal": {"status": "worker_failed", "reason": "OSError: cannot write state"}}
    result = supervisor.run([first])
    assert len(supervisor.calls) == 1
    job = result["jobs"][0]
    assert job["status"] == "incomplete"
    assert job["failure"]["status"] == "worker_failed"
    assert "bad" not in job["statuses"]


@pytest.fixture
def harness(monkeypatch):
    """The real worker with deterministic fake solver calls and a fake clock."""
    import threadpoolctl
    states, logs, calls = {}, [], []
    current = {"phase": None, "method": None}
    clock = {"ns": 0}
    def tick():
        clock["ns"] += 1_000_000
        return clock["ns"]
    monkeypatch.setattr(worker, "time", SimpleNamespace(perf_counter_ns=tick, monotonic=lambda: clock["ns"]/1e9))
    monkeypatch.setattr(worker, "Path", MemoryPath)
    def write(path, value):
        states[str(path)] = deepcopy(value)
        if str(path).endswith("state.json"):
            current.update(value)
    monkeypatch.setattr(worker, "write_json", write)
    monkeypatch.setattr(worker, "append_jsonl", lambda path, value: logs.append(deepcopy(value)))
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda **kwargs: None)
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [])
    monkeypatch.setattr(topology, "apply_and_verify", lambda cpus: {"cpus": cpus})
    monkeypatch.setattr(topology, "verify", lambda cpus: {"cpus": cpus})
    monkeypatch.setattr(datasets, "generate_input", lambda spec, index:
                        np.array([[0., 1., 2.], [2., 0., 1.]]))
    fault = {"function": None}
    def prepare(method, spec, index):
        def call(cost):
            calls.append((method, current["phase"], current.get("input_index")))
            if method == "bad" and fault["function"]:
                fault["function"](cost, current)
            return np.array([[0, 0], [1, 1]], dtype=np.int64)
        call.method_info = {"method": method}
        return call
    monkeypatch.setattr(adapters, "prepare_method", prepare)
    request = {"campaign_id": "test-campaign", "environment_id": "mock-environment", "day": "2026-10-30",
               "session_id": "session", "attempt_id": "attempt", "scope": "unit-test-no-measurements",
               "directory": "virtual-worker", "cpus": [0], "inputs": 2, "rounds": 2,
               "warmup_calls": 1, "target_seconds": .001, "schedule_seed": 1,
               "methods": ["bad", "good"], "case": {"id": "tiny", "kind": "single", "objective": "required",
                                                      "rows": 2, "cols": 3, "limit": None, "gate": None,
                                                      "definition_sha256": "case"}}
    return SimpleNamespace(request=request, calls=calls, states=states, logs=logs, fault=fault, current=current)


@pytest.mark.parametrize("phase,raise_after", [("screen", True), ("screen", False),
                                              ("warmup", False), ("primary", True)])
def test_mutation_invalidates_entire_attempt_even_when_solver_then_fails(harness, phase, raise_after):
    def mutate(cost, current):
        if current["phase"] == phase:
            cost.flags.writeable = True
            cost[0, 0] = 99
            if raise_after:
                raise ValueError("simulated solver failure after mutation")
    harness.fault["function"] = mutate
    with pytest.raises(worker.CorpusChanged, match="entire paired attempt"):
        worker.run(harness.request)
    assert not any(path.endswith("result.json") for path in harness.states)
    if phase in ("screen", "warmup"):
        assert not any(method == "good" for method, _, _ in harness.calls)


@pytest.mark.parametrize("failure_phase", ["screen", "primary"])
def test_worker_io_failure_propagates_instead_of_becoming_solver_correctness(harness, monkeypatch, failure_phase):
    def append(path, value):
        if value.get("method") == "bad" and value.get("phase") == failure_phase:
            raise OSError("simulated measurement-log storage failure")
        harness.logs.append(deepcopy(value))
    monkeypatch.setattr(worker, "append_jsonl", append)
    with pytest.raises(OSError, match="storage failure"):
        worker.run(harness.request)
    assert not any(path.endswith("result.json") for path in harness.states)
    assert not any(value.get("statuses", {}).get("bad", {}).get("status") == "correctness_failed"
                   for value in harness.states.values())


@pytest.mark.parametrize("exception", [adapters.UnsupportedCostDomain, worker.MethodStopped])
def test_mutation_cannot_be_hidden_by_a_noncorrectness_status(harness, exception):
    def mutate_and_stop(cost, current):
        cost.flags.writeable = True
        cost[0, 0] = 99
        raise exception("simulated domain or resource exit after mutation")
    harness.fault["function"] = mutate_and_stop
    with pytest.raises(worker.CorpusChanged):
        worker.run(harness.request)
    assert not any(method == "good" for method, _, _ in harness.calls)


def test_restored_active_budget_stops_method_before_any_new_solver_call(harness):
    harness.request["active_seed"] = {"bad": R1["method_active_seconds"], "good": 2.}
    result = worker.run(harness.request)
    assert result["statuses"]["bad"]["status"] == "resource_censored"
    assert result["statuses"]["good"]["status"] == "complete"
    assert not any(method == "bad" for method, _, _ in harness.calls)
    assert result["active_seconds"]["good"] > 2
    assert result["campaign_id"] == "test-campaign"
    timing = [x for x in harness.logs if x.get("record_type") == "timing"]
    assert len(timing) == 4 and all(x["method"] == "good" for x in timing)
    assert all(x["campaign_id"] == "test-campaign" for x in timing)
