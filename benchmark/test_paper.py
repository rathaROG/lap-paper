"""Exercise the paper entry point without builds, network access or measurements."""
from pathlib import Path
from contextlib import nullcontext
import json
from io import BytesIO, StringIO, TextIOWrapper
import subprocess
import sys

import pytest

from . import paper


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    commands, prepared = [], []
    topology = {
        "allowed_cpus": list(range(10)),
        "cores": [{"core_id": i, "logical_cpus": [2*i, 2*i+1],
                   "efficiency_class": 1, "smt": True} for i in range(5)],
        "recommended_batch_cpus": [0, 2, 4, 6], "recommended_single_cpu": 2,
    }
    monkeypatch.setattr(paper, "check_dependencies", lambda: None)
    monkeypatch.setattr(paper, "discover", lambda: topology)
    monkeypatch.setattr(paper, "HostLock", nullcontext)

    def campaign(cache, *, diagnostic=False):
        prepared.append(("diagnostic" if diagnostic else "primary", cache))
        return cache / ("diagnostic-build-index.json" if diagnostic else "build-index.json")

    def lsape(cache):
        prepared.append(("lsape", cache))
        return cache / "lsape-build"

    def execute(command, *, log):
        commands.append(list(command))
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("mock stage\n", encoding="utf-8")
        if module(command) == "benchmark.campaign" and command[4] == "diagnose":
            destination = Path(option(command, "--output"))
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "summary.json").write_text(
                json.dumps({"status": "finished", "jobs": [{"status": "complete"}]}), encoding="utf-8")

    monkeypatch.setattr(paper, "prepare_campaign", campaign)
    monkeypatch.setattr(paper, "prepare_lsape", lsape)
    monkeypatch.setattr(paper, "execute", execute)
    monkeypatch.setattr(paper.platform, "system", lambda: "Linux")
    return commands, prepared, tmp_path / "output", tmp_path / "cache"


def module(command):
    return command[command.index("-m") + 1]


def option(command, flag):
    return command[command.index(flag) + 1]


def test_linux_smoke_uses_selected_cpus_and_same_interpreter(workflow):
    commands, prepared, output, cache = workflow
    report = paper.run_workflow("smoke", output, cache)
    assert report["status"] == "complete"
    assert report["cpus"] == [0, 2, 4, 6]
    assert report["single_cpu"] == 2
    assert commands
    assert all(command[:3] == [sys.executable, "-B", "-m"] for command in commands)
    for command in commands:
        if module(command) == "benchmark.campaign" and command[4] in ("smoke", "diagnose"):
            offset = command.index("--cpus")
            assert command[offset+1:offset+5] == ["0", "2", "4", "6"]
            assert option(command, "--single-cpu") == "2"
        if module(command) == "benchmark.application" and "--smoke" in command:
            assert option(command, "--cpu") == "2"
    assert not any(kind == "lsape" for kind, _ in prepared)
    assert all(module(command) != "benchmark.lsape" for command in commands)


@pytest.mark.parametrize("single_cpu, expected_single", [(None, 8), (4, 4)])
def test_explicit_cpus_reach_every_measurement_stage(workflow, single_cpu, expected_single):
    commands, _, output, cache = workflow
    cpus = [8, 6, 4, 2]
    report = paper.run_workflow("run", output, cache, cpus=cpus, single_cpu=single_cpu)
    assert report["cpus"] == cpus and report["single_cpu"] == expected_single
    for command in commands:
        if module(command) == "benchmark.campaign" and command[4] in ("smoke", "run", "diagnose"):
            offset = command.index("--cpus")
            assert command[offset+1:offset+5] == list(map(str, cpus))
            assert option(command, "--single-cpu") == str(expected_single)
        elif module(command) == "benchmark.application" and any(flag in command for flag in ("--smoke", "--campaign")):
            assert option(command, "--cpu") == str(expected_single)


def test_single_cpu_override_uses_the_automatic_batch_selection(workflow):
    commands, _, output, cache = workflow
    report = paper.run_workflow("smoke", output, cache, single_cpu=4)
    assert report["cpus"] == [0, 2, 4, 6] and report["single_cpu"] == 4
    application = next(command for command in commands if module(command) == "benchmark.application" and "--smoke" in command)
    assert option(application, "--cpu") == "4"


@pytest.mark.parametrize("cpus, single_cpu, unknown_classes, message", [
    ([10, 11, 12, 13], None, False, "highest OS-reported performance class"),
    ([0, 2, 4, 10], None, False, "mixes OS-reported core classes"),
    ([0, 1, 4, 6], None, False, "SMT siblings"),
    ([0, 2, 4, 99], None, False, "outside the allowed mask"),
    ([0, 0, 4, 6], None, False, "distinct hardware threads"),
    ([0, 2, 4, 6], None, True, "lacks verified performance-core class"),
    (None, 8, False, "serial core must belong to the batch selection"),
    ([0], None, False, "Select four physical cores"),
], ids=["e-cores", "mixed-classes", "siblings", "outside", "duplicate", "unknown-hybrid", "serial-outside", "too-few"])
def test_cpu_overrides_reject_invalid_placement_before_preparation(
        workflow, monkeypatch, cpus, single_cpu, unknown_classes, message):
    commands, prepared, output, cache = workflow
    topology = paper.discover()
    topology["cores"] += [dict(core_id=5+i, logical_cpus=[10+i], efficiency_class=0, smt=False)
                          for i in range(4)]
    topology["allowed_cpus"] = list(range(14))
    if unknown_classes:
        for core in topology["cores"]:
            core["efficiency_class"] = None
    with pytest.raises(paper.WorkflowError, match=message):
        paper.run_workflow("smoke", output, cache, cpus=cpus, single_cpu=single_cpu)
    assert prepared == [] and commands == []
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_manual_cpus_still_require_a_supervisor_outside_measured_cores(workflow):
    commands, prepared, output, cache = workflow
    topology = paper.discover()
    topology["allowed_cpus"] = list(range(8))
    with pytest.raises(paper.WorkflowError, match="supervisor CPU"):
        paper.run_workflow("smoke", output, cache, cpus=[0, 2, 4, 6])
    assert prepared == [] and commands == []


@pytest.mark.parametrize("cpu_zero", ["e-core", "forbidden"])
def test_windows_rejects_unusable_lsape_cpu_zero_before_preparation(workflow, monkeypatch, cpu_zero):
    commands, prepared, output, cache = workflow
    monkeypatch.setattr(paper.platform, "system", lambda: "Windows")
    topology = paper.discover()
    if cpu_zero == "e-core":
        topology["cores"][0].update(logical_cpus=[0], efficiency_class=0, smt=False)
        topology["allowed_cpus"].remove(1)
    else:
        topology["allowed_cpus"].remove(0)
    cpus = [2, 4, 6, 8]
    assert len(paper.validate_selection(topology, cpus, cpus[0])) == 4
    with pytest.raises(paper.WorkflowError, match="LSAPE protocol requires.*CPU 0"):
        paper.run_workflow("smoke", output, cache, cpus=cpus)
    assert prepared == [] and commands == []
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_windows_allows_verified_lsape_cpu_zero_outside_manual_batch(workflow, monkeypatch):
    commands, prepared, output, cache = workflow
    monkeypatch.setattr(paper.platform, "system", lambda: "Windows")
    report = paper.run_workflow("smoke", output, cache, cpus=[2, 4, 6, 8])
    assert report["status"] == "complete" and 0 not in report["cpus"]
    assert any(kind == "lsape" for kind, _ in prepared)
    assert any(module(command) == "benchmark.lsape" and command[4] == "smoke" for command in commands)


def test_smoke_remains_bounded_and_keeps_each_stage_output_separate(workflow):
    commands, _, output, cache = workflow
    paper.run_workflow("smoke", output, cache)
    assert not any("--allow-full-campaign" in command or "--full" in command for command in commands)
    assert not any(module(command) == "benchmark.campaign" and command[4] == "run" for command in commands)
    assert not any("--campaign" in command for command in commands)
    outputs = [Path(option(command, "--output")) for command in commands if "--output" in command]
    assert len(outputs) == len(set(outputs))
    assert all(path.is_relative_to(output) for path in outputs)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["stages"][0]["name"] == "preparation"
    assert len(status["stages"]) == len(commands) + 1
    assert all(stage["status"] == "complete" for stage in status["stages"])


def test_windows_full_workflow_runs_three_lsape_sessions_after_readiness(workflow, monkeypatch):
    commands, prepared, output, cache = workflow
    monkeypatch.setattr(paper.platform, "system", lambda: "Windows")
    paper.run_workflow("run", output, cache)
    lsape = [command for command in commands if module(command) == "benchmark.lsape"]
    assert any(kind == "lsape" for kind, _ in prepared)
    assert [command[4] for command in lsape] == ["verify", "smoke", "run", "run", "run", "summarize"]
    sessions = [command for command in lsape if command[4] == "run"]
    assert [option(command, "--session-index") for command in sessions] == ["1", "2", "3"]
    assert all("--allow-full-campaign" in command for command in sessions)
    assignment = [command for command in commands if module(command) == "benchmark.campaign"]
    assert any(command[4] == "run" and "--allow-full-campaign" in command for command in assignment)
    assert any(command[4] == "summarize" for command in assignment)
    assert any(command[4] == "diagnose" and "--full" in command and "--allow-full-campaign" in command
               for command in assignment)
    assert any(module(command) == "benchmark.application" and "--campaign" in command
               and "--allow-full-campaign" in command for command in commands)
    assert commands.index(next(command for command in assignment if command[4] == "run")) > commands.index(
        next(command for command in commands if module(command) == "benchmark.application" and "--smoke" in command))


def test_stage_failure_stops_workflow_and_retains_failure_record(workflow, monkeypatch):
    commands, _, output, cache = workflow

    def fail_second(command, *, log):
        commands.append(list(command))
        if len(commands) == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(paper, "execute", fail_second)
    with pytest.raises(paper.WorkflowError):
        paper.run_workflow("smoke", output, cache)
    assert len(commands) == 2
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert [stage["status"] for stage in status["stages"]] == ["complete", "complete", "failed"]


@pytest.mark.parametrize("summary", [
    {"status": "postponed", "jobs": [{"status": "postponed"}]},
    {"status": "postponed", "jobs": [{"status": "complete"}]},
    {"status": "finished", "jobs": [{"status": "postponed"}]},
    None,
], ids=["postponed", "unfinished-session", "unfinished-job", "missing-summary"])
def test_successful_diagnostic_process_requires_completed_evidence_before_full_run(
        workflow, monkeypatch, summary):
    commands, _, output, cache = workflow
    successful_process = paper.execute

    def incomplete_diagnostics(command, *, log):
        successful_process(command, log=log)
        if module(command) == "benchmark.campaign" and command[4] == "diagnose":
            path = Path(option(command, "--output")) / "summary.json"
            if summary is None:
                path.unlink()
            else:
                path.write_text(json.dumps(summary), encoding="utf-8")

    monkeypatch.setattr(paper, "execute", incomplete_diagnostics)
    with pytest.raises(paper.WorkflowError):
        paper.run_workflow("run", output, cache)
    assert commands[-1][4] == "diagnose"
    assert not any("--allow-full-campaign" in command or "--full" in command for command in commands)
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["stages"][-1]["name"] == "diagnostic-smoke"
    assert status["stages"][-1]["status"] == "failed"


def test_unusable_topology_stops_before_artifact_preparation(workflow, monkeypatch):
    commands, prepared, output, cache = workflow
    monkeypatch.setattr(paper, "discover", lambda: {
        "recommended_batch_cpus": [], "recommended_single_cpu": None,
        "allowed_cpus": [], "cores": [],
    })
    with pytest.raises(paper.WorkflowError) as error:
        paper.run_workflow("smoke", output, cache)
    message = str(error.value)
    assert "benchmark.campaign topology" in message
    assert "--cpus" in message and "--single-cpu" in message
    assert prepared == [] and commands == []
    assert json.loads((output / "status.json").read_text())["status"] == "failed"


def test_default_paths_avoid_existing_output_and_partition_cache(tmp_path, monkeypatch):
    existing = tmp_path / "paper-smoke"
    existing.mkdir()
    called = []
    monkeypatch.setattr(paper, "default_output", lambda name: existing)
    monkeypatch.setattr(paper, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(paper.platform, "python_version", lambda: "3.14.10")
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: called.append((args, kwargs)))
    assert paper.main(["smoke"]) == 0
    assert called == [(("smoke", tmp_path / "paper-smoke-2",
                       tmp_path / "runs/.cache/linux-py3.14.10"),
                      {"cpus": None, "single_cpu": None, "profile": "full"})]


def test_main_forwards_optional_cpu_flags(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(paper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(paper, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))
    output = tmp_path / "output"
    assert paper.main(["smoke", "--output", str(output), "--cpus", "8", "6", "4", "2", "--single-cpu", "4"]) == 0
    assert calls[0][0][:2] == ("smoke", output)
    assert calls[0][1] == {"cpus": [8, 6, 4, 2], "single_cpu": 4, "profile": "full"}


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_light_workflow_selects_light_campaign_and_keeps_readiness_and_lsape(workflow, monkeypatch, system):
    commands, _, output, cache = workflow
    monkeypatch.setattr(paper.platform, "system", lambda: system)
    report = paper.run_workflow("run", output, cache, profile="light")
    assert report["profile"] == "light" and report["status"] == "complete"
    assignment = [command for command in commands if module(command) == "benchmark.campaign"]
    assert {command[4] for command in assignment} == {"validate", "smoke", "estimate", "run", "summarize", "diagnose"}
    assert all(option(command, "--profile") == "light" for command in assignment)
    application = [command for command in commands if module(command) == "benchmark.application"]
    assert any("--validate" in command for command in application)
    assert any("--smoke" in command for command in application)
    session = next(command for command in application if "--campaign" in command)
    assert option(session, "--profile") == "light" and "--allow-full-campaign" in session
    assert any(command[4] == "diagnose" and "--full" in command for command in assignment)
    lsape = [command for command in commands if module(command) == "benchmark.lsape"]
    assert len([command for command in lsape if command[4] == "run"]) == (3 if system == "Windows" else 0)
    assert all("--profile" not in command for command in lsape)
    assert json.loads((output / "status.json").read_text())["profile"] == "light"


@pytest.mark.parametrize("flag,profile,destination", [
    ("--allow-full-campaign", "full", "paper-run"),
    ("--allow-full-campaign-light", "light", "paper-run-light"),
])
def test_main_selects_protocol_and_separate_default_output(tmp_path, monkeypatch, flag, profile, destination):
    calls = []
    monkeypatch.setattr(paper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(paper.platform, "python_version", lambda: "3.11.18")
    monkeypatch.setattr(paper, "default_output", lambda name: tmp_path / name)
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert paper.main(["run", flag]) == 0
    assert calls[0][0][1] == tmp_path / destination
    assert calls[0][1]["profile"] == profile


@pytest.mark.parametrize("arguments", [
    ["run"], ["run", "--allow-full-campaign", "--allow-full-campaign-light"],
    ["smoke", "--allow-full-campaign-light"],
])
def test_invalid_protocol_opt_ins_never_start_workflow(tmp_path, monkeypatch, arguments):
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: pytest.fail("Unexpected measurements"))
    output = tmp_path / "output"
    with pytest.raises(SystemExit):
        paper.main([*arguments, "--output", str(output)])
    assert not output.exists()


def test_main_rejects_partial_batch_cpu_override_before_preparation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(SystemExit):
        paper.main(["smoke", "--cpus", "0", "2", "4"])
    assert not calls


@pytest.mark.parametrize("encoding", ["ascii", "cp1252"])
def test_main_replaces_unencodable_console_progress_but_keeps_utf8_log(tmp_path, monkeypatch, encoding):
    progress = "stage progress: 1/2 δ\nstage progress: 2/2 ✓\n"
    log = tmp_path / "logs/progress.log"
    command = [sys.executable, "-B", "-m", "benchmark.campaign", "validate"]
    terminated = []

    class Process:
        pid = 12345
        stdout = StringIO(progress)

        def wait(self):
            return 0

    buffer = BytesIO()
    console = TextIOWrapper(buffer, encoding=encoding, errors="strict")
    with monkeypatch.context() as changes:
        changes.setattr(paper.sys, "stdout", console)
        changes.setattr(paper.platform, "system", lambda: "Linux")
        changes.setattr(paper.subprocess, "Popen", lambda *args, **kwargs: Process())
        changes.setattr(paper, "terminate_tree", lambda pid: terminated.append(pid))
        changes.setattr(paper, "run_workflow", lambda *args, **kwargs: paper.execute(command, log=log))
        assert paper.main(["smoke", "--output", str(tmp_path / "output")]) == 0
        assert console.errors == "replace"
        console.flush()
        displayed = buffer.getvalue().decode(encoding)
    console.close()
    assert progress.encode(encoding, errors="replace").decode(encoding) in displayed.replace("\r\n", "\n")
    assert log.read_text(encoding="utf-8") == progress
    assert not terminated


@pytest.mark.parametrize("custom", [False, True], ids=["stringio", "custom-stream"])
def test_main_supports_stdout_without_callable_reconfigure(tmp_path, monkeypatch, custom):
    class Console:
        reconfigure = None

        def __init__(self):
            self.text = ""

        def write(self, text):
            self.text += text
            return len(text)

        def flush(self):
            pass

        def getvalue(self):
            return self.text

    console = Console() if custom else StringIO()
    calls = []
    monkeypatch.setattr(paper.sys, "stdout", console)
    monkeypatch.setattr(paper.platform, "system", lambda: "Linux")
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))
    assert paper.main(["smoke", "--output", str(tmp_path / "output")]) == 0
    assert len(calls) == 1
    assert "smoke complete:" in console.getvalue()


def test_help_names_the_public_module_command_instead_of_main_file(monkeypatch, capsys):
    monkeypatch.setattr(paper.sys, "argv", ["C:/project/benchmark/__main__.py"])
    with pytest.raises(SystemExit) as error:
        paper.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage: python -m benchmark " in help_text
    assert "__main__.py" not in help_text


def test_failed_command_retains_its_console_log(tmp_path, monkeypatch):
    class FailedProcess:
        stdout = StringIO("stage failure details\n")
        returncode = 3

        def wait(self):
            return self.returncode

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stdout.close()

    def fail(command, **options):
        assert options["stdout"] == subprocess.PIPE
        assert options["stderr"] == subprocess.STDOUT
        return FailedProcess()

    monkeypatch.setattr(paper.subprocess, "Popen", fail)
    monkeypatch.setattr(paper.subprocess, "run", lambda *args, **kwargs: pytest.fail("Expected live Popen output"))
    log = tmp_path / "logs/failure.log"
    with pytest.raises(paper.WorkflowError, match="exited 3"):
        paper.execute([sys.executable, "-B", "-m", "benchmark.campaign", "validate"], log=log)
    assert log.read_text(encoding="utf-8") == "stage failure details\n"


def test_command_progress_is_flushed_to_console_and_log_before_wait_finishes(tmp_path, monkeypatch):
    text = "stage progress: 1/2 δ\nstage progress: 2/2 ✓\n"
    log = tmp_path / "logs/progress.log"
    waited = []

    class Console(StringIO):
        flush_count = 0

        def flush(self):
            self.flush_count += 1
            super().flush()

    console = Console()

    class ProgressProcess:
        stdout = StringIO(text)
        returncode = 0

        def wait(self):
            assert console.getvalue() == text
            assert console.flush_count > 0
            assert log.read_text(encoding="utf-8") == text
            waited.append(True)
            return self.returncode

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.stdout.close()

    command = [sys.executable, "-B", "-m", "benchmark.campaign", "validate"]

    def progress(actual, **options):
        assert actual == command
        assert options["stdout"] == subprocess.PIPE
        assert options["stderr"] == subprocess.STDOUT
        assert options["text"] is True
        assert options["encoding"] == "utf-8"
        assert options["env"]["PYTHONUNBUFFERED"] == "1"
        assert options["env"]["PYTHONIOENCODING"].startswith("utf-8")
        return ProgressProcess()

    monkeypatch.setattr(paper.subprocess, "Popen", progress)
    monkeypatch.setattr(paper.subprocess, "run", lambda *args, **kwargs: pytest.fail("Expected live Popen output"))
    monkeypatch.setattr(paper.sys, "stdout", console)
    paper.execute(command, log=log)
    assert waited
    assert log.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("error", [OSError("console unavailable"), KeyboardInterrupt()],
                         ids=["console-error", "interrupted"])
def test_progress_failure_stops_child_and_keeps_written_log(tmp_path, monkeypatch, error):
    events = []
    log = tmp_path / "logs/interrupted.log"

    class Process:
        pid = 12345
        stdout = StringIO("progress before interruption\n")

        def wait(self):
            assert events == [("terminate", self.pid)]
            events.append(("wait", self.pid))
            return 0

    class BrokenConsole(StringIO):
        def write(self, text):
            raise error

    process = Process()
    monkeypatch.setattr(paper.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(paper.subprocess, "run", lambda *args, **kwargs: pytest.fail("Expected live Popen output"))
    monkeypatch.setattr(paper, "terminate_tree", lambda pid: events.append(("terminate", pid)))
    monkeypatch.setattr(paper.sys, "stdout", BrokenConsole())
    with pytest.raises(type(error)):
        paper.execute([sys.executable, "-B", "-m", "benchmark.campaign", "validate"], log=log)
    assert events == [("terminate", process.pid), ("wait", process.pid)]
    assert process.stdout.closed
    assert log.read_text(encoding="utf-8") == "progress before interruption\n"


def test_full_run_requires_opt_in_before_preparation(tmp_path, monkeypatch):
    output = tmp_path / "never-created"
    calls = []
    monkeypatch.setattr(paper, "run_workflow", lambda *args, **kwargs: calls.append((args, kwargs)))
    with pytest.raises(SystemExit):
        paper.main(["run", "--output", str(output)])
    assert calls == []
    assert not output.exists()
