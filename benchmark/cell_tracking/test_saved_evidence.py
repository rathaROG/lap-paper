"""Saved-record regression checks only; never import the measurement driver."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from . import recalculate as study
from ..paper_artifacts_microscopy import generate, load_saved

REPO = Path(__file__).resolve().parents[2]


def test_saved_session_identity_and_recorded_agreement():
    observations, rows = load_saved(REPO)
    assert observations["versions"]["lapx"] == "0.13.1"
    assert observations["python"] == "3.14.10"
    assert sum(row["frames"] for row in rows) == 816
    assert len(rows) == 9
    assert all(check == (True, True, 0) for row in rows for check in row["comparisons"].values())
    assert all(sum(d["profiles"][arm]["nonempty"] for d in observations["datasets"].values()) == 807
               for arm in study.ARMS)


def test_aggregate_uses_unrounded_profile_totals_not_primary_rounds():
    observations, rows = load_saved(REPO)
    expected = {arm: sum(d["frames"] * d["profiles"][arm]["association_ns_per_frame"] / 1e9
                         for d in observations["datasets"].values()) for arm in study.ARMS}
    totals = study.association_totals(rows)
    assert totals == pytest.approx(expected)
    assert totals[study.CAPPED] == pytest.approx(0.639493)
    assert totals[study.RUST] == pytest.approx(0.3160676)
    changed = deepcopy(observations)
    for dataset in changed["datasets"].values():
        for timing in dataset["rounds"]:
            for record in timing.values():
                record["ns_per_frame"] *= 100
    assert study.association_totals(study.derive(changed)) == totals


@pytest.mark.parametrize("fault, message", [("order", "six balanced rounds"),
                                           ("frames", "frame bounds"),
                                           ("tracks", "track hashes")])
def test_inconsistent_saved_records_are_rejected(fault, message):
    observations, _ = load_saved(REPO)
    data = next(iter(observations["datasets"].values()))
    if fault == "order":
        data["rounds"][0] = dict(reversed(list(data["rounds"][0].items())))
    elif fault == "frames":
        data["frames"] -= 1
    else:
        data["rounds"][0][study.RUST]["tracks"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        study.derive(observations)


def test_changed_observation_bytes_fail_before_projection(tmp_path):
    folder = tmp_path / "benchmark/cell_tracking"
    folder.mkdir(parents=True)
    for name in (*study.BOUND_FILES, "manifest.json"):
        shutil.copyfile(REPO / "benchmark/cell_tracking" / name, folder / name)
    with (folder / "observations.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_saved(tmp_path)


def test_projection_reproduces_included_tables_and_dispersion(tmp_path):
    folder = tmp_path / "benchmark/cell_tracking"
    folder.mkdir(parents=True)
    for name in (*study.BOUND_FILES, "manifest.json"):
        shutil.copyfile(REPO / "benchmark/cell_tracking" / name, folder / name)
    summary = generate(tmp_path)
    assert summary["capped_over_rust_round_ratio_range"] == pytest.approx([.935, 1.165], abs=.0005)
    assert summary["capped_median_ratios_inside_3_percent_band"] == 6
    assert summary["excerpts_with_round_favoring_capped"] == 3
    for name in ("T_microscopy.tex", "A_microscopy_profiles.tex"):
        assert (tmp_path / "latex/generated" / name).read_bytes() == (REPO / "latex/generated" / name).read_bytes()


def test_audit_runs_with_solver_and_tracker_imports_blocked():
    code = """
import importlib.abc, runpy, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'byotrack','pylapy','torch','numpy','scipy','lap','laprust','lapcpp'}:
            raise AssertionError('Unexpected native dependency: ' + fullname)
sys.meta_path.insert(0, BlockNative())
sys.argv = ['recalculate.py']
runpy.run_module('benchmark.cell_tracking.recalculate', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "9 excerpts, 816 frames" in result.stdout


def test_run_needs_opt_in_before_loading_tracker_or_creating_output(tmp_path):
    output = tmp_path / "not-created/observations.json"
    result = subprocess.run([sys.executable, "-B", "-m", "benchmark.cell_tracking", "run",
                             "--ctc", str(tmp_path), "--hydra", str(tmp_path / "video.avi"),
                             "--output", str(output)], cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert "--allow-full-campaign" in result.stderr
    assert not output.parent.exists()


def test_run_refuses_frozen_evidence_output(tmp_path):
    result = subprocess.run([sys.executable, "-B", "-m", "benchmark.cell_tracking", "run",
                             "--allow-full-campaign", "--ctc", str(tmp_path),
                             "--hydra", str(tmp_path / "video.avi"),
                             "--output", str(REPO / "benchmark/cell_tracking/new.json")],
                            cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert "frozen evidence cannot be replaced" in result.stderr


@pytest.fixture
def audit_copy(tmp_path):
    """Exercise both command entry points without risking the real frozen files."""
    folder = tmp_path / "benchmark/cell_tracking"
    folder.mkdir(parents=True)
    (folder.parent / "__init__.py").write_text("", encoding="utf-8")
    for name in (*study.BOUND_FILES, "manifest.json", "recalculate.py", "__main__.py", "__init__.py"):
        shutil.copyfile(REPO / "benchmark/cell_tracking" / name, folder / name)
    return folder


def run_audit_copy(folder, entry_point, *arguments):
    command = ([str(folder / "recalculate.py")] if entry_point == "direct"
               else ["-m", "benchmark.cell_tracking", "audit"])
    return subprocess.run([sys.executable, "-B", *command, *map(str, arguments)],
                          cwd=folder.parents[1], capture_output=True, text=True)


@pytest.mark.parametrize("entry_point", ["direct", "wrapper"])
@pytest.mark.parametrize("option", ["--table", "--compact-table", "--profile-table"])
@pytest.mark.parametrize("name", ["observations.json", "benchmark.py", "manifest.json", "new.tex"])
def test_table_exports_refuse_frozen_destinations(audit_copy, entry_point, option, name):
    before = {p.name: p.read_bytes() for p in audit_copy.iterdir()}
    # Include a parent-directory alias to exercise resolved containment.
    target = audit_copy / ".." / "cell_tracking" / name
    result = run_audit_copy(audit_copy, entry_point, option, target)
    assert result.returncode != 0
    assert "frozen evidence cannot be replaced" in result.stderr
    assert {p.name: p.read_bytes() for p in audit_copy.iterdir()} == before


@pytest.mark.parametrize("entry_point", ["direct", "wrapper"])
def test_all_exports_are_validated_before_any_write(audit_copy, entry_point):
    output = audit_copy.parents[1] / "safe.tex"
    manifest = audit_copy / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    before = manifest.read_bytes()
    arguments = ["--table", output, "--profile-table", audit_copy / "observations.json"]
    if entry_point == "direct":
        arguments.append("--write-manifest")
    result = run_audit_copy(audit_copy, entry_point, *arguments)
    assert result.returncode != 0
    assert "frozen evidence cannot be replaced" in result.stderr
    assert not output.exists()
    assert manifest.read_bytes() == before


def test_table_export_refuses_symlink_to_frozen_file(audit_copy):
    alias = audit_copy.parents[1] / "alias.tex"
    target = audit_copy / "observations.json"
    before = target.read_bytes()
    try:
        alias.symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable: {error}")
    result = run_audit_copy(audit_copy, "direct", "--table", alias)
    assert result.returncode != 0
    assert "frozen evidence cannot be replaced" in result.stderr
    assert target.read_bytes() == before


@pytest.mark.parametrize("entry_point", ["direct", "wrapper"])
def test_external_table_exports_still_reproduce_saved_results(audit_copy, entry_point):
    _, rows = load_saved(REPO)
    renderers = {"table": study.table, "compact-table": study.compact_table,
                 "profile-table": study.profile_table}
    arguments = []
    for option in renderers:
        arguments.extend(["--" + option, audit_copy.parents[1] / (option + ".tex")])
    result = run_audit_copy(audit_copy, entry_point, *arguments)
    assert result.returncode == 0, result.stderr
    for option, render in renderers.items():
        assert (audit_copy.parents[1] / (option + ".tex")).read_bytes() == render(rows).encode("utf-8")
