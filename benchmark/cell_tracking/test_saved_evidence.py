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
