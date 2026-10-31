"""Coverage and provenance checks; these fixtures execute no tracking/timing."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from .realbench_campaign import SOLVERS, load_inputs
from .realbench_data import SEQUENCES, SOURCES, sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def cache(sequence, source):
    return {"metadata": {
        "sequence": sequence, "source": source, "frames": 2,
        "video": {"decoded_frames": 2, "fully_decoded": True},
        "frame_mapping": "frame k maps to k",
        "reference": {"policy": "supplied boxes"},
        "inputs": {name: {"sha256": name + sequence} for name in ("video", "gt", "gt_map")},
        "detector": {"settings": {"conf": .2, "resolution": 768}, "checkpoint": {"sha256": "weights"}},
    }, "frames": [{"frame": index, "detections": [{"box": [0, 0, 10, 10]}],
                     "gt": [{"id": 1, "box": [0, 0, 10, 10]}]} for index in (1, 2)]}


def make_inputs(path, sources=SOURCES):
    for sequence in SEQUENCES:
        for source in sources:
            write(path / f"{sequence}_{source}.json", cache(sequence, source))


def test_rf_only_selects_two_caches_and_four_cases_from_both_sources(tmp_path):
    make_inputs(tmp_path)
    paths, caches, cases = load_inputs(tmp_path, 2, ["rfdetr_nano"])
    assert len(paths) == len(caches) == 2
    assert len(cases) == 4
    assert {row["source"] for row in cases} == {"rfdetr_nano"}
    assert {row["sequence"] for row in cases} == set(SEQUENCES)


def test_default_both_sources_retains_eight_cases(tmp_path):
    make_inputs(tmp_path)
    assert len(load_inputs(tmp_path, 2)[2]) == 8


def test_missing_selected_sequence_and_duplicate_metadata_are_rejected(tmp_path):
    make_inputs(tmp_path, ["rfdetr_nano"])
    path = tmp_path / f"{SEQUENCES[0]}_rfdetr_nano.json"
    value = json.loads(path.read_text())
    write(tmp_path / "zzz_duplicate.json", value)
    with pytest.raises(ValueError, match="Duplicate sequence/source"):
        load_inputs(tmp_path, 2, ["rfdetr_nano"])
    (tmp_path / "zzz_duplicate.json").unlink()
    path.unlink()
    with pytest.raises(ValueError, match="missing="):
        load_inputs(tmp_path, 2, ["rfdetr_nano"])


@pytest.mark.parametrize("change,error", [
    ("frames", "Decoded frame metadata"),
    ("source_identity", "Sources disagree"),
    ("checkpoint", "settings and checkpoint"),
    ("identity_leak", "leaks reference identity"),
])
def test_invalid_metadata_or_identity_leak_is_rejected(tmp_path, change, error):
    make_inputs(tmp_path)
    path = tmp_path / f"{SEQUENCES[0]}_rfdetr_nano.json"
    value = json.loads(path.read_text())
    if change == "frames":
        value["metadata"]["video"]["decoded_frames"] = 1
    elif change == "source_identity":
        value["metadata"]["inputs"]["video"]["sha256"] = "different-video"
    elif change == "checkpoint":
        value["metadata"]["detector"]["checkpoint"]["sha256"] = "different-weights"
    else:
        value["frames"][0]["detections"][0]["id"] = 1
    write(path, value)
    with pytest.raises(ValueError, match=error):
        load_inputs(tmp_path, 2)


def test_unexpected_filename_and_duplicate_source_request_are_rejected(tmp_path):
    make_inputs(tmp_path, ["gt"])
    path = tmp_path / f"{SEQUENCES[0]}_gt.json"
    path.rename(tmp_path / "bad_name.json")
    with pytest.raises(ValueError, match="filename disagrees"):
        load_inputs(tmp_path, 2, ["gt"])
    with pytest.raises(ValueError, match="unique selections"):
        load_inputs(tmp_path, 2, ["gt", "gt"])


def test_report_redacts_paths_from_existing_summary(tmp_path):
    from .realbench_report import generate_report
    summary = {"environment": {"executable": "C:\\Users\\Example User\\python.exe"}}
    write(tmp_path / "summary.json", summary)
    generate_report(tmp_path)
    for name in ("report.html", "report.txt"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "Example User" not in text
        assert "<external>" in text or "&lt;external&gt;" in text
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_report_only_redacts_copied_records_without_changing_source(tmp_path, monkeypatch):
    from . import run as portable
    source, output = tmp_path / "source", tmp_path / "report"
    values = {
        "summary.json": {"environment": {"executable": "C:\\Users\\Example User\\python.exe"}},
        "protocol.json": {"inputs": "/home/example/private/input.json"},
        "environment.json": {"executable": "C:\\Users\\Example User\\python.exe"},
    }
    for name, value in values.items():
        write(source / name, value)
    monkeypatch.setattr("sys.argv", ["run.py", "--report-only", "--collection", str(source), "--output", str(output)])
    portable.main()
    for name, value in values.items():
        text = (output / name).read_text(encoding="utf-8")
        assert "Example User" not in text
        assert "/home/example" not in text
        assert json.loads((source / name).read_text()) == value
    assert (output / "report.html").is_file()
