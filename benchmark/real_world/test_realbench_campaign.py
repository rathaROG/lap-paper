"""Coverage and provenance checks; these fixtures execute no tracking/timing."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from realbench_campaign import SOLVERS, load_inputs
from realbench_combine import combine, select_summary, validate_compatibility
from realbench_data import SEQUENCES, SOURCES, sha256_file


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


def make_campaign(path, source, timing):
    make_inputs(path / "inputs", [source])
    protocol = {"solvers": list(SOLVERS), "rounds": 1, "frames_per_clip": 2, "cpu": 0,
                "library_threads": 1, "tracker_input_precision": "float32", "replay_precisions": ["float32", "float64"],
                "warmup_frames": 1, "minimum_replay_block_seconds": .03, "adapter_scope": "unchanged adapter",
                "stock_tracker_semantics": "unchanged full assignment", "evaluation": "supplied boxes"}
    environment = {"cpu": "same CPU", "platform": "same OS", "python": "same interpreter", "executable": "same.exe",
                   "packages": {"numpy": "same"}, "solver_binaries": {"lap": {"native_files": [{"path": "same.pyd", "sha256": "binary"}]}},
                   "measurement_sources": {name: {"sha256": "unchanged"} for name in ("realbench_tracking.py", "realbench_evaluate.py")},
                   "pyppbox_sources": {"modules/trackers/sort/origin/sort.py": {"sha256": "unchanged"}}}
    tracking, replay, validation, samples = [], [], {}, []
    for sequence in SEQUENCES:
        for tracker in ("centroid", "sort"):
            case = f"{sequence}_{source}_{tracker}"
            common = {"case": case, "sequence": sequence, "source": source, "tracker": tracker}
            validation[case] = {"float32": {}, "float64": {}}
            matrix_path = path / "matrices" / f"{case}.npz"
            matrix_path.parent.mkdir(parents=True, exist_ok=True)
            matrix_path.write_bytes(b"opaque fixture archive; never executed")
            for precision in ("float32", "float64"):
                write(path / "replay" / f"{case}_{precision}.json", {"fixture": "saved replay"})
            for solver in SOLVERS:
                tracking.append({**common, "solver": solver, "precision": "float32", "rounds": 1,
                                 "frames": 2, "median_tracker_ns_per_frame": timing, "prediction_sha256": solver})
                for precision in ("float32", "float64"):
                    replay.append({**common, "solver": solver, "precision": precision, "rounds": 1,
                                   "median_ns_per_assignment": timing})
                name = f"{case}__{solver}.json"
                for folder in ("predictions", "metrics", "profiles"):
                    write(path / folder / name, {"fixture": "exact original bytes", "timing": timing})
                if solver == "scipy":
                    write(path / "profiles" / f"{case}__scipy_without_capture.json", {"fixture": "profile"})
                write(path / "timing" / "round_00" / name, {"fixture": "raw timing", "timing": timing})
                samples.append({**common, "solver": solver, "round": 0, "nanoseconds": timing})
    summary = {"protocol": protocol, "environment": environment, "tracking": tracking, "replay": replay,
               "inputs": [cache(sequence, source)["metadata"] for sequence in SEQUENCES], "validation": validation,
               "started_utc": "2026-10-27T10:00:00+00:00", "completed_utc": "2026-10-27T10:01:00+00:00",
               "reproducibility": {"all_repeated_prediction_hashes_identical": True, "all_instrumented_predictions_match": True}}
    for name, value in (("summary.json", summary), ("protocol.json", protocol), ("environment.json", environment),
                        ("file_manifest.json", {}), ("tracking_samples.json", samples)):
        write(path / name, value)
    write(path / "provenance" / "source_identity.json", {"fixture": "original provenance"})


def test_composition_keeps_raw_bytes_and_separate_sessions_without_pooling(tmp_path, monkeypatch):
    gt, rf, output = tmp_path / "gt", tmp_path / "rf", tmp_path / "combined"
    make_campaign(gt, "gt", 100)
    make_campaign(rf, "rfdetr_nano", 200)
    import realbench_report

    def report_stub(directory):
        (directory / "report.html").write_text("<div class='scope'>report</div>")
        (directory / "report.txt").write_text("report")

    monkeypatch.setattr(realbench_report, "generate_report", report_stub)
    hashes = {str(path): sha256_file(path) for root in (gt, rf) for path in root.rglob("*") if path.is_file()}
    combine(gt, rf, output)
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["tracking"]) == 72
    assert len(summary["replay"]) == 144
    assert {row["median_tracker_ns_per_frame"] for row in summary["tracking"] if row["source"] == "gt"} == {100}
    assert {row["median_tracker_ns_per_frame"] for row in summary["tracking"] if row["source"] == "rfdetr_nano"} == {200}
    assert summary["protocol"]["collection_mode"] == "derived_composition_no_measurements"
    assert set(summary["source_environment_by_source"]) == set(SOURCES)
    assert "GT was not rerun" in (output / "report.html").read_text()
    assert "GT was not rerun" in (output / "report.txt").read_text()
    composition = json.loads((output / "composition.json").read_text())
    campaigns = {"gt_v1": gt, "rf_v2": rf}
    for destination, record in composition["copied_files"].items():
        original = campaigns[record["origin_campaign"]] / record["original_path"]
        assert sha256_file(output / destination) == record["sha256"] == sha256_file(original)
    assert all(sha256_file(path) == digest for path, digest in hashes.items())
    manifest = json.loads((output / "file_manifest.json").read_text())
    assert str(Path("provenance/campaigns/gt_v1/file_manifest.json")) in manifest
    with pytest.raises(FileExistsError, match="fresh directory"):
        combine(gt, rf, output)


def test_composition_rejects_missing_case_and_incompatible_binary(tmp_path):
    gt, rf = tmp_path / "gt", tmp_path / "rf"
    make_campaign(gt, "gt", 100)
    make_campaign(rf, "rfdetr_nano", 200)
    selected_gt, selected_rf = select_summary(gt, "gt"), select_summary(rf, "rfdetr_nano")
    changed = deepcopy(selected_rf)
    changed["summary"]["environment"]["solver_binaries"]["lap"]["native_files"][0]["sha256"] = "different"
    with pytest.raises(ValueError, match="native solver binaries differ"):
        validate_compatibility(selected_gt, changed)
    summary = json.loads((rf / "summary.json").read_text())
    summary["tracking"].pop()
    write(rf / "summary.json", summary)
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        select_summary(rf, "rfdetr_nano")


def test_composition_rejects_unredacted_saved_evidence(tmp_path):
    gt, rf, output = tmp_path / "gt", tmp_path / "rf", tmp_path / "combined"
    make_campaign(gt, "gt", 100)
    make_campaign(rf, "rfdetr_nano", 200)
    write(gt / "provenance" / "source_identity.json", {"path": "C:\\Users\\person\\private.txt"})
    with pytest.raises(ValueError, match="Saved evidence contains local paths"):
        combine(gt, rf, output)
    assert not output.exists()


def test_report_redacts_paths_from_existing_summary(tmp_path):
    from realbench_report import generate_report
    summary = {"environment": {"executable": "C:\\Users\\Example User\\python.exe"}}
    write(tmp_path / "summary.json", summary)
    generate_report(tmp_path)
    for name in ("report.html", "report.txt"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert "Example User" not in text
        assert "<external>" in text or "&lt;external&gt;" in text
    assert json.loads((tmp_path / "summary.json").read_text()) == summary


def test_report_only_redacts_copied_records_without_changing_source(tmp_path, monkeypatch):
    import run as portable
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
