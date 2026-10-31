"""Saved-data-only protocol audit for supplement S1.

This module never imports benchmark generators, native solvers, or timing code.
The FULL session is used only for descriptive round truncation and coverage
checks. Its timing estimates must not enter the four-environment main results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import median


FULL = "linux-py3.11.18-20261030-1434/paper-run"
LIGHT = {
    "L311": "linux-py3.11.18-20261031-0455/paper-run-light",
    "L314": "linux-py3.14.10-20261031-0630/paper-run-light",
    "W311": "windows-py3.11.18-20261030-2322/paper-run-light",
    "W314": "windows-py3.14.10-20261031-0322/paper-run-light",
}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _classify(ratio):
    return "loss" if ratio < 0.97 else "win" if ratio > 1.03 else "tie"


def _csv(path, rows):
    assert rows
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _application_changes(samples, boundary, source):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[(sample["case"], sample.get("precision", "float32"), sample["solver"])].append(sample)
    records = []
    for (case, precision, method), values in sorted(grouped.items()):
        if method == "scipy":
            continue
        reference = grouped[case, precision, "scipy"]
        assert sorted(v["round"] for v in values) == list(range(9))
        assert sorted(v["round"] for v in reference) == list(range(9))
        full = median(v["nanoseconds"] for v in reference) / median(v["nanoseconds"] for v in values)
        short = median(v["nanoseconds"] for v in reference if v["round"] < 3) / median(v["nanoseconds"] for v in values if v["round"] < 3)
        records.append({
            "boundary": boundary, "case_id": case, "precision": precision,
            "method": method, "reference": "scipy", "round_indices": "0,1,2",
            "full_nine_round_ratio": full, "first_three_round_ratio": short,
            "relative_change_percent": 100 * (short / full - 1),
            "full_classification": _classify(full), "first_three_classification": _classify(short),
            "source": source,
            "source_fields": "samples[].{case,precision,solver,round,nanoseconds}" if boundary == "replay" else "[].{case,solver,round,nanoseconds}",
        })
    return records


def generate(repo: Path) -> dict:
    """Write source-indexed S1 outputs and return the summary dictionary."""
    repo = Path(repo).resolve()
    runs = repo / "benchmark/runs"
    full_root = runs / FULL
    out = repo / "benchmark/paper-artifacts"
    tex = repo / "latex/generated"
    out.mkdir(parents=True, exist_ok=True)
    tex.mkdir(parents=True, exist_ok=True)
    source_paths = set()

    def read(relative):
        path = repo / relative
        source_paths.add(path)
        return _read(path)

    full_relative = "benchmark/runs/" + FULL
    full = read(full_relative + "/campaign-summary.json")
    manifest = read(full_relative + "/campaign-session/manifest.json")
    full_config = read(full_relative + "/campaign-session/configuration.json")
    full_app_protocol = read(full_relative + "/application-session/protocol.json")
    assert full_config["inputs"] == full_config["rounds"] == 5
    assert full_app_protocol["rounds"] == 9
    cases = {c["id"]: c for c in manifest["cases"]}
    estimates = {(e["case_id"], e["method"]): e for e in full["estimates"] if e["status"] == "complete"}
    paired = [p for p in full["paired"] if p["status"] == "complete"
              and p["method"] in {"jvrust", "jvhybrid", "jvcpp"}
              and cases[p["case_id"]]["kind"] == "single"
              and p["reference"] == cases[p["case_id"]]["metadata"]["primary_comparator"]]
    assert len(paired) == 818
    subset_rows, comparison_rows = [], []
    for indices in combinations(range(5), 3):
        changed = 0
        for p in paired:
            method, reference = estimates[p["case_id"], p["method"]], estimates[p["case_id"], p["reference"]]
            assert method["attempt_id"] == reference["attempt_id"] == p["attempt_id"]
            assert method["input_hashes"] == reference["input_hashes"] == p["input_hashes"]
            assert len(method["round_means_seconds"]) == len(reference["round_means_seconds"]) == 5
            full_ratio = median(reference["round_means_seconds"]) / median(method["round_means_seconds"])
            assert abs(full_ratio / p["ratio"] - 1) < 1e-12
            short_ratio = median(reference["round_means_seconds"][i] for i in indices) / median(method["round_means_seconds"][i] for i in indices)
            full_class, short_class = _classify(full_ratio), _classify(short_ratio)
            changed += full_class != short_class
            comparison_rows.append({
                "case_id": p["case_id"], "method": p["method"], "reference": p["reference"],
                "round_indices": ",".join(map(str, indices)), "full_five_round_ratio": full_ratio,
                "three_round_ratio": short_ratio, "full_classification": full_class,
                "three_round_classification": short_class, "changed": full_class != short_class,
                "session_id": p["session_id"], "attempt_id": p["attempt_id"],
                "input_hashes": json.dumps(p["input_hashes"], sort_keys=True),
                "source": full_relative + "/campaign-summary.json",
                "source_fields": "estimates[].{case_id,method,round_means_seconds}; paired[].{reference,attempt_id,input_hashes}",
            })
        subset_rows.append({"round_indices": ",".join(map(str, indices)), "comparisons": len(paired), "classification_changes": changed})
    assert subset_rows[0]["classification_changes"] == 0
    assert max(r["classification_changes"] for r in subset_rows) == 2
    _csv(out / "s1_assignment_round_subsets.csv", subset_rows)
    _csv(out / "s1_assignment_comparisons.csv", comparison_rows)

    tracking_source = full_relative + "/application-session/tracking_samples.json"
    app_rows = _application_changes(read(tracking_source), "tracker_update", tracking_source)
    for path in sorted((full_root / "application-session/replay").glob("*.json")):
        rel = path.relative_to(repo).as_posix()
        app_rows += _application_changes(read(rel)["samples"], "replay", rel)
    assert len(app_rows) == 56 + 112
    _csv(out / "s1_application_round_truncation.csv", app_rows)
    max_changes = {boundary: max((r for r in app_rows if r["boundary"] == boundary), key=lambda r: abs(r["relative_change_percent"])) for boundary in ("tracker_update", "replay")}

    light_root = "benchmark/runs/" + LIGHT["L311"]
    light_manifest = read(light_root + "/campaign-session/manifest.json")
    light_config = read(light_root + "/campaign-session/configuration.json")
    light_app_protocol = read(light_root + "/application-session/protocol.json")
    light_cases = {c["id"]: c for c in light_manifest["cases"]}
    light_estimates = {(e["case_id"], e["method"]): e for e in read(light_root + "/campaign-summary.json")["estimates"] if e["status"] == "complete"}
    assert set(light_cases).issubset(cases)
    common_hash_checks = 0
    for key, e in light_estimates.items():
        if key in estimates:
            assert estimates[key]["input_hashes"] == e["input_hashes"]
            common_hash_checks += 1
    removed = []
    for case_id, case in cases.items():
        if case_id not in light_cases:
            assert case["family"] == "geometry"
            removed.append({"case_id": case_id, "rows": case["rows"], "cols": case["cols"], "objective": case["objective"], "q": case["metadata"]["q"], "motion": case["metadata"]["motion"], "reason": "geometry setting outside retained full panels"})
    assert len(removed) == 72
    _csv(out / "s1_omitted_geometry_cases.csv", removed)
    cutoff = []
    for case_id, case in light_cases.items():
        if case["family"] != "geometry" or case["rows"] <= 1024:
            continue
        for method in sorted(set(cases[case_id]["methods"]) - set(case["methods"])):
            assert method in {"jvx", "jvs", "scipy:square"}
            e = estimates.get((case_id, method))
            cutoff.append({
                "case_id": case_id, "rows": case["rows"], "cols": case["cols"], "objective": case["objective"], "method": method,
                "full_status": e["status"] if e else "not_complete", "full_seconds": e["seconds"] if e else "",
                "light_status": "not_planned", "reason": "geometry-only >1024-row baseline cutoff",
                "source": full_relative + "/campaign-summary.json", "source_fields": "estimates[].{case_id,method,status,seconds}",
            })
    assert cutoff
    _csv(out / "s1_geometry_cutoff.csv", cutoff)

    example_id = "geometry:1024x1280:q0.9:motion0.25:L1.5:gatenone"
    example = {method: estimates[example_id, method]["seconds"] * 1000 for method in ("jvrust", "jvx", "scipy:square")}
    light_audit = {}
    boundary_records = []
    for env, rel in LIGHT.items():
        prefix = "benchmark/runs/" + rel
        config = read(prefix + "/campaign-session/configuration.json")
        environment = read(prefix + "/campaign-session/environment.json")
        status = read(prefix + "/status.json")
        validation = read(prefix + "/campaign-validation.json")
        applications = read(prefix + "/application-validation/validation.json")
        assert config["inputs"] == 5 and config["rounds"] == 3 and config["target_seconds"] == .02
        assert config["batch_cpus"] == [0, 2, 4, 6] and config["single_cpu"] == 0
        assert environment["packages"]["numpy"] == "2.4.6" and environment["packages"]["scipy"] == "1.17.1"
        selected = [next(c for c in environment["topology"]["cores"] if cpu in c["logical_cpus"]) for cpu in config["batch_cpus"]]
        assert len({str(c["core_id"]) for c in selected}) == 4
        assert {c["efficiency_class"] for c in selected} == {max(c["efficiency_class"] for c in environment["topology"]["cores"])}
        counts = Counter(r["status"] for m in validation["methods"] for r in m["results"])
        assert not validation["correctness_failed"]
        assert counts["numerical_boundary_limitation"] == 11
        assert applications["matrices"] == 7539 and applications["checks"] == 135702
        assert all(r["objective_nonzero_gap_count"] == r["objective_mismatch_count"] == r["objective_max_abs_gap"] == 0 for r in applications["results"])
        for method in validation["methods"]:
            for record in method["results"]:
                if record["status"] == "numerical_boundary_limitation":
                    boundary_records.append({"environment": env, "source": prefix + "/campaign-validation.json", **record})
        light_audit[env] = {
            "run": rel, "packages": environment["packages"], "python": environment["python"],
            "platform": environment["platform"], "system_settings": environment["system_settings"],
            "logical_cpus": environment["logical_cpus"], "os_visible_physical_cores": len(environment["topology"]["cores"]),
            "single_cpu": config["single_cpu"], "batch_cpus": config["batch_cpus"],
            "selected_distinct_performance_cores": True, "supervisor_cpus": config["supervisor_cpus"],
            "session_id": config["session_id"], "environment_id": config["environment_id"],
            "started_utc": status["started_utc"], "finished_utc": status["finished_utc"], "wall_seconds": status["wall_seconds"],
            "validation_method_entries": len(validation["methods"]), "validation_fixture_status_counts": dict(counts),
            "validation_failed_methods": validation["correctness_failed"], "validation_methods_with_boundary_limitations": validation["passed_with_numerical_limitations"],
            "application_validation_matrices": applications["matrices"], "application_validation_checks": applications["checks"],
            "application_validation_nonzero_objective_gaps": 0,
            "sources": [prefix + suffix for suffix in ("/campaign-session/configuration.json", "/campaign-session/environment.json", "/status.json", "/campaign-validation.json", "/application-validation/validation.json")],
        }
    (out / "s1_light_protocol_audit.json").write_text(json.dumps(light_audit, indent=2) + "\n", encoding="utf-8")
    (out / "s1_numerical_boundary_records.json").write_text(json.dumps(boundary_records, indent=2) + "\n", encoding="utf-8")

    summary = {
        "scope": "FULL session supplementary protocol justification only; never pooled with light estimates",
        "full_run": FULL, "full_manifest_counts": manifest["counts"], "light_manifest_counts": light_manifest["counts"],
        "common_full_light_estimate_input_hash_checks": common_hash_checks,
        "assignment_comparisons": len(paired), "assignment_round_subsets": subset_rows,
        "assignment_first_three_classification_changes": subset_rows[0]["classification_changes"],
        "assignment_max_three_round_subset_classification_changes": max(r["classification_changes"] for r in subset_rows),
        "application_truncation_max_absolute_relative_change_percent": {b: abs(r["relative_change_percent"]) for b, r in max_changes.items()},
        "application_truncation_extrema": max_changes,
        "omitted_geometry_case_count": len(removed), "geometry_cutoff_method_conditions": len(cutoff),
        "full_geometry_example": {"case_id": example_id, "milliseconds": example, "source": full_relative + "/campaign-summary.json", "field": "estimates[].seconds"},
        "interpretation": "Descriptive truncation sensitivity within one saved session; not independent replication, confidence intervals, or statistical equivalence.",
        "sources": [{"path": p.relative_to(repo).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(source_paths)],
    }
    (out / "s1_sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fc, lc = manifest["counts"], light_manifest["counts"]
    latex = rf"""% Generated from saved protocol manifests; see s1_sensitivity.json.
\begin{{tabular}}{{lrr}}
\toprule
Quantity & Full protocol & Light protocol \\
\midrule
All cases & {fc['cases']} & {lc['cases']} \\
Single / geometry cases & {fc['single_cases']} / {fc['geometry_cases']} & {lc['single_cases']} / {lc['geometry_cases']} \\
Timing conditions & {fc['planned_case_method_jobs']-fc['memory_conditions']:,} & {lc['planned_case_method_jobs']-lc['memory_conditions']:,} \\
Memory conditions & {fc['memory_conditions']} & {lc['memory_conditions']} \\
Assignment inputs / rounds & {full_config['inputs']} / {full_config['rounds']} & {light_config['inputs']} / {light_config['rounds']} \\
Application rounds & {full_app_protocol['rounds']} & {light_app_protocol['rounds']} \\
\bottomrule
\end{{tabular}}
"""
    (tex / "s1_sensitivity.tex").write_text(latex, encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = generate(Path(__file__).resolve().parents[1])
    print(json.dumps({k: summary[k] for k in ("assignment_comparisons", "assignment_first_three_classification_changes", "assignment_max_three_round_subset_classification_changes", "application_truncation_max_absolute_relative_change_percent", "geometry_cutoff_method_conditions")}, indent=2))
