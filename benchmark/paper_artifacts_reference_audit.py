"""Post-hoc reference sensitivity from saved summaries; never calls a solver.

Different Python environments and collection sessions remain confounded. These
diagnostics neither correct primary estimates nor identify a causal slowdown.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import median


RUNS = {
    "L311": "linux-py3.11.18-20261031-0455",
    "L314": "linux-py3.14.10-20261031-0630",
    "W311": "windows-py3.11.18-20261030-2322",
    "W314": "windows-py3.14.10-20261031-0322",
}
SIBLINGS = {"L311": "L314", "L314": "L311", "W311": "W314", "W314": "W311"}


def _classification(ratio):
    return "win" if ratio > 1.03 else "loss" if ratio < .97 else "tie"


def generate(repo=None):
    """Export the seven screened events and all method-level sensitivities.

    Only four saved summaries and manifests are read. Round means, input
    medians, hashes and attempt identifiers are retained in those summaries.
    Individual round/input cells are outside this summary-level export's scope.
    """
    repo = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    data = {}
    sources = {}
    for env, run in RUNS.items():
        base = Path("benchmark/runs") / run / "paper-run-light"
        paths = {"summary": base / "campaign-summary.json",
                 "manifest": base / "campaign-session/manifest.json"}
        loaded = {}
        for kind, relative in paths.items():
            raw = (repo / relative).read_bytes()
            loaded[kind] = json.loads(raw.decode("utf-8-sig"))
            sources[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
        assert loaded["summary"]["profile"] == loaded["manifest"]["profile"] == "light"
        data[env] = {"paths": paths, "base": base,
                     "cases": {c["id"]: c for c in loaded["manifest"]["cases"]},
                     "estimates": {(r["case_id"], r["method"]): r for r in loaded["summary"]["estimates"]
                                   if r["status"] == "complete"}}

    events = []
    method_rows = []
    screened_pairs = 0
    for env, sibling in SIBLINGS.items():
        target, other = data[env], data[sibling]
        for case_id, case in target["cases"].items():
            if case["kind"] != "single":
                continue
            screened_pairs += 1
            reference = case["metadata"]["primary_comparator"]
            a = target["estimates"][case_id, reference]
            b = other["estimates"][case_id, reference]
            rust_a = target["estimates"][case_id, "jvrust"]
            rust_b = other["estimates"][case_id, "jvrust"]
            reference_ratio = a["seconds"] / b["seconds"]
            rust_ratio = rust_a["seconds"] / rust_b["seconds"]
            # No upper bound is selected after seeing the observed ratios.
            if reference_ratio < 1.30 or not .90 <= rust_ratio <= 1.10:
                continue
            assert a["input_hashes"] == b["input_hashes"] == rust_a["input_hashes"] == rust_b["input_hashes"]
            for estimate in (a, b, rust_a, rust_b):
                assert estimate["observed_blocks"] == estimate["expected_blocks"] == 15
                assert len(estimate["round_means_seconds"]) == 3 and len(estimate["inputs"]) == 5
                assert median(estimate["round_means_seconds"]) == estimate["seconds"]
            other_inputs = {r["input_index"]: r for r in b["inputs"]}
            input_ratios = [r["median_seconds"] / other_inputs[r["input_index"]]["median_seconds"]
                            for r in sorted(a["inputs"], key=lambda row: row["input_index"])]
            round_ratios = [x / y for x, y in zip(a["round_means_seconds"], b["round_means_seconds"])]
            event_id = f"{env}:{case_id}"
            event = {"event_id": event_id, "environment": env, "sibling_environment": sibling,
                     "case_id": case_id, "family": case["family"], "objective": case["objective"],
                     "reference_method": reference, "target_reference_seconds": a["seconds"],
                     "sibling_reference_seconds": b["seconds"], "reference_latency_ratio": reference_ratio,
                     "jvrust_latency_ratio": rust_ratio, "target_attempt_id": a["attempt_id"],
                     "sibling_attempt_id": b["attempt_id"], "matched_input_hashes": a["input_hashes"],
                     "reference_input_median_ratios": input_ratios, "reference_round_mean_ratios": round_ratios,
                     "all_five_input_medians_larger": all(x > 1 for x in input_ratios),
                     "all_three_round_means_larger": all(x > 1 for x in round_ratios),
                     "target_summary": target["paths"]["summary"].as_posix(),
                     "sibling_summary": other["paths"]["summary"].as_posix(),
                     "target_raw_timing_source": (target["base"] / "campaign-session/attempts" / a["attempt_id"] / "measurements.jsonl").as_posix(),
                     "sibling_raw_timing_source": (other["base"] / "campaign-session/attempts" / b["attempt_id"] / "measurements.jsonl").as_posix()}
            events.append(event)
            for method in case["methods"]:
                key = case_id, method
                if key not in target["estimates"] or key not in other["estimates"]:
                    continue
                own, sibling_method = target["estimates"][key], other["estimates"][key]
                assert own["input_hashes"] == sibling_method["input_hashes"] == a["input_hashes"]
                ratios = {"target_reference_over_method": a["seconds"] / own["seconds"],
                          "sibling_reference_over_sibling_method": b["seconds"] / sibling_method["seconds"],
                          "sibling_reference_over_target_method": b["seconds"] / own["seconds"],
                          "rust_scaled_sibling_reference_over_target_method": b["seconds"] * rust_ratio / own["seconds"]}
                classes = {name + "_class": _classification(value) for name, value in ratios.items()}
                current_class = classes["target_reference_over_method_class"]
                method_rows.append({"event_id": event_id, "environment": env, "sibling_environment": sibling,
                                    "case_id": case_id, "method": method,
                                    "target_method_seconds": own["seconds"], "sibling_method_seconds": sibling_method["seconds"],
                                    "target_method_attempt_id": own["attempt_id"], "sibling_method_attempt_id": sibling_method["attempt_id"],
                                    **ratios, **classes,
                                    "observed_sibling_class_disagrees": current_class != classes["sibling_reference_over_sibling_method_class"],
                                    "reference_substitution_changes_class": current_class != classes["sibling_reference_over_target_method_class"],
                                    "rust_scaled_reference_substitution_changes_class": current_class != classes["rust_scaled_sibling_reference_over_target_method_class"]})

    change_fields = ["observed_sibling_class_disagrees", "reference_substitution_changes_class",
                     "rust_scaled_reference_substitution_changes_class"]
    counts = {method: {field: sum(row[field] for row in method_rows if row["method"] == method)
                       for field in change_fields} for method in ("sparse:csgraph", "jvrust", "jvcpp")}
    result = {"schema_version": 1, "diagnostic_status": "post_hoc_descriptive_sensitivity_only",
              "screen": {"directed_comparisons": "Each environment against the other Python version on the same OS",
                         "scope": "All single-assignment cases, with matched saved input hashes",
                         "primary_reference_latency_ratio_min_inclusive": 1.30,
                         "primary_reference_latency_ratio_max": None,
                         "jvrust_latency_ratio_min_inclusive": .90, "jvrust_latency_ratio_max_inclusive": 1.10,
                         "screened_directed_case_pairs": screened_pairs},
              "estimator": "Median of three round means, each with five equally weighted frozen inputs",
              "classification": "Reference/method >1.03 win, <0.97 loss, otherwise tie; no environment pooling",
              "scope_limit": "The five input medians and three round means do not establish that all fifteen individual round/input cells are larger. Raw timing paths are source pointers, not additional reads by this module.",
              "interpretation_limit": "Interpreter, build and session differences are confounded. The screen establishes reference discrepancies while JVRUST differs by at most 10%, not that every other method is stable. Observed sibling class disagreements differ from holding target method latency fixed and substituting a sibling reference. Neither comparison establishes a causal attempt effect or a corrected primary estimate.",
              "event_count": len(events), "classification_change_counts": counts,
              "events": events, "methods": method_rows, "source_sha256": sources}
    assert screened_pairs == 4 * 268 and len(events) == 7
    assert [counts["sparse:csgraph"][field] for field in change_fields] == [3, 2, 2]
    assert not any(count for method in ("jvrust", "jvcpp") for count in counts[method].values())
    output = repo / "benchmark/paper-artifacts"
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference_sensitivity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    for name, rows in [("reference_sensitivity_events.csv", events), ("reference_sensitivity_methods.csv", method_rows)]:
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    return result


if __name__ == "__main__":
    report = generate()
    print(json.dumps({"event_count": report["event_count"], "classification_change_counts": report["classification_change_counts"]}, indent=2))
