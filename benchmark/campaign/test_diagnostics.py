"""Small graph/representation diagnostic tests; no native calls or campaigns."""
import json

import numpy as np
import pytest

from . import diagnostics
from .catalog import INPUT_COUNT, build_manifest
from .datasets import generate_input


def test_graph_counts_use_independent_gate_and_empty_graph():
    case = dict(id="small", rows=2, cols=3, kind="single", objective="optional", limit=1., gate=.25)
    a = np.array([[0., .75, np.inf], [.5, np.inf, .25]])
    report = diagnostics.input_diagnostics(case, a, 0)
    matrix = report["matrices"][0]
    assert matrix["raw_finite_edges"] == 4
    assert matrix["admissible_edges"] == 2
    assert matrix["finite_edge_density"] == 2/6
    assert matrix["row_degree"]["isolated"] == 0
    assert matrix["column_degree"]["isolated"] == 1
    case["gate"] = -1.
    empty = diagnostics.input_diagnostics(case, a)["matrices"][0]
    assert empty["admissible_edges"] == 0
    assert empty["row_degree"]["isolated"] == 2
    assert empty["column_degree"]["isolated"] == 3
    json.dumps(report, allow_nan=False)


def test_geometry_latent_recreates_exact_original_costs(tmp_path):
    case = next(c for c in build_manifest()["cases"] if c["family"] == "geometry"
                and c["rows"] == 64 and c["cols"] == 80 and c["objective"] == "required")
    cost = generate_input(case, 0)
    record = diagnostics.save_geometry_latent(tmp_path, case, 0)
    with np.load(tmp_path / record["path"]) as latent:
        source, observation = latent["canonical_source_points"], latent["canonical_observation_points"]
        expected = np.sqrt(((source[:, None] - observation[None])**2).sum(axis=2))
        np.testing.assert_array_equal(cost, expected)
        np.testing.assert_array_equal(latent["oriented_pairs"], latent["canonical_pairs"])
    graph = diagnostics.input_diagnostics(case, cost, 0)
    assert graph["geometry"]["survivors"] == round(case["metadata"]["q"] * 64)
    assert graph["matrices"][0]["finite_edge_density"] == 1.


def test_layout_records_are_construction_only_and_preserve_inputs():
    case = dict(id="small", rows=2, cols=3, kind="single", objective="optional", limit=1., gate=.25)
    a = np.array([[0., .75, np.inf], [.5, np.inf, .25]])
    before = a.copy()
    report = diagnostics.layout_diagnostics(case, a, "cost", "diag_split", repetitions=3)
    assert report["transformed_dimensions"] == [5, 5]
    assert report["cost_buffer_bytes"] == 25 * 8
    assert len(report["construction_seconds"]) == 3
    assert report["construction_elapsed_seconds"] == sum(report["construction_seconds"])
    assert report["diagnostic_timing_not_primary"]
    np.testing.assert_array_equal(a, before)


def test_layout_jobs_deduplicate_backend_and_full_cli_needs_guard(tmp_path):
    case = dict(methods=["cost:scipy:diag_split", "cost:jvx:diag_split", "policy:rust:reference"])
    methods, layouts = diagnostics._jobs(case)
    assert methods == ["policy:rust:reference"]
    assert layouts == [("cost", "diag_split")]
    with pytest.raises(SystemExit):
        diagnostics.main(["--output", str(tmp_path / "diag"), "--full"])
    assert not (tmp_path / "diag").exists()


@pytest.mark.parametrize("full, expected_inputs", [(False, 2), (True, INPUT_COUNT)])
def test_diagnostic_requests_only_use_declared_frozen_inputs(tmp_path, monkeypatch, full, expected_inputs):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import psutil
    from . import resources, runner, topology

    case = next(c for c in build_manifest()["cases"] if c["id"] == "base:near256x320")
    manifest = {"cases": [case], "definition_sha256": "fixture"}
    mask = list(range(8))

    def affinity(cpus=None):
        if cpus is not None:
            mask[:] = cpus
        return list(mask)

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(cpu_affinity=affinity))
    monkeypatch.setattr(topology, "discover", lambda: {})
    monkeypatch.setattr(topology, "validate_selection", lambda *args: [
        {"logical_cpus": [cpu]} for cpu in (0, 2, 4, 6)])
    monkeypatch.setattr(runner, "environment", lambda: {})
    monkeypatch.setattr(resources, "HostLock", nullcontext)
    monkeypatch.setattr(resources, "admission", lambda case, methods, **kwargs: {
        "host_status": "ready", "methods": {method: {"status": "admitted"} for method in methods}})
    monkeypatch.setattr(diagnostics, "_bounded_cases", lambda manifest: [case])
    requests = []

    def launch(request, module):
        # Exercise the same generator bounds as a real worker, without running
        # any diagnostic construction, native call or timing measurement.
        for index in range(request["inputs"]):
            generate_input(request["case"], index)
        requests.append(request)
        return {"status": "complete"}, {"exit_code": 0}

    monkeypatch.setattr(runner, "launch", launch)
    summary = diagnostics.run_diagnostics(manifest, tmp_path / "diagnostics", full=full,
                                          cpus=[0, 2, 4, 6], single_cpu=0)
    assert summary["status"] == "finished"
    assert len(requests) == 1 and requests[0]["inputs"] == expected_inputs
    assert mask == list(range(8))
