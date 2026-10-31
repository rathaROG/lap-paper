"""Small deterministic definition/data tests; no solvers or timing campaigns."""
from __future__ import annotations

import ast
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

from .catalog import (
    COST_LAYOUTS, SHAPE_LAYOUTS, LAYOUT_BACKENDS, INPUT_COUNT, REQUIRED_POLICY_ARMS, OPTIONAL_POLICY_ARMS,
    PADDED_ENTRY_LIMIT, build_manifest, canonical_identity, definition_hash, eligible_methods,
    layout_panel, padded_entries, planned_jobs, policy_panel, LEGACY_SOURCE_COMMIT,
)
from .datasets import generate_input, geometry_points, input_sha256


REPOSITORY = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def legacy_source(suite):
    """Read the pinned Git object without restoring any historical driver."""
    if suite not in ("base", "formulations", "scaling", "extended"):
        raise ValueError(f"Unknown historical generator suite: {suite}")
    reference = f"{LEGACY_SOURCE_COMMIT}:benchmark/{suite}/run.py"
    try:
        result = subprocess.run(["git", "-C", str(REPOSITORY), "show", reference],
                                capture_output=True, text=True, encoding="utf-8",
                                check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"Historical-input regression checks require Git object {reference}. "
            "Fetch the pinned commit when using a shallow/source-only checkout; "
            "do not copy the old drivers into the benchmark directory."
        ) from error
    return reference, result.stdout


def legacy_functions(suite, *names):
    """Load only generator definitions, never old modules' solver imports."""
    reference, source = legacy_source(suite)
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    scope = dict(np=np, math=math, hashlib=hashlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), reference, "exec"), scope)
    return scope


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = build_manifest()
        cls.cases = cls.manifest["cases"]
        cls.by_id = {case["id"]: case for case in cls.cases}

    def test_manifest_is_frozen_serializable_and_allocation_free(self):
        with patch("numpy.empty", side_effect=AssertionError("allocated input")), \
             patch("numpy.random.default_rng", side_effect=AssertionError("generated input")):
            other = build_manifest()
        self.assertEqual(self.manifest, other)
        self.assertEqual(json.loads(json.dumps(other, allow_nan=False)), other)
        self.assertEqual(len(self.by_id), len(self.cases))
        self.assertEqual(len({canonical_identity(c) for c in self.cases}), len(self.cases))
        payload = {key: value for key, value in other.items() if key != "definition_sha256"}
        self.assertEqual(definition_hash(payload), other["definition_sha256"])

    def test_frozen_coverage_counts(self):
        counts = self.manifest["counts"]
        for key, expected in dict(cases=374, single_cases=340, legacy_single_cases=148,
                                  geometry_cases=192, batch_cases=14,
                                  workspace_workloads=14, standard_batch_workloads=3,
                                  memory_cases=20, memory_conditions=91,
                                  required_single_cases=138, optional_single_cases=202,
                                  layout_panel_cases=25, policy_panel_cases=33,
                                  cost_layout_cases=14, shape_layout_cases=9,
                                  layout_case_method_jobs=624, policy_case_method_jobs=428,
                                  planned_case_method_jobs=3917).items():
            self.assertEqual(counts[key], expected, key)
        reused = [c for c in self.cases if c["metadata"].get("deduplicated_formulation_reuse")]
        self.assertEqual(len(reused), 5)
        self.assertTrue(all(len(c["sources"]) == 2 for c in reused))

    def test_policy_and_layout_eligibility_is_explicit(self):
        for case in self.cases:
            methods = eligible_methods(case)
            self.assertEqual(case["methods"], methods)
            self.assertEqual(len(methods), len(set(methods)))
            if case["kind"] != "single":
                continue
            optional = case["objective"] == "optional"
            rectangular = case["rows"] != case["cols"]
            cost = [m for m in methods if m.startswith("cost:")]
            shape = [m for m in methods if m.startswith("shape:")]
            sweep = len(LAYOUT_BACKENDS)
            self.assertEqual(len(cost), sweep * len(COST_LAYOUTS) if optional and layout_panel(case) else 0)
            self.assertEqual(len(shape), sweep * len(SHAPE_LAYOUTS)
                             if not optional and rectangular and layout_panel(case) else 0)
            self.assertTrue(all(padded_entries(case, m) <= PADDED_ENTRY_LIMIT for m in methods))
            for language in ("rust", "cpp"):
                arms = {m.split(":")[-1] for m in methods if m.startswith(f"policy:{language}:")}
                expected = (set() if not policy_panel(case) else
                            set(OPTIONAL_POLICY_ARMS if optional else REQUIRED_POLICY_ARMS)
                            if optional or rectangular else {"reference"})
                self.assertEqual(arms, expected)
        jobs = list(planned_jobs(self.manifest))
        self.assertEqual(len(jobs), self.manifest["counts"]["planned_case_method_jobs"])
        self.assertTrue(all(j["status"] == "planned" for j in jobs))

    def test_legacy_base_first_four_match_original_generator(self):
        original = legacy_functions("base", "make_cost", "case_specs", "single_inputs")
        for spec, inputs in original["single_inputs"](20261021, 4):
            case = self.by_id[f"base:{spec['case']}"]
            for index, expected in enumerate(inputs):
                np.testing.assert_array_equal(generate_input(case, index), expected)

    def test_formulation_extra_stream_and_scaling_small_input(self):
        original = legacy_functions("formulations", "make_cost", "base_case_specs", "datasets")
        for spec, inputs in original["datasets"](20261021, 4):
            if spec["source"] != "additional":
                continue
            case = self.by_id[f"formulations:{spec['case']}"]
            for index, expected in enumerate(inputs):
                np.testing.assert_array_equal(generate_input(case, index), expected)
        scaling = legacy_functions("scaling", "make_cost", "specifications")
        rng = np.random.default_rng(20261222)
        for spec in list(scaling["specifications"]())[:4]:
            for index in range(4):
                expected = scaling["make_cost"](spec, rng)
                np.testing.assert_array_equal(generate_input(self.by_id[f"scaling:{spec['case']}"], index), expected)

    def test_extended_parameters_and_transpose_match_original(self):
        original = legacy_functions("extended", "plan", "make_input", "array_sha256")
        original.update(MEMORY_REQUIRED=(), MEMORY_OPTIONAL=())
        retained = 0
        for spec in original["plan"]()["timing"]:
            case = self.by_id.get(f"extended:{spec['case']}")
            if case is None:
                # Structure families keep only 256x320; numbering is unchanged.
                self.assertTrue(spec["family"] in ("congestion", "forbidden", "ties", "rejection")
                                and (spec["n"], spec["m"]) != (256, 320), spec["case"])
                continue
            retained += 1
            self.assertEqual((case["rows"], case["cols"], case["limit"]), (spec["n"], spec["m"], spec["limit"]))
            if spec["family"] == "signed" or (spec["family"] == "forbidden" and spec["n"] == 256):
                expected, _ = original["make_input"](spec, 20261024, 3)
                np.testing.assert_array_equal(generate_input(case, 3), expected)
        self.assertEqual(retained, 84)

    def test_padded_baselines_are_bounded_for_timing_but_kept_for_memory(self):
        methods = {case["id"]: case["methods"] for case in self.cases}
        self.assertFalse(any("jvc" in m.split(":") for ms in methods.values() for m in ms))
        boundary = methods["extended:020_extreme_rectangle_required_64x4096"]
        self.assertTrue({"jvx", "jvs"} <= set(boundary))
        for case_id in ("extended:021_extreme_rectangle_optional_64x4096",
                        "extended:028_extreme_rectangle_required_64x16384",
                        "geometry:4096x5120:q0.9:motion0.25:L1.5:gatenone"):
            self.assertFalse({"jvx", "jvs", "scipy:square"} & set(methods[case_id]), case_id)
        self.assertIn("scipy:square", methods["memory:extended:optional:256x4096"])
        self.assertIn("jvrust", methods["extended:028_extreme_rectangle_required_64x16384"])

    def test_geometry_shared_inputs_and_gate_controls(self):
        selected = [c for c in self.cases if c["family"] == "geometry" and c["input_group"]["key"] == "geometry:256x320:q0.9:motion0.25"]
        canonical = generate_input(selected[0], 0)
        sources, observations, pairs = geometry_points(selected[0]["input_group"], 0)
        self.assertEqual(len(pairs), round(.9 * 256))
        self.assertEqual(len(np.unique(pairs[:, 0])), len(pairs))
        self.assertEqual(len(np.unique(pairs[:, 1])), len(pairs))
        self.assertFalse(np.array_equal(np.sort(pairs[:, 0]), np.arange(len(pairs))))
        expected = np.sqrt(np.sum((sources[:, None, :] - observations[None, :, :]) ** 2, axis=2))
        np.testing.assert_array_equal(canonical, expected)
        self.assertEqual(len(selected), 10)  # required plus nine optional settings.
        for case in selected:
            actual = generate_input(case, 0)
            np.testing.assert_array_equal(actual, canonical.T if case["transpose"] else canonical)
            self.assertTrue(actual.flags.c_contiguous)
            self.assertEqual(case["metadata"]["delta"], 1 / 16)
            if case["gate"] is not None:
                self.assertTrue(np.all(np.isfinite(actual)), "gate must be applied inside adapter")
                self.assertEqual(case["metadata"]["redundant_gate"], case["gate"] >= case["limit"])
        self.assertNotEqual(input_sha256(canonical), input_sha256(generate_input(selected[0], 1)))

    def test_batch_original_instance_and_transposes(self):
        original = legacy_functions("base", "make_cost", "batch_specs", "batch_inputs")
        for spec, expected in original["batch_inputs"](20261021):
            case = self.by_id[f"batch:{spec['case']}"]
            actual = generate_input(case, 0)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.ndim, 3)
            transpose = generate_input(self.by_id[f"batch:{spec['case']}:T"], 0)
            np.testing.assert_array_equal(transpose, actual.transpose(0, 2, 1))
            self.assertTrue(transpose.flags.c_contiguous)
            self.assertEqual(len(case["methods"]), 22)
        small = self.by_id["batch:batch1024x8x10"]
        self.assertNotEqual(input_sha256(generate_input(small, 0)), input_sha256(generate_input(small, 1)))

    def test_memory_trial_identity_and_input_indices(self):
        case = self.by_id["memory:extended:optional:256x256"]
        expected = np.random.default_rng([20261024, 30000, 256, 256, 4]).random((256, 256))
        np.testing.assert_array_equal(generate_input(case, 4), expected)
        for index in (-1, 5, 8, True, 1.5):
            with self.assertRaises(ValueError):
                generate_input(case, index)
        case = self.by_id["base:square128"]
        last = INPUT_COUNT - 1
        self.assertEqual(input_sha256(generate_input(case, last)), input_sha256(generate_input(copy.deepcopy(case), last)))
        self.assertNotEqual(input_sha256(generate_input(case, 3)), input_sha256(generate_input(case, 4)))
        with self.assertRaises(ValueError):
            generate_input(case, INPUT_COUNT)


class LightCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = build_manifest()
        cls.light = build_manifest(profile="light")

    def test_full_definition_is_unchanged_and_light_has_a_distinct_identity(self):
        self.assertEqual(self.full["definition_sha256"],
                         "2c468a53ec2768b965dbfed1ea418d39325bac754cbb27711cc7b5afc1284ec0")
        self.assertNotEqual(self.full["definition_sha256"], self.light["definition_sha256"])
        self.assertNotEqual(self.full["generation"], self.light["generation"])
        self.assertEqual(self.light["input_count"], 5)
        self.assertEqual(self.light["rounds"], 3)
        self.assertEqual(self.light["memory_trials"], 5)

    def test_light_preserves_every_size_and_every_non_geometry_case(self):
        full = {case["id"]: case for case in self.full["cases"]}
        light = {case["id"]: case for case in self.light["cases"]}
        shapes = lambda manifest: {(c["family"], c["kind"], c["objective"], c["rows"], c["cols"])
                                   for c in manifest["cases"]}
        self.assertEqual(shapes(self.full), shapes(self.light))
        for case in self.full["cases"]:
            if case["family"] != "geometry":
                self.assertEqual(case, light[case["id"]])
        for case in self.light["cases"]:
            self.assertEqual(case["input_group"], full[case["id"]]["input_group"])

    def test_complete_layout_policy_and_sensitivity_panels_remain(self):
        full = {c["id"]: c for c in self.full["cases"] if layout_panel(c) or policy_panel(c)}
        light = {c["id"]: c for c in self.light["cases"] if layout_panel(c) or policy_panel(c)}
        self.assertEqual(full, light)
        for shape in ((256, 320), (1024, 1280)):
            geometry = lambda manifest: [c for c in manifest["cases"] if c["family"] == "geometry"
                                         and (c["rows"], c["cols"]) == shape]
            self.assertEqual(geometry(self.full), geometry(self.light))
        for case in self.light["cases"]:
            if case["family"] == "geometry" and (case["rows"], case["cols"]) not in ((256, 320), (1024, 1280)):
                self.assertIn((case["metadata"]["q"], case["metadata"]["motion"]), ((.9, .25), (.6, 1.)))

    def test_geometry_baseline_cap_includes_unpadded_required_squares_only_in_light(self):
        for case in self.light["cases"]:
            if case["family"] == "geometry" and case["rows"] > 1024:
                self.assertFalse({"jvx", "jvs", "scipy:square"} & set(case["methods"]))
                self.assertTrue({"jvrust", "jvcpp", case["metadata"]["primary_comparator"]} <= set(case["methods"]))
        largest = next(c for c in self.full["cases"] if c["id"] ==
                       "geometry:4096x4096:q0.9:motion0.25:required")
        self.assertTrue({"jvx", "jvs"} <= set(largest["methods"]))
        self.assertEqual(padded_entries(largest, "jvx"), 0)


if __name__ == "__main__":
    unittest.main()
