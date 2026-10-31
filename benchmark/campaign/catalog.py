"""Frozen, allocation-free definition of the refreshed dense campaign.

This module lists work; it never imports a solver or runs a measurement.  A
method's presence means *planned coverage*, not that its adapter/build has
passed validation.  Unsupported and resource-censored jobs stay in this list.

Historical ``benchmark/<suite>/run.py`` source labels refer to
LEGACY_SOURCE_COMMIT below. Input generation is implemented in datasets.py.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math


SCHEMA_VERSION = 1
LEGACY_SOURCE_COMMIT = "398b813ac1fd7437d8f95e8ad72b8ed3a6e381a8"
GENERATION = "dense-compact-20261030-v2"
LIGHT_GENERATION = GENERATION + "-light"
LIGHT_ROUNDS = 3
LIGHT_GEOMETRY_SETTINGS = ((.9, .25), (.6, 1.))
LIGHT_GEOMETRY_FULL_PANELS = ((256, 320), (1024, 1280))
LIGHT_GEOMETRY_BASELINE_ROW_LIMIT = 1024
# One session per environment: every timed case/method condition uses
# INPUT_COUNT inputs, ROUNDS shuffled rounds and ~BLOCK_SECONDS blocks.
INPUT_COUNT = 5
ROUNDS = 5
BLOCK_SECONDS = .02
WARMUP_CALLS = 1
MEMORY_TRIALS = 5
SEED = 20261030
COST_LAYOUTS = (
    "fill_split", "diag_split", "diag_split_sparse", "fill_row_pad_zero",
    "fill_row_pad_inf", "fill_col_pad_zero", "fill_col_pad_inf",
    "diag_row_pad_zero", "diag_row_pad_inf", "diag_col_pad_0", "diag_col_pad_inf",
)
SHAPE_LAYOUTS = (
    "smallest_fill_zero", "smallest_fill_inf", "sum_fill_inf",
    "sum_split_inf", "sum_row_inf", "sum_col_inf",
)
LAYOUT_BACKENDS = ("scipy", "jvx", "jvs")
REQUIRED_METHODS = ("jvrust", "jvhybrid", "jvcpp", "jvx", "jvs", "scipy")
OPTIONAL_METHODS = (
    "jvrust", "jvcpp", "jvx", "scipy:capped", "scipy:dummies",
    "scipy:diagonal_dummies", "scipy:square",
)
# Sparse solvers receive only admissible edges (cost < L after any gate).
SPARSE_METHODS = ("sparse:csgraph", "sparse:lapmod")
REQUIRED_POLICY_ARMS = (
    "reference", "budget1", "budget2", "budget8", "ties_off", "cutoff_off",
    "ties_cutoff_off", "no_reductions",
)
OPTIONAL_POLICY_ARMS = tuple(
    arm for arm in REQUIRED_POLICY_ARMS if arm not in ("cutoff_off", "ties_cutoff_off")
)
# Padded baselines are timed while their square has at most 4096**2 entries
# (128 MiB float64). Memory cases keep them to measure the expansion itself.
PADDED_ENTRY_LIMIT = 1 << 24
PADDED_BACKENDS = ("jvx", "jvs")
# Complete layout sweeps and policy ablations are explanatory studies. They run
# on fixed representative panels; other cases use the canonical methods only.
LAYOUT_PANEL_GEOMETRY = ("geometry:256x320:",)
POLICY_PANEL_GEOMETRY = ("geometry:256x320:", "geometry:1024x1280:")


def definition_hash(value):
    """Portable identity of a JSON definition, not a materialized array hash."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group(generator, key, rows, cols, seed=SEED, **parameters):
    group = dict(generator=generator, version=1, key=key, seed=seed,
                 rows=rows, cols=cols, dtype="float64", parameters=parameters)
    group["definition_sha256"] = definition_hash(group)
    return group


def _case(case_id, family, rows, cols, limit, group, *, source, kind="single",
          transpose=False, gate=None, **metadata):
    return dict(id=case_id, family=family, kind=kind,
                objective="required" if limit is None else "optional",
                rows=rows, cols=cols, limit=limit, gate=gate, input_group=group,
                transpose=transpose, sources=[source],
                metadata=dict(dtype="float64", layout="C", output="int64_pairs",
                              input_count=1 if kind == "memory" else INPUT_COUNT, **metadata))


def _base_specs():
    return [
        ("square128", 128, 128, None, {}),
        ("square512", 512, 512, None, {}),
        ("wide64x512", 64, 512, None, {}),
        ("tall512x64", 512, 64, None, {}),
        ("near256x320", 256, 320, None, {}),
        ("blocks256x320", 256, 320, None, dict(blocks=True)),
        ("ties128x160", 128, 160, None, dict(ties=True)),
        ("optional128x160", 128, 160, .5, dict(forbidden=.5)),
        ("optional256x320", 256, 320, .5, dict(forbidden=.5)),
        ("tight128x160", 128, 160, .05, dict(forbidden=.5)),
        ("negative128x160", 128, 160, -.25, dict(forbidden=.2, signed=True)),
        ("optional_square128", 128, 128, .5, dict(forbidden=.5)),
    ]


def _legacy_stream_cases(specs, suite, seed):
    """Retain the first four old inputs without generating earlier matrices.

    Each old random double consumes one PCG64 output.  Cost and optional mask
    draws are the only random operations in these three legacy generators.
    Extra inputs use independently seeded streams, leaving old offsets fixed.
    """
    offset = 0
    for name, rows, cols, limit, parameters in specs:
        draws = rows * cols * (1 + bool(parameters.get("forbidden")))
        group = _group("legacy_stream", f"{suite}:{name}", rows, cols, seed,
                       stream_offset=offset, draws_per_input=draws,
                       retained_inputs=4, supplemental_seed=SEED,
                       **parameters)
        yield _case(f"{suite}:{name}", suite, rows, cols, limit, group,
                    source=f"benchmark/{suite}/run.py:{name}", legacy_suite=suite,
                    legacy_name=name)
        offset += 4 * draws


def _legacy_cases():
    base = list(_legacy_stream_cases(_base_specs(), "base", 20261022))
    for case in base:
        if case["objective"] == "optional":
            # Same values, objective, precision and interface: one measured case.
            case["sources"].append(f"benchmark/formulations/run.py:{case['metadata']['legacy_name']}")
            case["metadata"]["deduplicated_formulation_reuse"] = True
    yield from base
    extra = [
        ("reject128x160_L005", 128, 160, .005, dict(forbidden=.5)),
        ("reject128x160_L001", 128, 160, .001, dict(forbidden=.5)),
        ("reject_tall160x128", 160, 128, .005, dict(forbidden=.5)),
        ("optional_wide64x512", 64, 512, .005, dict(forbidden=.5)),
        ("optional_blocks256x320", 256, 320, .5, dict(blocks=True)),
    ]
    yield from _legacy_stream_cases(extra, "formulations", 20261122)
    scaling = []
    for regime in ("required", "optional"):
        for family in ("square", "wide", "fixed"):
            sizes = ((16, 32, 64, 128, 256, 512, 1024) if family == "square" else
                     (16, 32, 64, 128, 256, 512) if family == "wide" else
                     (64, 128, 256, 512, 1024, 2048, 4096))
            for size in sizes:
                n, m = ((size, size) if family == "square" else
                        (size, 4 * size) if family == "wide" else (64, size))
                scaling.append((f"{regime}_{family}_{n}x{m}", n, m,
                                .5 if regime == "optional" else None, {}))
    for n in (16, 32, 64, 128, 256, 512, 1024):
        m = 5 * n // 4
        scaling.append((f"optional_blocks_{n}x{m}", n, m, .5, dict(blocks=True)))
    yield from _legacy_stream_cases(scaling, "scaling", 20261222)
    yield from _extended_cases()


def _extended_cases():
    """Structure families run at 256x320 only; numbering keeps the legacy IDs."""
    cases, count = [], [0]

    def add(family, n, m, limit, key, **parameters):
        index, count[0] = count[0], count[0] + 1
        if family in ("congestion", "forbidden", "ties", "rejection") and (n, m) != (256, 320):
            return
        parameters.setdefault("distribution", "uniform")
        small, large = sorted((n, m))
        group = _group("extended", key, small, large, 20261024, **parameters)
        regime = "required" if limit is None else "optional"
        old_name = f"{index:03d}_{family}_{regime}_{n}x{m}"
        cases.append(_case(f"extended:{old_name}", family, n, m, limit, group,
                           source=f"benchmark/extended/run.py:{old_name}",
                           transpose=n > m, legacy_suite="extended", legacy_name=old_name))

    for n in (1000, 2000, 3000, 5000):
        for limit in (None, 2 / n):
            add("large_square", n, n, limit, f"large_square_{n}")
    for family, shapes in (
        ("large_rectangle", ((1000, 2000), (2000, 3000), (4000, 5000))),
        ("extreme_rectangle", ((64, 4096), (64, 8192), (64, 16384))),
    ):
        for n, m in shapes:
            for rows, cols in ((n, m), (m, n)):
                for limit in (None, 2 / max(n, m)):
                    add(family, rows, cols, limit, f"{family}_{n}x{m}")
    for n, m in ((256, 320), (512, 640)):
        for congestion in (.25, .5, .625, .75, .9):
            for limit in (None, .5):
                add("congestion", n, m, limit, f"congestion_{n}x{m}",
                    distribution="blocks", row_fraction=congestion, column_fraction=.25)
        for forbidden in (0., .5, .9, .99):
            for limit in (None, 2 / m):
                add("forbidden", n, m, limit, f"forbidden_{n}x{m}",
                    forbidden=forbidden, planted_feasible=True)
        for levels in (1, 2, 8, 256):
            for limit in (None, .25):
                add("ties", n, m, limit, f"ties_{n}x{m}", levels=levels)
    for n, m in ((256, 320), (320, 256)):
        for limit in (None, -.25):
            add("signed", n, m, limit, "signed_256x320", signed=True,
                forbidden=.5, planted_feasible=True)
    for n, m in ((128, 160), (256, 320)):
        for distribution in ("uniform", "blocks"):
            for limit in (0., .0005, .001, .002, .005, .01, .02, .05, .1, .5, 4.):
                parameters = (dict(forbidden=.5) if distribution == "uniform" else
                              dict(row_fraction=.75, column_fraction=.25))
                add("rejection", n, m, limit, f"rejection_{distribution}_{n}x{m}",
                    distribution=distribution, **parameters)
    return cases


def _geometry_cases(profile="full"):
    """Tracks (rows) against detections (columns); orientation is studied elsewhere.

    A doubling size sweep from tracking-scale 8 rows to 4096x5120, square and
    5:4. Wide and extreme aspects are in the scaling and extended families.
    """
    for r in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        delta = 1 / math.sqrt(r)
        for numerator, denominator in ((1, 1), (5, 4)):
            c = r * numerator // denominator
            for q in (.9, .6):
                for motion in (.25, 1.):
                    if (profile == "light" and (r, c) not in LIGHT_GEOMETRY_FULL_PANELS
                            and (q, motion) not in LIGHT_GEOMETRY_SETTINGS):
                        continue
                    key = f"geometry:{r}x{c}:q{q:g}:motion{motion:g}"
                    group = _group("geometry", key, r, c, q=q,
                                   motion=motion, delta=delta, boundary="unbounded_displacement")
                    sensitivity = r == 256 and numerator == 5
                    contracts = [(None, None), (1.5, None)]
                    if sensitivity:
                        contracts.extend((l, g) for l in (.5, 1.5, 3.)
                                         for g in (.75, 1.5, None)
                                         if (l, g) != (1.5, None))
                    for lm, gm in contracts:
                        suffix = "required" if lm is None else f"L{lm:g}:gate{'none' if gm is None else f'{gm:g}'}"
                        yield _case(f"{key}:{suffix}", "geometry", r, c,
                                    None if lm is None else lm * delta, group,
                                    source="proposal:geometry",
                                    gate=None if gm is None else gm * delta,
                                    q=q, motion=motion, delta=delta,
                                    survivors=round(q * r), miss_fraction=1 - round(q * r) / r,
                                    false_fraction=1 - round(q * r) / c,
                                    limit_over_delta=lm, gate_over_delta=gm,
                                    redundant_gate=gm is not None and gm >= lm,
                                    geometric_sensitivity=sensitivity and lm is not None)


def _batch_cases():
    specs = [
        ("batch64x64x80", 64, 64, 80, None, {}),
        ("batch16x128x160_optional", 16, 128, 160, .5, dict(forbidden=.5)),
        ("batch1024x8x10", 1024, 8, 10, None, {}),
    ]
    offset = 0
    for name, batch_size, n, m, limit, parameters in specs:
        draws = n * m * (1 + bool(parameters.get("forbidden")))
        group = _group("legacy_batch", name, n, m, 20261024,
                       batch_size=batch_size, stream_offset=offset,
                       draws_per_input=draws, supplemental_seed=SEED, **parameters)
        for transpose in (False, True):
            rows, cols = (m, n) if transpose else (n, m)
            yield _case(f"batch:{name}{':T' if transpose else ''}", "workspace", rows,
                        cols, limit, group, kind="batch", transpose=transpose,
                        source=f"benchmark/base/run.py:{name}", batch_size=batch_size,
                        standard_batch=not transpose, workspace=True,
                        original_batch=not transpose)
        offset += batch_size * draws
    for c in (4096, 8192):
        r = 256
        group = _group("uniform", f"workspace:8x{r}x{c}", r, c, batch_size=8)
        for transpose in (False, True):
            n, m = (c, r) if transpose else (r, c)
            for label, limit in (("required", None), ("optional", .5)):
                yield _case(f"batch:8x{n}x{m}:{label}", "workspace", n, m, limit,
                            group, kind="batch", transpose=transpose,
                            source="proposal:workspace", batch_size=8,
                            standard_batch=False, workspace=True,
                            orientation_payload_bytes=r * c * 8)


def _memory_cases():
    for m in (256, 512, 1024, 2048, 4096):
        for n, c in (((256, m),) if m == 256 else ((256, m), (m, 256))):
            group = _group("memory_extended", f"memory:{n}x{c}", n, c, 20261024)
            for limit in (None, .5):
                regime = "required" if limit is None else "optional"
                methods = (("jvrust", "jvcpp", "scipy", "jvx") if limit is None else
                           ("jvrust", "jvcpp", "scipy:capped", "scipy:dummies", "scipy:square"))
                yield _case(f"memory:extended:{regime}:{n}x{c}", "memory", n, c,
                            limit, group, source="benchmark/extended/run.py:memory",
                            kind="memory", memory_trials=MEMORY_TRIALS,
                            memory_methods=list(methods), legacy_suite="extended")
    for limit in (None, .5):
        regime = "required" if limit is None else "optional"
        group = _group("memory_base", "memory:base:256x4096", 256, 4096, 20261021)
        methods = ("jvrust", "jvhybrid", "jvcpp", "jvx",
                   "scipy" if limit is None else "scipy:square")
        yield _case(f"memory:base:{regime}:256x4096", "memory", 256, 4096,
                    limit, group, source="benchmark/base/run.py:memory", kind="memory",
                    memory_trials=MEMORY_TRIALS, memory_methods=list(methods),
                    legacy_suite="base", historical_single_trial=True)


def _primary_geometry(case):
    metadata = case["metadata"]
    return metadata.get("gate_over_delta") is None and metadata.get("limit_over_delta") in (None, 1.5)


def layout_panel(case):
    """Fixed cases that receive the complete cost/shape layout sweep."""
    suite, key = case["id"].split(":", 1)[0], case["input_group"]["key"]
    if case["kind"] != "single":
        return False
    if suite in ("base", "formulations"):
        return True
    return suite == "geometry" and key.startswith(LAYOUT_PANEL_GEOMETRY) and _primary_geometry(case)


def policy_panel(case):
    """Fixed cases that receive controlled policy builds and ablations."""
    suite, key = case["id"].split(":", 1)[0], case["input_group"]["key"]
    if case["kind"] != "single":
        return False
    if suite in ("base", "formulations"):
        return True
    return suite == "geometry" and key.startswith(POLICY_PANEL_GEOMETRY) and _primary_geometry(case)


def padded_entries(case, method):
    """Entries in the square a padded baseline solves; zero when none is built."""
    n, m = case["rows"], case["cols"]
    parts = method.split(":")
    optional = case["objective"] == "optional"
    if parts[0] == "cost" or method == "scipy:square" or (optional and method in PADDED_BACKENDS):
        return (n + m) ** 2
    if parts[0] == "shape":
        order = n + m if parts[-1].startswith("sum_") else max(n, m)
        return order ** 2
    if method in PADDED_BACKENDS and n != m:
        return max(n, m) ** 2
    return 0


def eligible_methods(case, *, profile="full"):
    """Return prespecified method IDs, independent of timing or local installs."""
    metadata = case["metadata"]
    if case["kind"] == "memory":
        return list(metadata["memory_methods"])
    if case["kind"] == "batch":
        methods = [f"workspace:{language}:{arm}:{workers}"
                   for language in ("rust", "cpp") for arm in ("w0", "w1", "w2")
                   for workers in (1, 4)]
        if metadata["standard_batch"]:
            methods += ["loop:jvrust", "loop:scipy"]
            methods += [f"batch:{backend}:{workers}"
                        for backend in ("jvrust", "jvhybrid", "jvcpp", "jvx")
                        for workers in (1, 4)]
        return methods
    optional = case["objective"] == "optional"
    rectangular = case["rows"] != case["cols"]
    methods = list(OPTIONAL_METHODS + SPARSE_METHODS if optional else REQUIRED_METHODS)
    if layout_panel(case) and optional:
        methods += [f"cost:{backend}:{layout}" for backend in LAYOUT_BACKENDS for layout in COST_LAYOUTS]
    elif layout_panel(case) and rectangular:
        methods += [f"shape:{backend}:{layout}" for backend in LAYOUT_BACKENDS for layout in SHAPE_LAYOUTS]
    if policy_panel(case) and (optional or rectangular):
        arms = OPTIONAL_POLICY_ARMS if optional else REQUIRED_POLICY_ARMS
        methods += [f"policy:{language}:{arm}" for language in ("rust", "cpp") for arm in arms]
    elif policy_panel(case):
        # Square initialization is a separate algorithm. Only the default-policy
        # scaffold control is meaningful; do not assign rectangular ablations.
        methods += [f"policy:{language}:reference" for language in ("rust", "cpp")]
    return [method for method in methods if padded_entries(case, method) <= PADDED_ENTRY_LIMIT
            and not (profile == "light" and case["family"] == "geometry"
                     and case["rows"] > LIGHT_GEOMETRY_BASELINE_ROW_LIMIT
                     and method in ("jvx", "jvs", "scipy:square"))]


def canonical_identity(case):
    """Identity includes measurement kind, objective, values and input layout."""
    return definition_hash({key: case[key] for key in (
        "kind", "objective", "rows", "cols", "limit", "gate", "input_group", "transpose"
    )} | {"dtype": case["metadata"]["dtype"], "layout": case["metadata"]["layout"],
         "output": case["metadata"]["output"]})


def manifest_counts(manifest):
    cases = manifest["cases"]
    single = [case for case in cases if case["kind"] == "single"]
    return dict(cases=len(cases), single_cases=len(single),
                legacy_single_cases=sum("legacy_suite" in case["metadata"] for case in single),
                geometry_cases=sum(case["family"] == "geometry" for case in single),
                layout_panel_cases=sum(layout_panel(case) for case in single),
                policy_panel_cases=sum(policy_panel(case) for case in single),
                batch_cases=sum(case["kind"] == "batch" for case in cases),
                workspace_workloads=sum(case["metadata"].get("workspace", False) for case in cases),
                standard_batch_workloads=sum(case["metadata"].get("standard_batch", False) for case in cases),
                memory_cases=sum(case["kind"] == "memory" for case in cases),
                memory_conditions=sum(len(case["methods"]) for case in cases if case["kind"] == "memory"),
                planned_case_method_jobs=sum(len(case["methods"]) for case in cases),
                required_single_cases=sum(case["objective"] == "required" for case in single),
                optional_single_cases=sum(case["objective"] == "optional" for case in single),
                cost_layout_cases=sum(any(m.startswith("cost:") for m in case["methods"]) for case in single),
                shape_layout_cases=sum(any(m.startswith("shape:") for m in case["methods"]) for case in single),
                policy_case_method_jobs=sum(m.startswith("policy:") for case in single for m in case["methods"]),
                layout_case_method_jobs=sum(m.startswith(("cost:", "shape:")) for case in single for m in case["methods"]),
                by_family=dict(sorted(Counter(case["family"] for case in cases).items())))


def planned_jobs(manifest):
    """Expand planned case/method coverage lazily; this never executes a job."""
    for case in manifest["cases"]:
        for method in case["methods"]:
            yield dict(case_id=case["id"], method=method, kind=case["kind"],
                       inputs=1 if case["kind"] == "memory" else INPUT_COUNT,
                       trials=MEMORY_TRIALS if case["kind"] == "memory" else None,
                       status="planned")


def build_manifest(*, profile="full"):
    """Build complete prespecified coverage without allocating any input array."""
    if profile not in ("full", "light"):
        raise ValueError("Choose the full or light campaign profile")
    definitions = [*_legacy_cases(), *_geometry_cases(profile), *_batch_cases(), *_memory_cases()]
    cases, identities = [], {}
    for case in definitions:
        identity = canonical_identity(case)
        if identity in identities:
            identities[identity]["sources"].extend(case["sources"])
            continue
        case["definition_sha256"] = identity
        case["methods"] = eligible_methods(case, profile=profile)
        case["metadata"]["primary_comparator"] = (
            "scipy:capped" if case["objective"] == "optional" else "scipy"
        )
        cases.append(case)
        identities[identity] = case
    manifest = dict(schema_version=SCHEMA_VERSION, generation=GENERATION,
                    input_count=INPUT_COUNT, rounds=ROUNDS, target_block_seconds=BLOCK_SECONDS,
                    warmup_calls=WARMUP_CALLS, memory_trials=MEMORY_TRIALS,
                    cases=cases, resource_policy="R1",
                    input_hash_policy="definition frozen now; materialized SHA256 recorded before measurement",
                    exclusions={"hard_gated_max_cardinality_timing": "validation only",
                                "application_and_lsape": "separate frozen campaigns"})
    if profile == "light":
        manifest.update(generation=LIGHT_GENERATION, profile=profile, rounds=LIGHT_ROUNDS,
                        light_coverage=dict(
                            geometry_settings=[dict(q=q, motion=motion) for q, motion in LIGHT_GEOMETRY_SETTINGS],
                            full_geometry_panels=[list(shape) for shape in LIGHT_GEOMETRY_FULL_PANELS],
                            geometry_baseline_row_limit=LIGHT_GEOMETRY_BASELINE_ROW_LIMIT,
                            capped_geometry_methods=["jvx", "jvs", "scipy:square"]))
    manifest["counts"] = manifest_counts(manifest)
    manifest["definition_sha256"] = definition_hash(manifest)
    return manifest


# Convenient public re-export; datasets imports no solver or catalog module.
from .datasets import generate_input  # noqa: E402
