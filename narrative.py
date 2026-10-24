"""Numbers quoted in the manuscript text, derived from the archived results.

values() computes every narrative number and asserts each qualitative claim the
text makes about the data. expected_sentences() returns the exact manuscript
fragments that check_artifact.py requires. Run directly to print the values.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

PAPER = Path(__file__).resolve().parent
RESULTS = PAPER / "results"
WORDS = {5: "five", 6: "six", 7: "seven", 9: "nine", 10: "ten"}
REQUIRED = ("square128", "square512", "wide64x512", "tall512x64", "near256x320",
            "blocks256x320", "ties128x160")
FINITE = ("optional128x160", "optional256x320", "tight128x160", "negative128x160",
          "optional_square128")
OURS = ("jvrust", "jvhybrid", "jvcpp")

# Manuscript fragments; {name} fields are filled from values().
TEMPLATES = (
    ("6.1", "The installed packages pass their own test suites: {tests}"),
    ("6.2", "An implementation of the proposed solver is fastest among the configured methods in {required_wins} of the seven required cases and in all five finite-limit cases."),
    ("6.2", "On the uniform 256 × 320 problem, JVRUST takes {near_jvrust} microseconds versus {near_scipy} for SciPy."),
    ("6.2", "On congested blocks, JVRUST requires {blocks_jvrust_ms} ms while SciPy requires {blocks_scipy_ms} ms: SciPy is about {blocks_ratio} times faster"),
    ("6.2", "JVRUST takes {ties_jvrust} microseconds and JVS {ties_jvs}."),
    ("6.2", "Finite-limit JVRUST runtimes range from {finite_min} to {finite_max} microseconds. The 128 × 160, L=0.5 case is {finite_ratio} times faster than the square-augmented SciPy configuration."),
    ("6.2", "JVCPP is {cpp_required_low}–{cpp_required_high}% slower than JVRUST on the required cases other than the tall input, {cpp_finite_low}–{cpp_finite_high} times slower on the finite-limit cases, and {cpp_tall_ratio} times slower on the tall 512 × 64 case."),
    ("6.3", "JVRUST's four-worker call is {batch_ratio_1} times faster than its one-worker call on the 64 × 80 batch, {batch_ratio_2} times faster on the finite-limit batch, and {batch_ratio_3} times faster on the 1024-problem batch."),
    ("6.3", "the batch call is {tiny_serial_ratio} times faster than the loop even with one worker"),
    ("6.3", "four workers make it {tiny_parallel_ratio} times faster than the loop."),
    ("6.3", "where its four-worker call takes {cpp_tiny_t4} ms against {jvrust_tiny_t4} ms."),
    ("6.4", "The three implementations add {memory_low}–{memory_high} MiB in both conditions."),
    ("6.5", "JVRUST is {capped_ratio} times faster than capped SciPy in this run, compared with {square_ratio} times faster than square-expanded SciPy."),
    ("6.5", "JVRUST is faster than capped SciPy in nine of ten finite-limit cases, by factors of {capped_low}–{capped_high} on those cases"),
    ("6.5", "capped SciPy takes {block_capped} ms and JVRUST takes {block_jvrust} ms, making capped SciPy {block_ratio} times faster."),
    ("6.6", "All {scaling_checks} timed-instance feasibility and objective checks pass, and {scaling_samples} timing-round samples are archived across {scaling_groups} method/setting combinations."),
    ("6.6", "At 1024 × 1024 without a finite limit, JVRUST takes {sq_jvrust} ms, JVHYBRID takes {sq_jvhybrid} ms, JVCPP takes {sq_jvcpp} ms, and SciPy takes {sq_scipy} ms."),
    ("6.6", "At 64 × 4096, JVRUST takes {fx_jvrust} ms and SciPy takes {fx_scipy} ms, while JVX takes {fx_jvx} ms."),
    ("6.6", "For finite-limit 512 × 2048 inputs at L=0.5, JVRUST takes {ow_jvrust} ms, capped SciPy takes {ow_capped} ms, square-expanded SciPy takes {ow_square} ms, and JVX takes {ow_jvx} ms."),
    ("6.6", "In these uniform sweeps, which contain no forbidden entries, JVCPP takes up to {cpp_sweep_max} times as long as JVRUST; at the largest size of each family, their median runtimes differ by at most {cpp_sweep_large}% relative to JVRUST, and neither implementation is faster at every point."),
    ("6.6", "JVRUST is faster than capped SciPy at N={block_win_sizes}; capped SciPy is faster at N={block_lose_sizes}."),
    ("6.6", "At 1024 × 1280, JVRUST takes {bl_jvrust} ms and capped SciPy takes {bl_capped} ms, a {bl_ratio}-fold advantage for capped SciPy."),
    ("Abstract", "a 128 × 160 finite-limit problem takes {abs_jvrust} microseconds with JVRUST, {abs_capped} with SciPy on capped costs of the original dimensions, and {abs_square} with square-expanded SciPy."),
    ("Abstract", "capped SciPy is {block_ratio} times faster on a congested-block case"),
    ("Abstract", "by {tiny_parallel_ratio} times against a Python loop for 1024 small problems with four workers."),
)


def load(results=None):
    results = Path(results or RESULTS)

    def times(name):
        with (results / name).open(newline="") as stream:
            return {(row["case"], row["solver"]): float(row["median_ns"]) for row in csv.DictReader(stream)}
    return (times("single_summary.csv"), times("batch_summary.csv"), times("formulations/summary.csv"),
            times("scaling/summary.csv"), json.loads((results / "formulations/datasets.json").read_text()),
            json.loads((results / "scaling/datasets.json").read_text()),
            {name: sum(suite["reported_pass_counts"]) for name, suite in
             json.loads((results / "test_validation.json").read_text())["suites"].items()})


ARCHIVE = PAPER / "collections" / "python-3.11.17-20261024" / "results"

# Appendix A: the completed Python 3.11.17 replication.
APPENDIX_TEMPLATES = (
    ("A", "All of its correctness checks passed: {validation_calls:,} enumeration calls with {infeasible} correctly reported infeasible problems, {formulation_calls:,} calls in the formulation validation, {scaling_checks:,} scaling checks, and the same package test suites as in Section 6.1."),
    ("A", "The four-worker JVCPP and JVRUST medians on the 64 × 80 batch are approximately tied in Python 3.11, unlike the larger gap in Python 3.14."),
    ("A", "In both runs, JVRUST is faster than capped SciPy at N={both_win_sizes}, while capped SciPy is faster at N={both_lose_sizes}."),
)


def block_crossover(results=None):
    """Congested sweep sizes N where JVRUST beats capped SciPy, and where it does not."""
    _, _, _, scaling, _, scaling_datasets, _ = load(results)
    blocks = sorted((d for d in scaling_datasets if d["family"] == "blocks"), key=lambda d: d["n"])
    wins = [d["n"] for d in blocks if scaling[d["case"], "jvrust"] < scaling[d["case"], "scipy_capped"]]
    return wins, [d["n"] for d in blocks if d["n"] not in wins]

# Table A4 rows: label and how to compute the value from one run.
PRINCIPAL_RATIOS = (
    "t(JVRUST) / t(SciPy), required blocks (×)",
    "t(capped SciPy) / t(JVRUST), smallest of nine wins (×)",
    "t(capped SciPy) / t(JVRUST), largest of nine wins (×)",
    "t(JVRUST) / t(capped SciPy), blocks 256 × 320 (×)",
    "t(JVRUST) / t(capped SciPy), blocks 1024 × 1280 (×)",
    "t(loop) / t(batch), 1024 × (8 × 10), one worker (×)",
    "t(loop) / t(batch), 1024 × (8 × 10), four workers (×)",
    "t(JVCPP) / t(JVRUST), largest finite-limit gap in Tables 3/A1 (×)",
    "t(JVCPP) / t(JVRUST), tall 512 × 64 (×)",
    "JVRUST, 128 × 160 at L=0.5 (µs)",
)


def principal_ratios(results=None):
    """The quantities compared across runs in Table A4, without asserting claims."""
    single, batch, formulations, scaling, datasets, scaling_datasets, _ = load(results)
    capped = sorted(formulations[d["case"], "scipy_capped"] / formulations[d["case"], "jvrust"]
                    for d in datasets
                    if formulations[d["case"], "jvrust"] < formulations[d["case"], "scipy_capped"])
    loop = batch["batch1024x8x10", "jvrust_loop"]
    return [
        single["blocks256x320", "jvrust"] / single["blocks256x320", "scipy"],
        capped[0], capped[-1],
        formulations["optional_blocks256x320", "jvrust"] / formulations["optional_blocks256x320", "scipy_capped"],
        scaling["optional_blocks_1024x1280", "jvrust"] / scaling["optional_blocks_1024x1280", "scipy_capped"],
        loop / batch["batch1024x8x10", "jvrust_batch_t1"],
        loop / batch["batch1024x8x10", "jvrust_batch_t4"],
        max(single[case, "jvcpp"] / single[case, "jvrust"] for case in FINITE),
        single["tall512x64", "jvcpp"] / single["tall512x64", "jvrust"],
        formulations["optional128x160", "jvrust"] / 1000,
    ]


def appendix_values(archive=None):
    archive = Path(archive or ARCHIVE)
    validation = json.loads((archive / "validation.json").read_text())
    formulation = json.loads((archive / "formulations/validation.json").read_text())
    assert not validation["failures"] and validation["maximum_absolute_objective_error"] == 0
    assert formulation["status"] == "passed" and formulation["maximum_absolute_grid_objective_error"] == 0
    checks = json.loads((archive / "scaling/checks.json").read_text())
    assert all(abs(row["objective"] - row["reference_objective"]) <= 1e-10 * max(1, abs(row["reference_objective"]))
               for row in checks)
    archived_tests = load(archive)[6]
    assert archived_tests == load()[6], "the archived run passed the same suites"
    _, batch, *_ = load(archive)
    gap = batch["batch64x64x80", "jvcpp_batch_t4"] / batch["batch64x64x80", "jvrust_batch_t4"] - 1
    assert abs(gap) < .05, "the replicated four-worker medians are approximately tied"
    crossover = block_crossover(archive)
    assert crossover == block_crossover(), "the crossover is at the same size in both runs"
    return {"both_win_sizes": join_sizes(crossover[0]), "both_lose_sizes": join_sizes(crossover[1]),
            "validation_calls": validation["solver_calls"], "infeasible": validation["infeasible_calls"],
            "formulation_calls": formulation["grid_solver_calls"] + formulation["targeted_solver_calls"],
            "scaling_checks": len(checks), "cpp_t4_gap": f"{100 * gap:.0f}"}


def appendix_sentences(archive=None):
    filled = appendix_values(archive)
    return [(heading, template.format(**filled)) for heading, template in APPENDIX_TEMPLATES]


def join_sizes(sizes):
    sizes = [str(size) for size in sizes]
    return ", ".join(sizes[:-1]) + ", and " + sizes[-1] if len(sizes) > 2 else " and ".join(sizes)


def values(single, batch, formulations, scaling, datasets, scaling_datasets, tests):
    us = {key: value / 1000 for key, value in single.items()}
    ms_batch = {key: value / 1e6 for key, value in batch.items()}
    fu = {key: value / 1000 for key, value in formulations.items()}
    ms_scale = {key: value / 1e6 for key, value in scaling.items()}
    all_solvers = {solver for _, solver in single}

    # Claims in Section 6.2.
    fastest = {case: min(all_solvers, key=lambda solver: us[case, solver]) for case in REQUIRED + FINITE}
    required_wins = sum(fastest[case] in OURS for case in REQUIRED)
    assert all(fastest[case] in OURS for case in FINITE)
    assert fastest["blocks256x320"] == "scipy"
    assert all(us["blocks256x320", "jvc"] < us["blocks256x320", solver] for solver in OURS)
    for solver in ("jvrust", "scipy"):
        assert us["tall512x64", solver] > us["wide64x512", solver]
    for case in ("square128", "square512"):
        assert abs(us[case, "jvhybrid"] / us[case, "jvrust"] - 1) < .05
    others = [case for case in REQUIRED if case != "tall512x64"]
    cpp_required = [100 * (us[case, "jvcpp"] / us[case, "jvrust"] - 1) for case in others]
    cpp_finite = [us[case, "jvcpp"] / us[case, "jvrust"] for case in FINITE]
    assert min(cpp_required) > 0 and min(cpp_finite) > 1

    # Claims in Section 6.3.
    loop = ms_batch["batch1024x8x10", "jvrust_loop"]
    for case in ("batch64x64x80", "batch16x128x160_optional"):
        assert abs(ms_batch[case, "jvrust_batch_t1"] / ms_batch[case, "jvrust_loop"] - 1) < .1
    for case in ("batch64x64x80", "batch16x128x160_optional", "batch1024x8x10"):
        assert ms_batch[case, "jvcpp_batch_t1"] > ms_batch[case, "jvrust_batch_t1"]
    for case in ("batch16x128x160_optional", "batch1024x8x10"):
        assert ms_batch[case, "jvcpp_batch_t4"] > ms_batch[case, "jvrust_batch_t4"]
    assert ms_batch["batch64x64x80", "jvcpp_batch_t4"] > ms_batch["batch64x64x80", "jvrust_batch_t4"]
    assert all(ms_batch[case, "jvx_batch_t4"] > ms_batch[case, "jvx_batch_t1"]
               for case in ("batch64x64x80", "batch1024x8x10"))
    assert ms_batch["batch16x128x160_optional", "jvx_batch_t4"] < ms_batch["batch16x128x160_optional", "jvx_batch_t1"]

    # Claims in Section 6.5.
    capped_wins = [d["case"] for d in datasets if fu[d["case"], "jvrust"] < fu[d["case"], "scipy_capped"]]
    assert len(capped_wins) == 9 and "optional_blocks256x320" not in capped_wins
    assert [d["case"] for d in datasets if fu[d["case"], "jvcpp"] < fu[d["case"], "scipy_capped"]] == capped_wins
    capped_factors = [fu[case, "scipy_capped"] / fu[case, "jvrust"] for case in capped_wins]
    reference = "optional128x160"
    assert fu[reference, "scipy_capped"] < fu[reference, "scipy_dummies"] < fu[reference, "scipy_square"]
    # Section 7.4: outside the congested blocks, a proposed implementation has the lowest median everywhere.
    for table in (us, fu, ms_scale):
        for case in {case for case, _ in table if "blocks" not in case}:
            assert min((value, solver) for (key, solver), value in table.items() if key == case)[1] in OURS, case
    # Section 7.4: capping is the fastest SciPy formulation in every formulation and scaling case.
    for table in (fu, ms_scale):
        for case in {case for case, solver in table if solver == "scipy_capped"}:
            assert table[case, "scipy_capped"] < min(table[case, "scipy_dummies"], table[case, "scipy_square"])
    unmatched = {d["case"]: min(d["n"], d["m"]) - d["mean_pairs"] for d in datasets}
    assert unmatched["reject128x160_L005"] == 87.75 and unmatched["reject128x160_L001"] == 117

    # Claims in Section 6.4.
    memory = {(row["case"], row["solver"]): row["additional_peak"] / 2**20
              for row in json.loads((RESULTS / "memory.json").read_text())}
    ours_memory = [memory[case, solver] for case in ("required", "optional") for solver in OURS]
    assert max(ours_memory) < 1 and memory["required", "scipy"] < 1 and memory["optional", "scipy"] > 100

    # Claims in Section 6.6.
    uniform = [d for d in scaling_datasets if d["family"] != "blocks"]
    sweep = {d["case"]: ms_scale[d["case"], "jvcpp"] / ms_scale[d["case"], "jvrust"] for d in uniform}
    largest_cases = [max((d for d in uniform if (d["regime"], d["family"]) == key), key=lambda d: d["n"] * d["m"])["case"]
                     for key in {(d["regime"], d["family"]) for d in uniform}]
    large_gap = max(abs(sweep[case] - 1) for case in largest_cases)
    assert max(sweep.values()) == max(sweep[d["case"]] for d in uniform if d["regime"] == "optional")
    assert max(sweep.values()) > 1.5
    assert min(sweep.values()) < 1, "neither implementation is faster at every point"
    blocks = sorted((d for d in scaling_datasets if d["family"] == "blocks"), key=lambda d: d["n"])
    wins = [d["n"] for d in blocks if ms_scale[d["case"], "jvrust"] < ms_scale[d["case"], "scipy_capped"]]
    loses = [d["n"] for d in blocks if d["n"] not in wins]
    assert wins == sorted(wins) and max(wins) < min(loses), "a single crossover"
    assert all(d["mean_unmatched"] / d["n"] == .4375 for d in blocks)
    assert all(d["mean_unmatched"] == 0 for d in scaling_datasets
               if d["regime"] == "optional" and d["family"] != "blocks")
    largest = "optional_blocks_1024x1280"

    names = {"lapx_python": "lapx", "lapcpp_python": "lapcpp", "laprust_python": "laprust"}
    native = "" if "lapcpp_native" not in tests else ", and lapcpp's native C++ test program"
    test_text = ("{lapx:,} LAPX, {lapcpp:,} lapcpp, and {laprust:,} laprust Python tests, "
                 "{rr} Rust unit tests for laprust and {xr} for LAPX's solver core with all features enabled{native}.").format(
        **{names[key]: value for key, value in tests.items() if key in names},
        rr=tests["laprust_rust"], xr=tests["lapx_rust"], native=native)
    scaling_groups = len(scaling)
    return {
        "tests": test_text,
        "required_wins": WORDS[required_wins],
        "near_jvrust": f"{us['near256x320', 'jvrust']:.2f}", "near_scipy": f"{us['near256x320', 'scipy']:.2f}",
        "blocks_jvrust_ms": f"{us['blocks256x320', 'jvrust'] / 1000:.2f}",
        "blocks_scipy_ms": f"{us['blocks256x320', 'scipy'] / 1000:.2f}",
        "blocks_ratio": f"{us['blocks256x320', 'jvrust'] / us['blocks256x320', 'scipy']:.1f}",
        "ties_jvrust": f"{us['ties128x160', 'jvrust']:.2f}", "ties_jvs": f"{us['ties128x160', 'jvs']:.2f}",
        "finite_min": f"{min(us[case, 'jvrust'] for case in FINITE):.2f}",
        "finite_max": f"{max(us[case, 'jvrust'] for case in FINITE):.2f}",
        "finite_ratio": f"{us[reference, 'scipy'] / us[reference, 'jvrust']:.1f}",
        "cpp_required_low": f"{min(cpp_required):.0f}", "cpp_required_high": f"{max(cpp_required):.0f}",
        "cpp_finite_low": f"{min(cpp_finite):.1f}", "cpp_finite_high": f"{max(cpp_finite):.1f}",
        "cpp_tall_ratio": f"{us['tall512x64', 'jvcpp'] / us['tall512x64', 'jvrust']:.1f}",
        "batch_ratio_1": f"{ms_batch['batch64x64x80', 'jvrust_batch_t1'] / ms_batch['batch64x64x80', 'jvrust_batch_t4']:.2f}",
        "batch_ratio_2": f"{ms_batch['batch16x128x160_optional', 'jvrust_batch_t1'] / ms_batch['batch16x128x160_optional', 'jvrust_batch_t4']:.2f}",
        "batch_ratio_3": f"{ms_batch['batch1024x8x10', 'jvrust_batch_t1'] / ms_batch['batch1024x8x10', 'jvrust_batch_t4']:.2f}",
        "tiny_serial_ratio": f"{loop / ms_batch['batch1024x8x10', 'jvrust_batch_t1']:.1f}",
        "tiny_parallel_ratio": f"{loop / ms_batch['batch1024x8x10', 'jvrust_batch_t4']:.1f}",
        "cpp_tiny_t4": f"{ms_batch['batch1024x8x10', 'jvcpp_batch_t4']:.3f}",
        "jvrust_tiny_t4": f"{ms_batch['batch1024x8x10', 'jvrust_batch_t4']:.3f}",
        "capped_ratio": f"{fu[reference, 'scipy_capped'] / fu[reference, 'jvrust']:.1f}",
        "square_ratio": f"{fu[reference, 'scipy_square'] / fu[reference, 'jvrust']:.1f}",
        "capped_low": f"{min(capped_factors):.1f}", "capped_high": f"{max(capped_factors):.1f}",
        "block_capped": f"{fu['optional_blocks256x320', 'scipy_capped'] / 1000:.3f}",
        "block_jvrust": f"{fu['optional_blocks256x320', 'jvrust'] / 1000:.3f}",
        "block_ratio": f"{fu['optional_blocks256x320', 'jvrust'] / fu['optional_blocks256x320', 'scipy_capped']:.1f}",
        "memory_low": f"{min(ours_memory):.2f}", "memory_high": f"{max(ours_memory):.2f}",
        "cpp_sweep_max": f"{max(sweep.values()):.1f}", "cpp_sweep_large": f"{100 * large_gap:.0f}",
        "scaling_checks": f"{len(json.loads((RESULTS / 'scaling/checks.json').read_text())):,}",
        "scaling_samples": f"{scaling_groups * 9:,}", "scaling_groups": str(scaling_groups),
        "sq_jvrust": f"{ms_scale['required_square_1024x1024', 'jvrust']:.3f}",
        "sq_jvhybrid": f"{ms_scale['required_square_1024x1024', 'jvhybrid']:.3f}",
        "sq_jvcpp": f"{ms_scale['required_square_1024x1024', 'jvcpp']:.3f}",
        "sq_scipy": f"{ms_scale['required_square_1024x1024', 'scipy']:.3f}",
        "fx_jvrust": f"{ms_scale['required_fixed_64x4096', 'jvrust']:.3f}",
        "fx_scipy": f"{ms_scale['required_fixed_64x4096', 'scipy']:.3f}",
        "fx_jvx": f"{ms_scale['required_fixed_64x4096', 'jvx']:.3f}",
        "ow_jvrust": f"{ms_scale['optional_wide_512x2048', 'jvrust']:.3f}",
        "ow_capped": f"{ms_scale['optional_wide_512x2048', 'scipy_capped']:.3f}",
        "ow_square": f"{ms_scale['optional_wide_512x2048', 'scipy_square']:.3f}",
        "ow_jvx": f"{ms_scale['optional_wide_512x2048', 'jvx']:.3f}",
        "block_win_sizes": join_sizes(wins), "block_lose_sizes": join_sizes(loses),
        "bl_jvrust": f"{ms_scale[largest, 'jvrust']:.3f}", "bl_capped": f"{ms_scale[largest, 'scipy_capped']:.3f}",
        "bl_ratio": f"{ms_scale[largest, 'jvrust'] / ms_scale[largest, 'scipy_capped']:.1f}",
        "abs_jvrust": f"{fu[reference, 'jvrust']:.1f}", "abs_capped": f"{fu[reference, 'scipy_capped']:.1f}",
        "abs_square": f"{fu[reference, 'scipy_square']:.1f}",
    }


def expected_sentences(*data):
    filled = values(*data)
    return [(heading, template.format(**filled)) for heading, template in TEMPLATES]


if __name__ == "__main__":
    for key, value in values(*load()).items():
        print(f"{key:20s} {value}")
