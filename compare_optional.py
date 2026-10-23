"""Compare implicit rejection with three equivalent SciPy formulations.

JVRUST and JVCPP (the Rust and C++ implementations) run against square-expanded,
dummy-column and capped-cost SciPy. JVHYBRID shares JVRUST's rectangular and
rejection paths, so it is not repeated here. Run after record_runtime.py.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
from pathlib import Path
import random

import run_experiments as base

PAPER = Path(__file__).resolve().parent
METHODS = ("jvrust", "jvcpp", "scipy_square", "scipy_dummies", "scipy_capped")


def solve(method, cost, limit):
    np = base.np
    if method in base.NATIVE:
        return base.solve(method, cost, limit)
    if method == "scipy_square":
        return base.solve("scipy", cost, limit)
    if method == "scipy_capped":
        # Capping precedes optimization; filtering an uncapped solve is wrong.
        rows, cols = base.lsa(np.minimum(cost, limit))
        keep = cost[rows, cols] < limit  # Reject equality ties consistently.
        return np.column_stack((rows[keep], cols[keep])).astype(np.int64, copy=False)
    if method == "scipy_dummies":
        transpose = cost.shape[0] > cost.shape[1]
        oriented = cost.T if transpose else cost
        r, c = oriented.shape
        work = np.empty((r, c + r), dtype=np.float64)
        work[:, :c] = oriented
        work[:, c:] = limit
        rows, cols = base.lsa(work)
        keep = cols < c
        rows, cols = rows[keep], cols[keep]
        return np.column_stack((cols, rows) if transpose else (rows, cols)).astype(np.int64, copy=False)
    raise ValueError(method)


def validate(seed):
    np = base.np
    rng = np.random.default_rng(seed)
    calls = instances = 0
    maximum_error = 0.0
    for n in range(6):
        for m in range(7):
            for _ in range(10):
                cost = rng.integers(-16, 17, size=(n, m)).astype(np.float64) / 8
                cost[rng.random((n, m)) < .2] = np.inf
                for limit in (-1.0, 0.0, .125, 1.0, 2.0):
                    expected = base.brute_objective(cost, limit)
                    instances += 1
                    for method in METHODS:
                        pairs = solve(method, cost, limit)
                        actual = base.check_pairs(cost, pairs, limit)
                        maximum_error = max(maximum_error, abs(actual - expected))
                        assert actual == expected, (method, n, m, limit, actual, expected)
                        calls += 1
    # Explicit boundary cases, including all-forbidden and all-equality inputs.
    targeted = [
        (np.full((3, 5), np.inf), .5),
        (np.full((5, 3), .5), .5),
        (np.array([[0., .9], [.9, np.inf]]), 1.),
        (np.array([[-2., -1.], [-1., np.inf]]), -1.),
    ]
    for cost, limit in targeted:
        expected = base.brute_objective(cost, limit)
        for method in METHODS:
            actual = base.check_pairs(cost, solve(method, cost, limit), limit)
            assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    return {"status": "passed", "seed": seed, "grid_instances": instances,
            "grid_solver_calls": calls, "targeted_solver_calls": len(targeted) * len(METHODS),
            "maximum_absolute_grid_objective_error": maximum_error,
            "scope": "float64, binary-rational grid; ties compare objectives, not pairs"}


def datasets(seed, instances):
    # Consume every original case in the original order to reproduce its inputs.
    rng = base.np.random.default_rng(seed + 1)
    for spec in base.case_specs():
        inputs = [base.make_cost(spec, rng) for _ in range(instances)]
        if math.isfinite(spec["limit"]):
            yield dict(spec, source="original"), inputs
    rng = base.np.random.default_rng(seed + 101)
    extra = [
        {"case": "reject128x160_L005", "n": 128, "m": 160, "limit": .005, "forbidden": .5},
        {"case": "reject128x160_L001", "n": 128, "m": 160, "limit": .001, "forbidden": .5},
        {"case": "reject_tall160x128", "n": 160, "m": 128, "limit": .005, "forbidden": .5},
        {"case": "optional_wide64x512", "n": 64, "m": 512, "limit": .005, "forbidden": .5},
        {"case": "optional_blocks256x320", "n": 256, "m": 320, "limit": .5, "blocks": True},
    ]
    for spec in extra:
        yield dict(spec, source="additional"), [base.make_cost(spec, rng) for _ in range(instances)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = json.loads((PAPER / "results" / "runtime.json").read_text())
    parser.add_argument("--output", type=Path, default=PAPER / "results" / "optional_followup")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--block-seconds", type=float, default=.03)
    options = parser.parse_args()
    if options.output.exists():
        raise SystemExit("Choose a new output directory to preserve previous measurements.")
    base.initialize()
    from threadpoolctl import threadpool_limits
    process = base.psutil.Process()
    original_affinity = process.cpu_affinity()
    samples, checks, descriptions = [], [], []
    order_rng = random.Random(options.seed + 102)
    try:
        process.cpu_affinity(original_affinity[:1])
        with threadpool_limits(limits=1):
            metadata = base.environment(options, original_affinity[:1], [])
            for name, binary in metadata["native_modules"].items():
                assert binary["sha256"] == runtime["native_modules"][name]["sha256"]
            validation = validate(options.seed + 100)
            print("Independent optional validation passed:", validation["grid_solver_calls"], flush=True)
            for spec, inputs in datasets(options.seed, options.instances):
                limit = spec["limit"]
                reference = [base.objective(a, solve("scipy_square", a, limit), limit) for a in inputs]
                calls, cycles = {}, {}
                for method in METHODS:
                    calls[method] = functools.partial(solve, method, limit=limit)
                    for index, cost in enumerate(inputs):
                        pairs = calls[method](cost)
                        actual = base.check_pairs(cost, pairs, limit)
                        assert base.np.isclose(actual, reference[index], rtol=1e-10, atol=1e-10)
                        checks.append({"case": spec["case"], "solver": method, "instance": index,
                                       "pairs": len(pairs), "objective": actual,
                                       "reference_objective": reference[index]})
                    probe = base.timed_block(calls[method], inputs, 1)
                    cycles[method] = max(1, min(100000, round(options.block_seconds * 1e9 / (probe * len(inputs)))))
                for round_index in range(options.rounds):
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    for position, method in enumerate(order):
                        elapsed = base.timed_block(calls[method], inputs, cycles[method])
                        samples.append({"case": spec["case"], "solver": method, "round": round_index,
                                        "position": position, "instances": len(inputs),
                                        "cycles": cycles[method], "nanoseconds": elapsed})
                counts = [row["pairs"] for row in checks if row["case"] == spec["case"] and row["solver"] == "jvrust"]
                descriptions.append({**spec, "instances": len(inputs), "mean_pairs": sum(counts) / len(counts),
                    "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in inputs]})
                print(spec["case"], "mean pairs", descriptions[-1]["mean_pairs"], flush=True)
            metadata.update({"supplement_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "methods": METHODS, "scope": "Separate paired rerun; original results preserved"})
    finally:
        process.cpu_affinity(original_affinity)
    options.output.mkdir(parents=True)
    summaries = base.summary(samples, ("case", "solver"))
    base.csv_write(options.output / "samples.csv", samples)
    base.csv_write(options.output / "summary.csv", summaries)
    for name, value in (("environment", metadata), ("validation", validation),
                        ("checks", checks), ("datasets", descriptions)):
        (options.output / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
    for spec in descriptions:
        values = {row["solver"]: row["median_ns"] / 1000 for row in summaries if row["case"] == spec["case"]}
        print(spec["case"], {key: round(value, 2) for key, value in values.items()})


if __name__ == "__main__":
    main()
