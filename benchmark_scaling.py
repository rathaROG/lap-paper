"""Paired matrix-size/runtime sweeps using the recorded installed packages.

Archive new runs separately. Original and optional-comparison results are not
modified. All input generation and objective checks occur outside timing.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
from pathlib import Path
import random
import time

import compare_optional as optional
import run_experiments as base

PAPER = Path(__file__).resolve().parent
REQUIRED = base.SOLVERS
# JVHYBRID shares JVRUST's rectangular and rejection paths; it is timed for required inputs.
OPTIONAL = ("jvrust", "jvcpp", "jvx", "scipy_square", "scipy_dummies", "scipy_capped")


def specifications():
    """Fixed, predeclared size grid; no points selected from observed timings."""
    for regime in ("required", "optional"):
        for family in ("square", "wide", "fixed"):
            sizes = (16, 32, 64, 128, 256, 512, 1024) if family == "square" else (
                (16, 32, 64, 128, 256, 512) if family == "wide" else
                (64, 128, 256, 512, 1024, 2048, 4096))
            for size in sizes:
                n, m = (size, size) if family == "square" else (
                    (size, 4 * size) if family == "wide" else (64, size))
                yield {"case": f"{regime}_{family}_{n}x{m}", "regime": regime,
                       "family": family, "n": n, "m": m,
                       "limit": .5 if regime == "optional" else math.inf}
    for n in (16, 32, 64, 128, 256, 512, 1024):
        m = 5 * n // 4
        yield {"case": f"optional_blocks_{n}x{m}", "regime": "optional",
               "family": "blocks", "n": n, "m": m, "limit": .5, "blocks": True}


def solve(method, cost, limit):
    if method.startswith("scipy_"):
        return optional.solve(method, cost, limit)
    return base.solve(method, cost, limit)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_atomic(target, value):
    """Keep the previous checkpoint intact if a Windows file scan holds a handle."""
    for attempt in range(8):
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            if target.suffix == ".csv":
                if not value:
                    return
                with temporary.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(value[0]))
                    writer.writeheader()
                    writer.writerows(value)
            else:
                temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            temporary.replace(target)
            return
        except OSError:
            if attempt == 7:
                raise
            time.sleep(.1 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    runtime = json.loads((PAPER / "results" / "runtime.json").read_text())
    parser.add_argument("--output", type=Path, default=PAPER / "results" / "scaling")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--block-seconds", type=float, default=.03)
    options = parser.parse_args()
    if options.output.exists():
        raise SystemExit("Choose a new output directory to preserve previous measurements.")
    if min(options.instances, options.rounds, options.block_seconds) <= 0:
        raise SystemExit("Instances, rounds and block duration must be positive.")
    base.initialize()
    from threadpoolctl import threadpool_limits
    process = base.psutil.Process()
    original_affinity = process.cpu_affinity()
    input_rng = base.np.random.default_rng(options.seed + 201)
    order_rng = random.Random(options.seed + 202)
    samples, checks, datasets = [], [], []
    specs = list(specifications())
    started = time.perf_counter()
    try:
        process.cpu_affinity(original_affinity[:1])
        with threadpool_limits(limits=1):
            metadata = base.environment(options, original_affinity[:1], [])
            for name, native in metadata["native_modules"].items():
                assert native["sha256"] == runtime["native_modules"][name]["sha256"]
            metadata.update({"status": "running", "scaling_script_sha256": digest(Path(__file__)),
                "optional_script_sha256": digest(PAPER / "compare_optional.py"),
                "input_seed": options.seed + 201, "order_seed": options.seed + 202,
                "planned_cases": len(specs), "required_methods": REQUIRED, "optional_methods": OPTIONAL,
                "costs": "Independent U[0,1); no forbidden entries; block cases add 3 to competing blocks.",
                "scope": "Separate paired scaling run; fixed grid, all adapters and int64 output work included."})
            options.output.mkdir(parents=True)

            def checkpoint():
                save_atomic(options.output / "samples.csv", samples)
                save_atomic(options.output / "summary.csv", base.summary(samples, ("case", "solver")))
                for name, value in (("environment", metadata), ("datasets", datasets), ("checks", checks)):
                    save_atomic(options.output / (name + ".json"), value)

            checkpoint()
            for case_index, spec in enumerate(specs):
                limit = spec["limit"]
                inputs = [base.make_cost(spec, input_rng) for _ in range(options.instances)]
                methods = OPTIONAL if math.isfinite(limit) else REQUIRED
                reference_method = "scipy_square" if math.isfinite(limit) else "scipy"
                reference = [base.check_pairs(a, solve(reference_method, a, limit), limit) for a in inputs]
                calls, cycles = {}, {}
                for method in methods:
                    calls[method] = functools.partial(solve, method, limit=limit)
                    for index, cost in enumerate(inputs):
                        pairs = calls[method](cost)
                        value = base.check_pairs(cost, pairs, limit)
                        assert base.np.isclose(value, reference[index], rtol=1e-10, atol=1e-10), (
                            spec["case"], method, index, value, reference[index])
                        checks.append({"case": spec["case"], "solver": method, "instance": index,
                                       "pairs": len(pairs), "objective": value,
                                       "reference_objective": reference[index]})
                    probe = base.timed_block(calls[method], inputs, 1)
                    cycles[method] = max(1, min(100000, round(options.block_seconds * 1e9 / (probe * len(inputs)))))
                for round_index in range(options.rounds):
                    order = list(methods)
                    order_rng.shuffle(order)
                    for position, method in enumerate(order):
                        elapsed = base.timed_block(calls[method], inputs, cycles[method])
                        samples.append({"case": spec["case"], "solver": method, "round": round_index,
                                        "position": position, "instances": len(inputs),
                                        "cycles": cycles[method], "nanoseconds": elapsed})
                counts = [row["pairs"] for row in checks if row["case"] == spec["case"] and row["solver"] == "jvrust"]
                # Null denotes infinity in the strict-JSON dataset manifest.
                datasets.append({**spec, "limit": limit if math.isfinite(limit) else None,
                    "instances": len(inputs), "mean_pairs": sum(counts) / len(counts),
                    "mean_unmatched": min(spec["n"], spec["m"]) - sum(counts) / len(counts),
                    "input_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in inputs]})
                checkpoint()
                print(f"{case_index + 1}/{len(specs)} {spec['case']} "
                      f"unmatched={datasets[-1]['mean_unmatched']:.2f} "
                      f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
            metadata.update({"status": "completed", "elapsed_seconds": time.perf_counter() - started,
                             "completed_cases": len(datasets), "objective_checks": len(checks),
                             "timing_samples": len(samples)})
            checkpoint()
    finally:
        process.cpu_affinity(original_affinity)
    print(json.dumps({key: metadata[key] for key in
                      ("status", "completed_cases", "objective_checks", "timing_samples", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
