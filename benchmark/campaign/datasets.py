"""On-demand deterministic float64 input generation, with no solver imports.

Returned costs are unmasked by geometric gates. Adapters must apply the case's
gate inside their complete-call boundary and validation uses the original costs.
The case metadata declares the input count; calling this allocates only one.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _key32(key):
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _supplemental_rng(group, index):
    # Full 256-bit group identity avoids collisions between supplemental streams.
    digest = hashlib.sha256(group["key"].encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]
    return np.random.default_rng([group["parameters"].get("supplemental_seed", group["seed"]),
                                  *words, index])


def _legacy_cost(rng, rows, cols, parameters):
    cost = rng.random((rows, cols))
    if parameters.get("signed"):
        cost = 2 * cost - 1
    if parameters.get("blocks"):
        r, c = 3 * rows // 4, cols // 4
        cost[:r, c:] += 3
        cost[r:, :c] += 3
    if parameters.get("ties"):
        cost = np.floor(cost * 8) / 8
    if parameters.get("forbidden"):
        mask = rng.random(cost.shape) < parameters["forbidden"]
        if parameters.get("planted_feasible"):
            np.fill_diagonal(mask, False)
        cost[mask] = np.inf
    return cost


def _extended_cost(group, index):
    p = group["parameters"]
    rng = np.random.default_rng([group["seed"], _key32(group["key"]), index])
    n, m = group["rows"], group["cols"]
    cost = rng.random((n, m))
    if p.get("signed"):
        cost = 2 * cost - 1
    if "levels" in p:
        cost = np.floor(cost * p["levels"]) / p["levels"]
    if p["distribution"] == "blocks":
        r, c = int(p["row_fraction"] * n), int(p["column_fraction"] * m)
        cost[:r, c:] += 3
        cost[r:, :c] += 3
    if "forbidden" in p:
        mask = rng.random((n, m)) < p["forbidden"]
        if p.get("planted_feasible"):
            np.fill_diagonal(mask, False)
        cost[mask] = np.inf
    return cost


def geometry_points(group, index):
    """Reproduce latent geometry for untimed diagnostics (not accuracy labels)."""
    rng = _supplemental_rng(group, index)
    r, c, p = group["rows"], group["cols"], group["parameters"]
    sources = rng.random((r, 2))
    survivors = rng.choice(r, size=round(p["q"] * r), replace=False)
    moved = sources[survivors] + rng.normal(0., p["motion"] * p["delta"], (len(survivors), 2))
    observations = np.concatenate((moved, rng.random((c - len(survivors), 2))))
    permutation = rng.permutation(c)
    inverse = np.empty(c, dtype=np.int64)
    inverse[permutation] = np.arange(c)
    pairs = np.column_stack((survivors, inverse[:len(survivors)])).astype(np.int64, copy=False)
    return sources, observations[permutation], pairs


def _geometry_cost(group, index):
    sources, observations, _ = geometry_points(group, index)
    # Bounded row blocks avoid a second full r*c*2 displacement tensor at 4096x5120.
    result = np.empty((len(sources), len(observations)), dtype=np.float64)
    for start in range(0, len(sources), 64):
        difference = sources[start:start + 64, None, :] - observations[None, :, :]
        result[start:start + 64] = np.sqrt(np.sum(difference * difference, axis=2))
    return result


def generate_input(case, input_index):
    """Generate one frozen input (or one batch), never an entire catalogue.

    Memory input_index denotes the fresh-process trial; ordinary cases use
    0 through input_count - 1. Base memory intentionally repeats its old fixed input
    across process trials. Extended memory preserves the trial-specific seeds.
    """
    metadata = case["metadata"]
    count = metadata["memory_trials"] if case["kind"] == "memory" else metadata["input_count"]
    if isinstance(input_index, bool) or not isinstance(input_index, (int, np.integer)) or not 0 <= input_index < count:
        raise ValueError(f"input_index must be an integer in [0, {count})")
    group, index = case["input_group"], int(input_index)
    p, generator = group["parameters"], group["generator"]
    rows, cols = group["rows"], group["cols"]
    if generator == "legacy_stream":
        if index < p["retained_inputs"]:
            bitgen = np.random.PCG64(group["seed"])
            bitgen.advance(p["stream_offset"] + index * p["draws_per_input"])
            rng = np.random.Generator(bitgen)
        else:
            rng = _supplemental_rng(group, index)
        result = _legacy_cost(rng, rows, cols, p)
    elif generator == "legacy_batch":
        if index == 0:
            bitgen = np.random.PCG64(group["seed"])
            bitgen.advance(p["stream_offset"])
            rng = np.random.Generator(bitgen)
        else:
            rng = _supplemental_rng(group, index)
        result = [_legacy_cost(rng, rows, cols, p) for _ in range(p["batch_size"])]
    elif generator == "extended":
        result = _extended_cost(group, index)
    elif generator == "geometry":
        result = _geometry_cost(group, index)
    elif generator == "memory_extended":
        rng = np.random.default_rng([group["seed"], 30000, rows, cols, index])
        result = rng.random((rows, cols))
    elif generator == "memory_base":
        result = np.random.default_rng(group["seed"]).random((rows, cols))
    elif generator == "uniform":
        rng = _supplemental_rng(group, index)
        result = ([rng.random((rows, cols)) for _ in range(p["batch_size"])]
                  if "batch_size" in p else rng.random((rows, cols)))
    else:
        raise ValueError(f"unknown input generator: {generator}")

    def orient(cost):
        return np.ascontiguousarray(cost.T if case["transpose"] else cost, dtype=np.float64)

    if isinstance(result, list):
        # Preserve the old public batch input contract: one C-contiguous 3-D
        # array, rather than changing wrapper conversion by passing a list.
        batch = np.empty((len(result), case["rows"], case["cols"]), dtype=np.float64)
        for position, cost in enumerate(result):
            batch[position] = cost.T if case["transpose"] else cost
        return batch
    return orient(result)


def input_sha256(value):
    """Hash a materialized matrix/batch, including shape, dtype and boundaries."""
    is_batch = isinstance(value, list) or value.ndim == 3
    arrays = value if is_batch else [value]
    digest = hashlib.sha256(b"batch" if is_batch else b"matrix")
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()
