"""Physical-core discovery and effective thread-mask verification outside timing."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import struct


def _parse_windows_cores(raw):
    """Decode the documented 64-bit PROCESSOR_RELATIONSHIP records."""
    offset, cores = 0, []
    seen = set()
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise RuntimeError("Truncated Windows topology header")
        relation, length = struct.unpack_from("<II", raw, offset)
        if length < 8 or offset + length > len(raw):
            raise RuntimeError("Invalid Windows topology record length")
        if relation == 0:
            if length < 48:
                raise RuntimeError("Truncated Windows processor-core record")
            flags, efficiency = struct.unpack_from("BB", raw, offset + 8)
            groups = struct.unpack_from("<H", raw, offset + 30)[0]
            if groups != 1 or 32 + 16 * groups > length:
                raise RuntimeError("A processor core must have one complete group mask")
            mask, group = struct.unpack_from("<QH", raw, offset + 32)
            if group != 0:
                raise RuntimeError("Multiple processor groups require an explicit supported configuration")
            cpus = [i for i in range(64) if mask & (1 << i)]
            if not cpus or seen.intersection(cpus):
                raise RuntimeError("Empty or overlapping physical-core processor masks")
            seen.update(cpus)
            cores.append({"core_id": len(cores), "logical_cpus": cpus,
                          "efficiency_class": efficiency, "smt": bool(flags & 1), "group": 0,
                          "class_source": "GetLogicalProcessorInformationEx.Processor.EfficiencyClass"})
        offset += length
    if not cores:
        raise RuntimeError("No physical-core records returned by Windows")
    return cores


def windows_cores():
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("The campaign requires a 64-bit Windows interpreter")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = kernel.GetLogicalProcessorInformationEx
    fn.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    fn.restype = ctypes.c_int
    size = ctypes.c_ulong()
    fn(0, None, ctypes.byref(size))
    if not size.value:
        raise OSError(ctypes.get_last_error(), "Cannot query processor topology")
    buffer = ctypes.create_string_buffer(size.value)
    if not fn(0, buffer, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error(), "Cannot read processor topology")
    return _parse_windows_cores(buffer.raw[:size.value])


def _cpu_list(text):
    """Parse a kernel CPU list such as '0-15,18'."""
    cpus = set()
    for part in text.strip().split(","):
        if part:
            start, _, end = part.partition("-")
            cpus.update(range(int(start), int(end or start) + 1))
    return cpus


def _linux_hybrid_classes(sysfs):
    """Intel hybrid PMU lists: E cores (cpu_atom) 0, P cores (cpu_core) 1."""
    classes = {}
    for name, value in (("cpu_atom", 0), ("cpu_core", 1)):
        path = sysfs / "devices" / name / "cpus"
        if path.exists():
            classes.update(dict.fromkeys(_cpu_list(path.read_text()), value))
    return classes


def linux_cores(sysfs=Path("/sys")):
    hybrid = _linux_hybrid_classes(sysfs)
    grouped = {}
    for directory in (sysfs / "devices" / "system" / "cpu").glob("cpu[0-9]*"):
        cpu = int(directory.name[3:])
        top = directory / "topology"
        try:
            package = int((top / "physical_package_id").read_text())
            core = int((top / "core_id").read_text())
        except OSError:
            continue
        # Higher values are faster cores, matching Windows EfficiencyClass.
        core_type, source = None, None
        if (top / "core_type").exists():
            core_type, source = int((top / "core_type").read_text()), "topology/core_type"
        elif cpu in hybrid:
            core_type, source = hybrid[cpu], "devices/cpu_core and devices/cpu_atom"
        elif (directory / "cpu_capacity").exists():
            core_type, source = int((directory / "cpu_capacity").read_text()), "cpu_capacity"
        entry = grouped.setdefault((package, core), {"core_id": f"{package}:{core}",
                 "logical_cpus": [], "efficiency_class": core_type, "package": package,
                 "class_source": source})
        entry["logical_cpus"].append(cpu)
    if not grouped:
        raise RuntimeError("Physical-core topology unavailable")
    for entry in grouped.values():
        entry["logical_cpus"].sort()
        entry["smt"] = len(entry["logical_cpus"]) > 1
    return list(grouped.values())


def discover():
    import psutil
    allowed = psutil.Process().cpu_affinity()
    cores = windows_cores() if os.name == "nt" else linux_cores()
    usable = [dict(c, usable_cpus=sorted(set(c["logical_cpus"]) & set(allowed))) for c in cores]
    usable = [c for c in usable if c["usable_cpus"]]
    usable.sort(key=lambda c: c["usable_cpus"][0])
    # Determine the performance class from the whole host, not just the current
    # mask: an externally restricted all-E mask must not relabel E cores as P.
    classes = {c["efficiency_class"] for c in cores if c["efficiency_class"] is not None}
    if len(classes) > 1:
        highest = max(classes)
        selected = [c for c in usable if c["efficiency_class"] == highest]
        evidence = "highest OS-reported efficiency/core class"
    elif len({c["smt"] for c in cores}) > 1:
        # Do not assert that SMT implies P-core. Require explicit selection instead.
        selected = []
        evidence = "heterogeneous SMT topology without differentiated OS core class; performance class unverified"
    else:
        selected = usable
        evidence = "uniform OS-reported core class"
    cpus = [c["usable_cpus"][0] for c in selected[:4]]
    return {"cores": cores, "allowed_cpus": allowed, "selection_basis": evidence,
            "recommended_batch_cpus": cpus if len(cpus) == 4 else [],
            "recommended_single_cpu": cpus[0] if cpus else None}


def validate_selection(topology, cpus, single_cpu=None):
    if len(cpus) not in (1, 4) or len(set(cpus)) != len(cpus):
        raise ValueError("Select exactly one or four distinct hardware threads")
    matches = []
    for cpu in cpus:
        if cpu not in topology["allowed_cpus"]:
            raise ValueError(f"CPU {cpu} is outside the allowed mask")
        matches.append(next(c for c in topology["cores"] if cpu in c["logical_cpus"]))
    if len({str(c["core_id"]) for c in matches}) != len(cpus):
        raise ValueError("Batch selection contains SMT siblings of the same physical core")
    if len({c["efficiency_class"] for c in matches}) != 1:
        raise ValueError("Batch selection mixes OS-reported core classes")
    classes = {c["efficiency_class"] for c in topology["cores"] if c["efficiency_class"] is not None}
    if len(classes) > 1 and matches[0]["efficiency_class"] != max(classes):
        raise ValueError("Primary placement requires the highest OS-reported performance class")
    if len(classes) <= 1 and len({c["smt"] for c in topology["cores"]}) > 1:
        raise ValueError("Heterogeneous host lacks verified performance-core class information")
    if len({c["smt"] for c in matches}) != 1:
        raise ValueError("Batch selection mixes SMT and non-SMT core types")
    if single_cpu is not None and single_cpu not in cpus:
        raise ValueError("The designated serial core must belong to the batch selection")
    return matches


def effective_thread_masks():
    import psutil
    process = psutil.Process()
    process_mask = set(process.cpu_affinity())
    rows = []
    if os.name != "nt":
        for thread in process.threads():
            mask = set(os.sched_getaffinity(thread.id))
            rows.append({"thread_id": thread.id, "allowed_cpus": sorted(mask)})
        return rows
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    class GroupAffinity(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_size_t), ("Group", ctypes.c_ushort),
                    ("Reserved", ctypes.c_ushort * 3)]
    kernel.OpenThread.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenThread.restype = ctypes.c_void_p
    kernel.GetThreadGroupAffinity.argtypes = [ctypes.c_void_p, ctypes.POINTER(GroupAffinity)]
    kernel.GetThreadGroupAffinity.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    for thread in process.threads():
        handle = kernel.OpenThread(0x0040, False, thread.id)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"Cannot verify thread {thread.id}")
        try:
            group = GroupAffinity()
            if not kernel.GetThreadGroupAffinity(handle, ctypes.byref(group)):
                raise OSError(ctypes.get_last_error(), "Cannot verify thread affinity")
            if group.Group != 0:
                raise RuntimeError("Unexpected worker processor group")
            requested = {i for i in range(64) if group.Mask & (1 << i)}
            rows.append({"thread_id": thread.id, "allowed_cpus": sorted(requested & process_mask),
                         "thread_requested_cpus": sorted(requested), "group": int(group.Group)})
        finally:
            kernel.CloseHandle(handle)
    return rows


def apply_and_verify(cpus):
    import psutil
    psutil.Process().cpu_affinity(cpus)
    if os.name != "nt":
        # Linux affinity belongs to threads; setting the calling thread alone
        # does not update already-created BLAS/runtime workers.
        for thread in psutil.Process().threads():
            os.sched_setaffinity(thread.id, cpus)
    return verify(cpus)


def verify(cpus):
    import psutil
    actual = psutil.Process().cpu_affinity()
    if set(actual) != set(cpus):
        raise RuntimeError(f"Process placement mismatch: {actual} != {cpus}")
    threads = effective_thread_masks()
    for thread in threads:
        mask = set(thread["allowed_cpus"])
        if mask != set(cpus):
            raise RuntimeError(f"Thread placement mismatch: {thread}")
    return {"process_mask": actual, "threads": threads,
            "binding": "pool-wide allowed mask; worker migration within mask permitted"}


def verify_ephemeral_jvx(cpus, call, cost):
    """Observe JVX's per-call Python pool in a separate untimed probe.

    The released API creates a ThreadPoolExecutor on each four-worker call.
    Only this excluded probe replaces its executor class, adding an initializer
    that checks inherited masks. A barrier makes all four initializer records
    observable even on tiny tasks. The original class is restored before any
    timing; this proves inheritance in the probe, not primary CPU residency.
    """
    import importlib
    import threading

    if len(cpus) != 4 or len(cost) < 4:
        raise ValueError("The four-worker JVX probe requires four CPUs and at least four matrices")
    verify(cpus)
    function = importlib.import_module("lap").lapjvxa_batch
    module = importlib.import_module(function.__module__)
    original = getattr(module, "ThreadPoolExecutor", None)
    if original is None:
        raise RuntimeError("Cannot observe this JVX API's per-call executor implementation")
    rows, lock = [], threading.Lock()
    barrier = threading.Barrier(4, timeout=10)

    class ObservedExecutor(original):
        def __init__(self, *args, **kwargs):
            initializer = kwargs.pop("initializer", None)
            initargs = kwargs.pop("initargs", ())

            def observed():
                if initializer is not None:
                    initializer(*initargs)
                tid = threading.get_native_id()
                matching = [row for row in effective_thread_masks() if row["thread_id"] == tid]
                if len(matching) != 1 or set(matching[0]["allowed_cpus"]) != set(cpus):
                    barrier.abort()
                    raise RuntimeError(f"JVX probe worker mask mismatch: {matching}")
                with lock:
                    rows.append(matching[0])
                barrier.wait()

            super().__init__(*args, initializer=observed, **kwargs)

    module.ThreadPoolExecutor = ObservedExecutor
    try:
        call(cost)
    finally:
        module.ThreadPoolExecutor = original
    if len({row["thread_id"] for row in rows}) != 4:
        raise RuntimeError(f"Expected four observed JVX worker threads, found {len(rows)}")
    after = verify(cpus)
    return {"workers": sorted(rows, key=lambda row: row["thread_id"]),
            "process_after": after, "pool_lifetime": "per complete API call",
            "probe": "excluded executor-initializer mask observation; executor restored before timing",
            "scope": "inherited allowed masks in this untimed probe; no primary CPU-residency claim"}
