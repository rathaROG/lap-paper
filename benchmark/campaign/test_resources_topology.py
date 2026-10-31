"""Safety/protocol tests using small allocations and mocked watchdog clocks."""
from __future__ import annotations

import hashlib
import json
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from . import resources, topology
from .catalog import INPUT_COUNT
from .adapters import COST_LAYOUTS, SHAPE_LAYOUTS, cost_layout, shape_layout
from .common import input_digest


def case(rows=2, cols=3, optional=True, kind="single", batch=8):
    return dict(rows=rows, cols=cols, objective="optional" if optional else "required",
                limit=.5 if optional else None, kind=kind, metadata=dict(batch_size=batch))


def topology_fixture():
    cores = [dict(core_id=i, logical_cpus=[2*i, 2*i+1], efficiency_class=1, smt=True)
             for i in range(4)]
    cores += [dict(core_id=i+4, logical_cpus=[8+i], efficiency_class=0, smt=False)
              for i in range(4)]
    return dict(cores=cores, allowed_cpus=list(range(12)))


class ResourceTests(unittest.TestCase):
    def test_all_declared_external_layout_cost_arrays(self):
        a = np.arange(6, dtype=np.float64).reshape(2, 3)
        for layout in COST_LAYOUTS:
            actual, _ = cost_layout(a, .5, layout)
            for backend in ("scipy", "jvx", "jvs"):
                estimate = resources.allocation_estimate(case(), f"cost:{backend}:{layout}")
                self.assertEqual(actual.nbytes, estimate["transformed_cost_array_bytes"])
        for layout in SHAPE_LAYOUTS:
            actual, _ = shape_layout(a, layout)
            estimate = resources.allocation_estimate(case(optional=False), f"shape:scipy:{layout}")
            self.assertEqual(actual.nbytes, estimate["transformed_cost_array_bytes"])

    def test_batch_jvx_and_memory_resident_counts(self):
        for optional, expected in ((True, 25*8), (False, 9*8)):
            estimate = resources.allocation_estimate(case(optional=optional, kind="batch"), "batch:jvx:4")
            self.assertEqual(estimate["transformed_cost_array_bytes"], expected)
            self.assertEqual(estimate["workers"], 4)
            self.assertEqual(estimate["input_bytes"], INPUT_COUNT * 8 * 2 * 3 * 8)
            self.assertEqual(estimate["output_reserve_bytes"], 2 * 8 * 2 * 16)
        self.assertEqual(resources.allocation_estimate(case(kind="memory"), "scipy:square")["input_count"], 1)
        large = resources.allocation_estimate(case(4096, 16384), "scipy:square")
        self.assertEqual(large["transformed_cost_array_bytes"], 3.125 * resources.GIB)
        self.assertEqual(large["input_bytes"], INPUT_COUNT * resources.GIB / 2)
        self.assertLess(large["estimated_peak_bytes"], resources.R1["admission_bytes"])

    def test_unavailable_host_is_postponed_not_solver_censored(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=resources.GIB)):
            record = resources.admission(case(), ["scipy:capped"])
        self.assertEqual(record["host_status"], "postponed")
        self.assertEqual(record["methods"]["scipy:capped"]["status"], "postponed")

    def test_host_lock_initialization_failure_closes_handle_and_explains_contention(self):
        for operation in ("seek", "read", "write", "flush"):
            with self.subTest(operation=operation):
                lock_file = Mock()
                lock_file.read.return_value = b""
                getattr(lock_file, operation).side_effect = PermissionError("locked byte")
                with patch("builtins.open", return_value=lock_file):
                    with self.assertRaisesRegex(RuntimeError, "Another campaign measurement owns the host lock"):
                        resources.HostLock().__enter__()
                lock_file.close.assert_called_once_with()

    def test_watchdog_enforces_accumulated_method_budget(self):
        process = Mock(pid=123, returncode=-1)
        process.poll.return_value = None
        state = dict(monotonic=99., phase="primary", method="x", active_seconds=7199.5)
        with patch.object(resources.time, "monotonic", return_value=100.), \
             patch.object(resources, "process_tree_rss", return_value=1), \
             patch("benchmark.campaign.common.read_json", return_value=state), \
             patch.object(resources, "terminate_tree") as terminate:
            result = resources.watch(process, "unused")
        terminate.assert_called_once_with(123)
        self.assertEqual(result["reason"], "method_active_time_limit")

    def test_watchdog_fails_closed_when_memory_cannot_be_observed(self):
        process = Mock(pid=123, returncode=-1)
        process.poll.return_value = None
        with patch.object(resources, "process_tree_rss", side_effect=PermissionError("denied")), \
             patch.object(resources, "terminate_tree") as terminate:
            result = resources.watch(process, "unused")
        terminate.assert_called_once_with(123)
        self.assertEqual(result["reason"], "resource_monitor_error")

    def test_input_digest_preserves_previous_bytes_contract(self):
        for value in (np.arange(12, dtype=np.float64).reshape(3, 4),
                      np.arange(12, dtype=np.float64).reshape(3, 4).T,
                      np.empty((0, 3)), np.arange(24, dtype=np.float64).reshape(2, 3, 4)):
            h = hashlib.sha256()
            h.update(b"1")
            h.update(json.dumps([list(value.shape), value.dtype.str]).encode())
            h.update(np.ascontiguousarray(value).tobytes())
            self.assertEqual(input_digest(value), h.hexdigest())


class TopologyTests(unittest.TestCase):
    def test_windows_core_record_parsing_and_bounds(self):
        def record(mask, efficiency, smt):
            raw = bytearray(48)
            struct.pack_into("<II", raw, 0, 0, 48)
            struct.pack_into("BB", raw, 8, int(smt), efficiency)
            struct.pack_into("<H", raw, 30, 1)
            struct.pack_into("<QH", raw, 32, mask, 0)
            return raw
        raw = record(3, 1, True) + record(4, 0, False)
        cores = topology._parse_windows_cores(raw)
        self.assertEqual(cores[0]["logical_cpus"], [0, 1])
        self.assertEqual(cores[0]["efficiency_class"], 1)
        self.assertEqual(cores[1]["logical_cpus"], [2])
        with self.assertRaises(RuntimeError):
            topology._parse_windows_cores(raw[:-1])
        with self.assertRaises(RuntimeError):
            topology._parse_windows_cores(record(3, 1, True) * 2)

    def test_primary_selection_rejects_siblings_mixed_and_all_e(self):
        t = topology_fixture()
        self.assertEqual(len(topology.validate_selection(t, [0, 2, 4, 6], 0)), 4)
        for cpus in ([0, 1, 4, 6], [0, 2, 4, 8], [8, 9, 10, 11]):
            with self.assertRaises(ValueError):
                topology.validate_selection(t, cpus)

    def test_linux_hybrid_classes_come_from_the_pmu_cpu_lists(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            sysfs = Path(directory)
            # Four SMT P cores (cpus 0-7) and four E cores (cpus 8-11), like 12th-gen Intel.
            for cpu in range(12):
                top = sysfs / "devices" / "system" / "cpu" / f"cpu{cpu}" / "topology"
                top.mkdir(parents=True)
                (top / "physical_package_id").write_text("0")
                (top / "core_id").write_text(str(cpu // 2 if cpu < 8 else cpu))
            for name, cpus in (("cpu_core", "0-7"), ("cpu_atom", "8-11")):
                (sysfs / "devices" / name).mkdir(parents=True)
                (sysfs / "devices" / name / "cpus").write_text(cpus + "\n")
            cores = topology.linux_cores(sysfs)
        self.assertEqual(sorted(len(c["logical_cpus"]) for c in cores), [1] * 4 + [2] * 4)
        self.assertEqual({c["efficiency_class"] for c in cores if c["smt"]}, {1})
        self.assertEqual({c["efficiency_class"] for c in cores if not c["smt"]}, {0})
        t = dict(cores=cores, allowed_cpus=list(range(12)))
        self.assertEqual(len(topology.validate_selection(t, [0, 2, 4, 6], 0)), 4)
        with self.assertRaises(ValueError):
            topology.validate_selection(t, [8, 9, 10, 11])
        self.assertEqual(topology._cpu_list("0-2,5,7-8"), {0, 1, 2, 5, 7, 8})

    def test_verify_requires_the_exact_mask_on_every_thread(self):
        process = Mock()
        process.cpu_affinity.return_value = [0, 2, 4, 6]
        with patch("psutil.Process", return_value=process), \
             patch.object(topology, "effective_thread_masks", return_value=[dict(thread_id=7, allowed_cpus=[0])]):
            with self.assertRaises(RuntimeError):
                topology.verify([0, 2, 4, 6])


if __name__ == "__main__":
    unittest.main()
