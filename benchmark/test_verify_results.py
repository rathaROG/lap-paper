"""Saved-result integrity and numerical tables must verify without measurements."""
import io
import csv
import copy
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import verify_results as verifier


def archive_bytes(value):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("samples.json", json.dumps(value))
    return stream.getvalue()


def identity(data):
    return {"bytes": len(data), "sha256": verifier.digest(data)}


class SavedResultIntegrityTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / "results"
        root.mkdir()
        value = {"samples": [3, 1.25, None, True]}
        current = json.dumps(value).encode()
        (root / "data.json").write_bytes(current)
        packed = archive_bytes(value)
        (root / "records.zip").write_bytes(packed)
        with zipfile.ZipFile(io.BytesIO(packed)) as archive:
            member = identity(archive.read("samples.json"))
        manifest = {"files": {"data.json": identity(current), "records.zip": identity(packed)},
                    "archives": {"records.zip": {"samples.json": member}}}
        (root / "manifest.json").write_text(json.dumps(manifest))
        return root, manifest

    def test_current_manifest_verifies_saved_files_and_archive_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            self.assertEqual(verifier.verify_manifest(root), {"files": 2, "archive_members": 1})

    def test_changed_result_file_fails_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            (root / "data.json").write_text('{"samples": [99]}')
            with self.assertRaisesRegex(ValueError, "file checksum"):
                verifier.verify_manifest(root)

    def test_archive_member_is_checked_even_if_archive_hash_was_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            root, manifest = self.fixture(directory)
            current = archive_bytes({"samples": [0]})
            (root / "records.zip").write_bytes(current)
            manifest["files"]["records.zip"] = identity(current)
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "archive member checksum"):
                verifier.verify_manifest(root)

    def test_manifest_cannot_read_outside_result_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Unsafe manifest path"):
                verifier.local(Path(directory), "../private.json")

    def test_zip_extraction_preserves_archive_and_removes_temporary_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            package = Path(directory) / "saved.zip"
            with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in root.iterdir():
                    archive.write(path, path.name)
            original_archive = package.read_bytes()
            with verifier.results_directory(package) as extracted:
                self.assertNotEqual(extracted, root)
                self.assertEqual(verifier.verify_manifest(extracted), {"files": 2, "archive_members": 1})
            self.assertFalse(extracted.exists())
            self.assertEqual(package.read_bytes(), original_archive)

    def test_existing_results_directory_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            with verifier.results_directory(root) as selected:
                self.assertEqual(selected, root)
                self.assertEqual(verifier.verify_manifest(selected), {"files": 2, "archive_members": 1})
            self.assertTrue(root.is_dir())

    def test_current_bundles_are_required_without_historical_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "results.zip").write_bytes(b"obsolete archive must not be read")
            with patch.object(verifier, "HERE", base):
                self.assertIsNone(verifier.default_windows_results())
                self.assertIsNone(verifier.default_linux_results())
                with self.assertRaisesRegex(ValueError, "Required current results bundle is missing"):
                    verifier.verify()

    def test_archive_rejects_unsafe_members_before_extracting_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for member in ("../private.json", "/private.json", "C:/private.json", "folder\\private.json"):
                archive_path = base / "unsafe.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("valid.json", "{}")
                    info = zipfile.ZipInfo(member)
                    info.filename = member
                    archive.writestr(info, "{}")
                output = base / "extracted"
                with self.subTest(member=member), self.assertRaisesRegex(ValueError, "Unsafe"):
                    verifier.extract_results(archive_path, output)
                self.assertFalse(output.exists())

    def test_archive_rejects_symlinks_and_case_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            archive_path = base / "unsafe.zip"
            info = zipfile.ZipInfo("linked.json")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(info, "../outside.json")
            with self.assertRaisesRegex(ValueError, "links and special files"):
                verifier.extract_results(archive_path, base / "linked")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("samples.json", "{}")
                archive.writestr("SAMPLES.JSON", "{}")
            with self.assertRaisesRegex(ValueError, "Duplicate results archive member"):
                verifier.extract_results(archive_path, base / "aliased")
            self.assertFalse((base / "linked").exists())
            self.assertFalse((base / "aliased").exists())


class PaperTableTests(unittest.TestCase):
    TABLE = r"""\begin{table}[H]
\caption{A caption with 999 unrelated numbers.}
\label{tab:4}
\begin{tabularx}{\linewidth}{Xrr}
\textbf{Case} & \textbf{First method} & \textbf{Second method} \\
Shape 64 \ensuremath{\times} 512 & 1.25 & 23.50 \\
\end{tabularx}
\end{table}
"""

    def test_numeric_table_check_allows_prose_changes(self):
        expected = {"4": ([[1.246, 23.5]], 2)}
        self.assertEqual(verifier.check_paper_tables(self.TABLE, expected), 1)
        revised = "A new public introduction.\n" + self.TABLE.replace("A caption with 999 unrelated numbers.",
                                                                     "An ordinary result caption.")
        self.assertEqual(verifier.check_paper_tables(revised, expected), 1)

    def test_numeric_table_check_detects_changed_measurement(self):
        with self.assertRaisesRegex(ValueError, "Paper table does not match"):
            verifier.check_paper_tables(self.TABLE.replace("23.50", "23.51"),
                                        {"4": ([[1.25, 23.5]], 2)})

    def test_ranges_and_interval_cells_use_per_value_precision(self):
        table = self.TABLE.replace(r"\textbf{First method}", r"\textbf{Python 3.11.17}")
        table = table.replace("1.25 & 23.50", r"\shortstack[r]{1.25\\{[1.2, 1.3]}} & 23.50--24.75")
        expected = {"4": ([[1.246, 1.2, 1.3, 23.504, 24.75]], [2, 1, 1, 2, 2])}
        self.assertEqual(verifier.check_paper_tables(table, expected), 1)
        with self.assertRaisesRegex(ValueError, "Paper table does not match"):
            verifier.check_paper_tables(table.replace("[1.2, 1.3]", "[1.2, 1.4]"), expected)


class UpdatedWindowsTests(unittest.TestCase):
    def test_updated_observations_pass_independent_saved_data_audit(self):
        with verifier.results_directory(verifier.default_windows_results()) as root:
            result = verifier.verify_updated_windows(root)
        self.assertEqual(result["environments"], 2)
        self.assertEqual(result["timing_samples"], 17892)
        self.assertEqual(result["memory_observations"], 830)
        self.assertEqual(result["exact_objectives_compared"], 3925)

    def test_changed_summary_fails_arithmetic_even_with_refreshed_checksum(self):
        with verifier.results_directory(verifier.default_windows_results()) as root:
            path = root / "py311-results/base/single_summary.csv"
            records = verifier.rows(path)
            original = path.read_text(encoding="utf-8")
            first = records[0]["median_ns"]
            changed = original.replace(first, str(float(first) + 100), 1)
            path.write_text(changed, encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = verifier.read(manifest_path)
            manifest["files"][path.relative_to(root).as_posix()] = identity(path.read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from"):
                verifier.verify_updated_windows(root)

    def test_windows_source_only_controls_its_tables_and_b7(self):
        with verifier.results_directory(verifier.default_windows_results()) as current, \
                verifier.results_directory(verifier.default_linux_results()) as linux_root:
            windows = tuple(verifier.table_data(current / name) for name in ("py314-results", "py311-results"))
            linux = tuple(verifier.table_data(linux_root / name) for name in ("py314-results", "py311-results"))
            original = verifier.controlled_paper_tables(windows, linux)
            changed = copy.deepcopy(windows)
            changed[0]["single"]["square128", "jvrust"] *= 2
            revised = verifier.controlled_paper_tables(changed, linux)
            self.assertNotEqual(original["4"], revised["4"])
            for label in ("B1", "B2", "B3", "B4", "B5", "B6"):
                self.assertEqual(revised[label], original[label])


@unittest.skipUnless(verifier.default_linux_results(), "Current Linux bundle is not present")
class UpdatedLinuxTests(unittest.TestCase):
    def test_updated_linux_observations_pass_independent_audit(self):
        with verifier.results_directory(verifier.default_linux_results()) as root:
            result = verifier.verify_updated_linux(root)
        self.assertEqual(result["environments"], 2)
        self.assertEqual(result["timing_samples"], 17892)
        self.assertEqual(result["memory_observations"], 830)
        self.assertEqual(result["exact_objectives_compared"], 3925)

    def test_linux_summary_tampering_fails_even_with_refreshed_checksum(self):
        with verifier.results_directory(verifier.default_linux_results()) as root:
            path = root / "py314-results/scaling/summary.csv"
            records = verifier.rows(path)
            original = path.read_text(encoding="utf-8")
            value = records[0]["median_ns"]
            path.write_text(original.replace(value, str(float(value) + 100), 1), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = verifier.read(manifest_path)
            manifest["files"][path.relative_to(root).as_posix()] = identity(path.read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from"):
                verifier.verify_updated_linux(root)

    def test_wrong_platform_cannot_be_selected_as_linux_evidence(self):
        with verifier.results_directory(verifier.default_windows_results()) as root:
            with self.assertRaisesRegex(ValueError, "platform differs"):
                verifier.verify_updated_linux(root)

    def test_changed_harness_identity_fails_even_with_refreshed_checksum(self):
        with verifier.results_directory(verifier.default_linux_results()) as root:
            path = root / "py311-results/base/environment.json"
            environment = verifier.read(path)
            environment["script_sha256"] = "0" * 64
            path.write_text(json.dumps(environment), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = verifier.read(manifest_path)
            manifest["files"][path.relative_to(root).as_posix()] = identity(path.read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "harness source differs"):
                verifier.verify_updated_linux(root)

    def test_linux_controls_only_appendix_b_and_b7_uses_all_current_environments(self):
        with verifier.results_directory(verifier.default_windows_results()) as windows_root, \
                verifier.results_directory(verifier.default_linux_results()) as linux_root:
            windows = tuple(verifier.table_data(windows_root / name) for name in ("py314-results", "py311-results"))
            linux = tuple(verifier.table_data(linux_root / name) for name in ("py314-results", "py311-results"))
            original = verifier.controlled_paper_tables(windows, linux)
            changed = copy.deepcopy(linux)
            changed[0]["single"]["square128", "jvrust"] *= 2
            revised = verifier.controlled_paper_tables(windows, changed)
            self.assertNotEqual(original["B1"], revised["B1"])
            for label in original:
                if not label.startswith("B"):
                    self.assertEqual(revised[label], original[label])
            self.assertEqual(original["B7"][0], [verifier.native_overview(data)
                             for data in (*windows, *linux)])
            self.assertEqual(original["B7"][1], [0, 0, 0, 3])
            from apply_publication_tables import format_numeric_cells
            cells = format_numeric_cells({"values": original["B7"][0], "places": original["B7"][1]}, 0)
            self.assertEqual(cells[:3], ["195", "197", "76"])
            self.assertEqual(cells[3], "1.162")


@unittest.skipUnless(verifier.default_application_results(), "Current application bundle is not present")
class UpdatedApplicationTests(unittest.TestCase):
    def test_current_application_is_recomputed_from_raw_observations(self):
        with verifier.results_directory(verifier.default_application_results()) as root:
            result = verifier.verify_application_update(root)
        self.assertEqual(result["tracking_samples"], 648)
        self.assertEqual(result["replay_samples"], 1296)
        self.assertEqual(result["captured_matrices"], 7539)
        self.assertEqual(result["objective_checks"], 150780)

    def test_changed_application_projection_fails_even_with_refreshed_checksum(self):
        with verifier.results_directory(verifier.default_application_results()) as root:
            path = root / "real_world/projections.json"
            projection = verifier.read(path)
            projection["tables"]["C2"]["numeric_values"][0][0] += 0.1
            path.write_text(json.dumps(projection), encoding="utf-8")
            manifest_path = root / "manifest.json"
            manifest = verifier.read(manifest_path)
            manifest["files"][path.relative_to(root).as_posix()] = identity(path.read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "projection differs from saved observations"):
                verifier.verify_application_update(root)


@unittest.skipUnless(verifier.default_lsape_results(), "Current LSAPE bundle is not present")
class UpdatedLSAPETests(unittest.TestCase):
    def test_fresh_lsape_raw_sessions_and_strict_records_verify(self):
        with verifier.results_directory(verifier.default_lsape_results()) as root:
            result = verifier.verify_lsape_update(root)
        self.assertEqual(result["timing_round_averages"], 9045)
        self.assertEqual(result["independent_raw_audit"]["timed_input_identities"], 804)
        self.assertEqual(result["independent_raw_audit"]["strict_certificates"], 50950)

    def test_changed_paired_ratio_fails_even_when_manifest_and_projection_are_updated(self):
        with verifier.results_directory(verifier.default_lsape_results()) as root:
            raw = root / "lsape/summary/all_cases.csv"
            records = verifier.rows(raw)
            records[0]["s1_ratio"] = str(float(records[0]["s1_ratio"]) + .1)
            with raw.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            flat = root / "lsape/all_cases.csv"
            flat.write_bytes(raw.read_bytes())
            manifest_path = root / "manifest.json"
            manifest = verifier.read(manifest_path)
            for path in (raw, flat):
                manifest["files"][path.relative_to(root).as_posix()] = identity(path.read_bytes())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "paired-log session ratio"):
                verifier.verify_lsape_update(root)

    def test_lsape_table_projection_is_confined_to_its_two_tables(self):
        with verifier.results_directory(verifier.default_windows_results()) as windows, \
                verifier.results_directory(verifier.default_linux_results()) as linux, \
                verifier.results_directory(verifier.default_application_results()) as application, \
                verifier.results_directory(verifier.default_lsape_results()) as lsape:
            original = verifier.expected_paper_tables(windows, linux, application, lsape)
            path = lsape / "lsape/paper_table.csv"
            data = verifier.rows(path)
            value = data[0]["O_ms"]
            path.write_text(path.read_text(encoding="utf-8").replace(value, str(float(value) + 1), 1), encoding="utf-8")
            revised = verifier.expected_paper_tables(windows, linux, application, lsape)
            self.assertNotEqual(original["10"], revised["10"])
            for label in original:
                if label not in ("10", "11"):
                    self.assertEqual(revised[label], original[label])


if __name__ == "__main__":
    unittest.main()
