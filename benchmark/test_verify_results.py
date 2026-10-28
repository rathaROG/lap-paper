"""Saved-result integrity and numerical tables must verify without measurements."""
import io
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

    def test_default_zip_verifies_manifest_and_removes_temporary_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            package = Path(directory) / "results.zip"
            with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in root.iterdir():
                    archive.write(path, path.name)
            original_archive = package.read_bytes()
            extracted = []

            def verify_extracted(path):
                extracted.append(path)
                self.assertNotEqual(path, root)
                self.assertTrue((path / "manifest.json").is_file())
                return verifier.verify_manifest(path)

            root.rename(Path(directory) / "fixture-data")
            with patch.object(verifier, "HERE", Path(directory)), \
                 patch.object(verifier, "_verify_directory", side_effect=verify_extracted):
                self.assertEqual(verifier.default_results(), package)
                self.assertEqual(verifier.verify(), {"files": 2, "archive_members": 1})
            self.assertEqual(len(extracted), 1)
            self.assertFalse(extracted[0].exists())
            self.assertEqual(package.read_bytes(), original_archive)
            self.assertFalse(root.exists())

    def test_existing_results_directory_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            with patch.object(verifier, "HERE", Path(directory)), \
                 patch.object(verifier, "_verify_directory", side_effect=verifier.verify_manifest):
                self.assertEqual(verifier.default_results(), root)
                self.assertEqual(verifier.verify(root), {"files": 2, "archive_members": 1})
            self.assertTrue(root.is_dir())

    def test_default_archive_is_used_when_results_contains_new_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            package = Path(directory) / "results.zip"
            with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in root.iterdir():
                    archive.write(path, path.name)
            root.rename(Path(directory) / "fixture-data")
            fresh = root / "full-20261029"
            fresh.mkdir(parents=True)
            sample = fresh / "samples.csv"
            sample.write_text("nanoseconds\n123\n", encoding="utf-8")
            with patch.object(verifier, "HERE", Path(directory)), \
                 patch.object(verifier, "_verify_directory", side_effect=verifier.verify_manifest):
                self.assertEqual(verifier.default_results(), package)
                self.assertEqual(verifier.verify(), {"files": 2, "archive_members": 1})
                package.unlink()
                self.assertEqual(verifier.default_results(), package)
            self.assertEqual(sample.read_text(encoding="utf-8"), "nanoseconds\n123\n")

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


if __name__ == "__main__":
    unittest.main()
