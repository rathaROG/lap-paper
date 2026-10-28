"""Input integrity, path portability and unchanged measured core semantics."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run as portable
import realbench_paths as paths


class PortableTests(unittest.TestCase):
    def test_packaged_data_verifies_without_original_videos_or_models(self):
        result = portable.verify_inputs()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["cases"]), 4)
        self.assertEqual({row["frames"] for row in result["cases"]}, {999})

    def test_gzip_materialization_preserves_exact_frozen_json_bytes(self):
        destination = paths.materialize_inputs()
        for name, identity in paths.data_manifest()["files"].items():
            if name.startswith("inputs/"):
                self.assertEqual(paths.sha256(destination / Path(name).name.removesuffix(".gz")), identity["uncompressed_sha256"])

    def test_corrupt_packaged_input_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lap-realworld-corrupt-") as directory:
            root = Path(directory)
            (root / "provenance").mkdir()
            (root / "data" / "inputs").mkdir(parents=True)
            name = "inputs/mot16_04_gt.json.gz"
            identity = paths.data_manifest()["files"][name]
            (root / "provenance" / "data_manifest.json").write_text(json.dumps({"files": {name: identity}}), encoding="utf-8")
            (root / "data" / name).write_bytes(b"corrupted input")
            with patch.object(portable, "PACKAGE", root), patch.object(paths, "PACKAGE", root):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    portable.verify_inputs()

    def test_solver_and_timing_core_ast_matches_measured_snapshots(self):
        manifest = json.loads((paths.PACKAGE / "provenance" / "core_semantics.json").read_text(encoding="utf-8"))
        for filename, functions in manifest["ast_sha256"].items():
            tree = ast.parse((paths.PACKAGE / filename).read_text(encoding="utf-8"))
            nodes = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
            for name, expected in functions.items():
                with self.subTest(file=filename, function=name):
                    actual = hashlib.sha256(ast.dump(nodes[name], include_attributes=False).encode()).hexdigest()
                    self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
