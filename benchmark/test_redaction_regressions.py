"""Path cleaning must preserve benchmark values, URLs and every identity record."""
import importlib.util
import json
from pathlib import Path
import unittest

BENCHMARK = Path(__file__).resolve().parent
FILES = [BENCHMARK / suite / "run.py" for suite in ("base", "formulations", "scaling", "extended")]
FILES += [BENCHMARK / "lsape" / "redaction.py", BENCHMARK / "real_world" / "realbench_paths.py"]


def load_redactors():
    for index, path in enumerate(FILES):
        spec = importlib.util.spec_from_file_location(f"redaction_regression_{index}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield path.parent.name, module.redact


class RedactionRegressions(unittest.TestCase):
    def test_numerical_json_is_unchanged(self):
        record = {"median": 1.25, "rounds": 9, "passed": True, "missing": None,
                  "samples": [0, -2.5, 3.14159], "hash": "a" * 64}
        for suite, redact in load_redactors():
            with self.subTest(suite=suite):
                self.assertEqual(redact(record), record)
                self.assertEqual(json.loads(json.dumps(redact(record))), record)

    def test_urls_including_queries_and_fragments_are_unchanged(self):
        urls = ["https://example.org/a/b?q=/foo/bar#section/path",
                "https://example.org/search?path=C:/Windows/System32/file.exe",
                "ftp://example.org/folder/file.zip"]
        for suite, redact in load_redactors():
            with self.subTest(suite=suite):
                self.assertEqual(redact(urls), urls)
                self.assertEqual(redact("See " + " and ".join(urls)), "See " + " and ".join(urls))

    def test_distinct_path_keys_keep_all_records_and_filenames(self):
        record = {r"C:\Other\one.py": {"sha256": "one"},
                  r"C:\Other\two.py": {"sha256": "two"},
                  r"C:\Other\same.py": {"sha256": "three"},
                  r"D:\Elsewhere\same.py": {"sha256": "four"}}
        for suite, redact in load_redactors():
            with self.subTest(suite=suite):
                cleaned = redact(record)
                self.assertEqual(len(cleaned), len(record))
                self.assertEqual({item["sha256"] for item in cleaned.values()}, {"one", "two", "three", "four"})
                self.assertIn("<external>/one.py", cleaned)
                self.assertIn("<external>/two.py", cleaned)
                self.assertEqual(redact(cleaned), cleaned)

    def test_spaced_paths_compiler_flags_and_surrounding_prose(self):
        messages = [r'Opening C:\Other User\Private Data\run.py failed',
                    'Opening /home/Example User/Private Data/run.py failed',
                    r'Opening "C:\Other User\private log.json" failed',
                    'Opening "/home/Example User/private log.json" failed',
                    r'\\server\share\Example User\run.py',
                    '-I/home/Example User/include', '-L/home/Example User/lib',
                    r'/IC:\Program Files\Build Tools\include']
        for suite, redact in load_redactors():
            with self.subTest(suite=suite):
                cleaned = redact(messages)
                rendered = json.dumps(cleaned)
                for private in ("Other User", "Example User", "Private Data", "server", "share", "Program Files", "Build Tools"):
                    self.assertNotIn(private, rendered)
                self.assertTrue(cleaned[0].endswith("run.py failed"))
                self.assertTrue(cleaned[1].endswith("run.py failed"))
                self.assertIn("private log.json", cleaned[2])
                self.assertTrue(cleaned[5].startswith("-I<external>"))
                self.assertTrue(cleaned[6].startswith("-L<external>"))

    def test_two_windows_paths_keep_intervening_diagnostic_text(self):
        message = r"Include C:\Program Files\Build Tools\include and source D:\private\bridge.cpp"
        for suite, redact in load_redactors():
            with self.subTest(suite=suite):
                self.assertEqual(redact(message), "Include <external>/include and source <external>/bridge.cpp")


if __name__ == "__main__":
    unittest.main()
