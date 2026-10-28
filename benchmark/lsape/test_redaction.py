import unittest
from pathlib import Path

from redaction import redact


class RedactionTests(unittest.TestCase):
    def test_paths_in_nested_records_and_compiler_arguments(self):
        record = {
            "executable": r"C:\Users\Example\python.exe",
            "flags": [r"/IC:\private\includes", r"/FoD:\build\output.obj",
                      r"/LIBPATH:E:\libraries", r"C:\Windows\System32\cmd.exe"],
            "network": r"\\server\share\private\report.json",
            "traceback": 'File "/home/example/project/run.py", line 4',
        }
        cleaned = str(redact(record)).lower()
        for path in (r"c:\users", r"c:\private", r"d:\build", r"e:\libraries",
                     r"c:\windows", r"\\server\share", "/home/example"):
            self.assertNotIn(path, cleaned)
        self.assertIn("<external>", cleaned)

    def test_benchmark_root_is_portable(self):
        cleaned = redact(str(Path(__file__).resolve().parent / "results" / "report.json"))
        self.assertTrue(cleaned.startswith("<lsape>"))
        self.assertIn("report.json", cleaned)


if __name__ == "__main__":
    unittest.main()
