"""Worker failures and timeouts must leave useful logs without local paths."""
import importlib.util
import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("extended_benchmark", Path(__file__).parent / "extended" / "run.py")
extended = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extended)


class WorkerLogTests(unittest.TestCase):
    def test_failed_worker_traceback_is_redacted(self):
        log = io.StringIO()
        result = subprocess.CompletedProcess(["worker"], 1,
                    'File "C:\\Private User\\project\\run.py", line 42\nValueError: invalid input\n')
        with patch.object(extended.subprocess, "run", return_value=result):
            self.assertEqual(extended.run_logged(["worker"], log, 10).returncode, 1)
        self.assertNotIn("Private User", log.getvalue())
        self.assertIn("ValueError: invalid input", log.getvalue())

    def test_timeout_preserves_redacted_partial_output(self):
        log = io.StringIO()
        error = subprocess.TimeoutExpired(["worker"], 10,
                    output=b'File "/home/private/project/run.py", line 42\n')
        with patch.object(extended.subprocess, "run", side_effect=error):
            with self.assertRaises(subprocess.TimeoutExpired):
                extended.run_logged(["worker"], log, 10)
        self.assertNotIn("/home/private", log.getvalue())
        self.assertIn("line 42", log.getvalue())


if __name__ == "__main__":
    unittest.main()
