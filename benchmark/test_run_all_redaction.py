"""The launcher must not leak paths into logs, console output, or records."""
import contextlib
from datetime import date
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_all


class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.stdout = iter(['worker: C:\\Other\\private\\result.json\n',
                            'system: C:\\Windows\\System32\\tool.exe\n',
                            'see https://example.org/benchmark/results\n'])

    def wait(self):
        return 0


class RedactionTests(unittest.TestCase):
    def test_default_output_uses_local_date_and_script_directory_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / 'benchmark'
            benchmark.mkdir()
            other = root / 'elsewhere'
            other.mkdir()
            output = benchmark / 'results' / 'full-20261102'
            with patch.object(run_all, 'HERE', benchmark), \
                 patch.object(run_all.datetime, 'date') as local_date, \
                 patch('sys.argv', ['run_all.py']), \
                 patch.object(run_all.subprocess, 'Popen', side_effect=FakeProcess) as process, \
                 patch.object(run_all.threading, 'Thread'), \
                 patch.object(run_all.time, 'sleep'), \
                 contextlib.chdir(other), contextlib.redirect_stdout(io.StringIO()):
                local_date.today.return_value = date(2026, 11, 2)
                run_all.main()
            record = json.loads((output / 'run_all.json').read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'completed')
            self.assertEqual([item['suite'] for item in record['suites']], list(run_all.SUITES))
            self.assertEqual(process.call_count, len(run_all.SUITES))
            for suite, call in zip(run_all.SUITES, process.call_args_list):
                self.assertEqual(call.args[0][-1], str(output / suite))
                self.assertEqual(call.kwargs['cwd'], benchmark / suite)
            self.assertFalse((other / 'results').exists())

    def test_existing_default_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            benchmark = Path(directory) / 'benchmark'
            output = benchmark / 'results' / 'full-20261102'
            output.mkdir(parents=True)
            marker = output / 'saved.json'
            marker.write_bytes(b'{"saved":123}\n')
            with patch.object(run_all, 'HERE', benchmark), \
                 patch.object(run_all.datetime, 'date') as local_date, \
                 patch('sys.argv', ['run_all.py']), \
                 patch.object(run_all.subprocess, 'Popen') as process:
                local_date.today.return_value = date(2026, 11, 2)
                with self.assertRaisesRegex(SystemExit, 'Choose a new output directory'):
                    run_all.main()
            process.assert_not_called()
            self.assertEqual(marker.read_bytes(), b'{"saved":123}\n')
            self.assertEqual(list(output.iterdir()), [marker])

    def test_scrub_preserves_urls_and_hides_unknown_and_system_paths(self):
        line = 'C:\\Other\\secret\\data.json C:\\Windows\\System32\\cmd.exe https://example.org/a/b'
        clean = run_all.scrub(line)
        self.assertNotIn('C:\\', clean)
        self.assertIn('https://example.org/a/b', clean)

    def test_quoted_windows_path_with_spaces_and_posix_traceback(self):
        message = ('opening "C:\\Other User\\Private Data\\result.json"\n'
                   '  File "/home/private/project/run.py", line 3, in main\n')
        clean = run_all.scrub(message)
        self.assertNotIn('Other User', clean)
        self.assertNotIn('Private Data', clean)
        self.assertNotIn('/home/private/project', clean)
        self.assertIn('line 3, in main', clean)

    def test_launcher_scrubs_console_saved_log_and_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results'
            stdout = io.StringIO()
            with patch('sys.argv', ['run_all.py', '--output', str(output), '--suites', 'base']), \
                 patch.object(run_all.subprocess, 'Popen', FakeProcess), \
                 patch.object(run_all.threading, 'Thread') as thread, \
                 patch.object(run_all.time, 'sleep'), contextlib.redirect_stdout(stdout):
                run_all.main()
            thread.return_value.start.assert_called_once()
            contents = stdout.getvalue() + (output / 'base.log').read_text(encoding='utf-8')
            contents += (output / 'run_all.json').read_text(encoding='utf-8')
            self.assertNotIn(str(output), contents)
            self.assertNotIn('C:\\Other', contents)
            self.assertNotIn('C:\\Windows', contents)
            self.assertIn('https://example.org/benchmark/results', contents)
            record = json.loads((output / 'run_all.json').read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'completed')


if __name__ == '__main__':
    unittest.main()
