import os
import json
import unittest
from textwrap import dedent
from tempfile import TemporaryDirectory

from numba.tests.support import TestCase, run_in_subprocess
from numba.misc.chrome_trace import ChromeTraceConfig


class TestChromeTraceModule(TestCase):
    """
    Test chrome tracing generated file(s).
    """

    def test_trace_output(self):
        code = """
            from numba import njit
            import numpy as np

            x = np.arange(100).reshape(10, 10)

            @njit
            def go_fast(a):
                trace = 0.0
                for i in range(a.shape[0]):
                    trace += np.tanh(a[i, i])
                return a + trace

            go_fast(x)
        """

        src = dedent(code)
        with TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_trace.json")
            env = os.environ.copy()
            env['NUMBA_CHROME_TRACE'] = path
            run_in_subprocess(src, env=env)
            with open(path) as file:
                events = json.load(file)
                self.assertIsInstance(events, list)
                for ev in events:
                    self.assertIsInstance(ev, dict)
                    # check that each record has the right fields.
                    self.assertEqual(
                        set(ev.keys()),
                        {"cat", "pid", "tid", "ph", "name", "args", "ts"},
                    )


class TestChromeTraceConfig(TestCase):
    def test_basic(self):
        ctc = ChromeTraceConfig.parse("chrome.json")
        self.assertFalse(ctc.support_multiprocessing)
        self.assertEqual(ctc.apply(), "chrome.json")
        self.assertEqual(ctc.filename_pattern, "chrome.json")

    def test_with_pid_pattern(self):
        ctc = ChromeTraceConfig.parse("trace_{pid}.json")
        self.assertTrue(ctc.support_multiprocessing)
        result = ctc.apply()
        self.assertTrue(result.startswith("trace_"))
        self.assertTrue(result.endswith(".json"))
        self.assertIn(str(os.getpid()), result)

    def test_with_ts_pattern(self):
        ctc = ChromeTraceConfig.parse("trace_{ts}.json")
        self.assertTrue(ctc.support_multiprocessing)
        result = ctc.apply()
        self.assertTrue(result.startswith("trace_"))
        self.assertTrue(result.endswith(".json"))

    def test_with_multiple_patterns(self):
        ctc = ChromeTraceConfig.parse("trace_{pid}_{ts}.json")
        self.assertTrue(ctc.support_multiprocessing)
        result = ctc.apply()
        self.assertTrue(result.startswith("trace_"))
        self.assertTrue(result.endswith(".json"))
        self.assertIn(str(os.getpid()), result)

    def test_empty_pattern(self):
        ctc = ChromeTraceConfig.parse("")
        self.assertFalse(ctc.support_multiprocessing)
        self.assertEqual(ctc.apply(), "")
        self.assertFalse(ctc)

    def test_parsed_parts(self):
        ctc = ChromeTraceConfig.parse("prefix_{pid}_suffix")
        self.assertEqual(len(ctc.parsed_parts), 2)
        self.assertEqual(ctc.parsed_parts[0].literal_text, "prefix_")
        self.assertEqual(ctc.parsed_parts[0].field_name, "pid")
        self.assertEqual(ctc.parsed_parts[1].literal_text, "_suffix")

    def test_no_patterns(self):
        ctc = ChromeTraceConfig.parse("simple_filename.json")
        self.assertFalse(ctc.support_multiprocessing)
        self.assertEqual(ctc.apply(), "simple_filename.json")
        field_names = [part.field_name
                       for part in ctc.parsed_parts if part.field_name]
        self.assertEqual(field_names, [])

    def test_invalid_pattern(self):
        ctc = ChromeTraceConfig.parse("trace_{invalid_field}.json")
        self.assertTrue(ctc.support_multiprocessing)
        with self.assertRaises(ValueError) as cm:
            ctc.apply()
        error_msg = str(cm.exception)
        self.assertIn("invalid_field", error_msg)
        self.assertIn("Unsupported format field", error_msg)
        self.assertIn("pid", error_msg)
        self.assertIn("ts", error_msg)


if __name__ == "__main__":
    unittest.main()
