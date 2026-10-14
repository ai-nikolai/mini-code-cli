"""Tests for tool refactors (change #4) + shell timeout (change #3).

Ensures tools return their results/errors as strings instead of printing,
and that the shell tool honors a timeout.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli import tools


class TestToolsReturnValues(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "note.txt")

    def test_write_file_returns_message(self):
        r = tools.write_file(self.path, "hello")
        self.assertIsInstance(r, str)
        self.assertIn("Wrote to file", r)
        self.assertIn(self.path, r)
        with open(self.path) as f:
            self.assertEqual(f.read(), "hello")

    def test_write_file_error_returned_not_printed(self):
        # Writing into a non-existent directory should yield an Error string.
        bad = "/nonexistent_dir_xyz/deep/note.txt"
        r = tools.write_file(bad, "x")
        self.assertIsInstance(r, str)
        self.assertTrue(r.startswith("Error"), r)

    def test_read_file_returns_contents(self):
        tools.write_file(self.path, "body text")
        self.assertEqual(tools.read_file(self.path), "body text")

    def test_read_missing_returns_error_string(self):
        r = tools.read_file(os.path.join(self.tmp.name, "nope.txt"))
        self.assertIsInstance(r, str)
        self.assertTrue(r.startswith("Error"), r)

    def test_run_shell_command_returns_output(self):
        r = tools.run_shell_command("echo hi", timeout=10)
        self.assertEqual(r.strip(), "hi")

    def test_run_shell_command_includes_stderr(self):
        r = tools.run_shell_command("echo oops 1>&2", timeout=10)
        self.assertIn("STDERR", r)
        self.assertIn("oops", r)

    def test_schema_exposes_timeout_param(self):
        schema = tools.util_get_tools()
        rc = next(t for t in schema if t["function"]["name"] == "run_shell_command")
        self.assertIn("timeout", rc["function"]["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
