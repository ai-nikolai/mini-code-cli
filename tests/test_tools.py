"""Tests for the tool implementations: paged reads, atomic writes, fetch caps.

These are the small correctness properties the model relies on: a paged read
returns exactly the requested lines, an interrupted write cannot leave a
truncated file, and a fetch cannot pull an unbounded body into memory.
"""
import os
import shutil
import tempfile
import unittest
from unittest import mock

from mini_code_cli import tools


class ToolTestCase(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def path(self, name):
        return os.path.join(self.directory, name)


class ReadFileRangeTest(ToolTestCase):

    def write(self, content):
        path = self.path("sample.txt")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return path

    def test_whole_file_read_is_unchanged(self):
        path = self.write("one\ntwo\nthree\n")
        self.assertEqual(tools.read_file(path), "one\ntwo\nthree\n")

    def test_offset_and_limit_return_the_slice_with_a_header(self):
        path = self.write("one\ntwo\nthree\nfour\nfive\n")
        self.assertEqual(tools.read_file(path, offset=2, limit=2),
                         "[lines 2-3 of 5]\ntwo\nthree\n")

    def test_offset_only_reads_to_the_end(self):
        path = self.write("one\ntwo\nthree\n")
        self.assertEqual(tools.read_file(path, offset=2),
                         "[lines 2-3 of 3]\ntwo\nthree\n")

    def test_limit_only_reads_from_the_start(self):
        path = self.write("one\ntwo\nthree\n")
        self.assertEqual(tools.read_file(path, limit=2),
                         "[lines 1-2 of 3]\none\ntwo\n")

    def test_a_limit_larger_than_the_file_is_clamped(self):
        path = self.write("one\ntwo\n")
        self.assertEqual(tools.read_file(path, offset=1, limit=99),
                         "[lines 1-2 of 2]\none\ntwo\n")

    def test_no_trailing_newline_is_preserved(self):
        path = self.write("one\ntwo")
        self.assertEqual(tools.read_file(path, offset=2),
                         "[lines 2-2 of 2]\ntwo")

    def test_offset_past_the_end_is_reported(self):
        path = self.write("one\n")
        result = tools.read_file(path, offset=2)
        self.assertIn("past the end", result)

    def test_missing_file_and_directory_are_reported(self):
        self.assertIn("no such file", tools.read_file(self.path("nope")))
        self.assertIn("is a directory", tools.read_file(self.directory))


class AtomicWriteTest(ToolTestCase):

    def test_overwrite_and_append(self):
        path = self.path("out.txt")
        self.assertIn("Wrote to", tools.write_file(path, "a"))
        self.assertIn("Appended to", tools.write_file(path, "b", append=True))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "ab")

    def test_create_dirs(self):
        path = self.path("a/b/c.txt")
        tools.write_file(path, "x")
        self.assertTrue(os.path.isfile(path))

    def test_overwrite_preserves_the_file_mode(self):
        path = self.path("script.sh")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("old")
        os.chmod(path, 0o755)
        tools.write_file(path, "new")
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o755)

    def test_no_temporary_file_is_left_behind(self):
        path = self.path("out.txt")
        tools.write_file(path, "x")
        tools.edit(path, "x", "y")
        leftovers = [n for n in os.listdir(self.directory)
                     if n.startswith(".mini-write-")]
        self.assertEqual(leftovers, [])

    def test_edit_preserves_crlf_line_endings(self):
        path = self.path("crlf.txt")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("a\r\nb\r\n")
        tools.edit(path, "a\r\n", "A\r\n")
        with open(path, "r", encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read(), "A\r\nb\r\n")

    def test_edit_refuses_missing_or_ambiguous_matches(self):
        path = self.path("edit.txt")
        tools.write_file(path, "x x")
        self.assertIn("not found", tools.edit(path, "y", "z"))
        self.assertIn("ambiguous", tools.edit(path, "x", "y"))
        self.assertIn("replaced 2", tools.edit(path, "x", "y", replace_all=True))


class FetchCapTest(unittest.TestCase):

    def test_read_website_caps_the_download_to_the_requested_size(self):
        calls = {}

        def fake_fetch(url, timeout=None, max_bytes=None):
            calls["max_bytes"] = max_bytes
            return "body"

        with mock.patch.object(tools, "_fetch", new=fake_fetch):
            tools.read_website("http://example.invalid", max_chars=1000)
        # Four bytes per requested character (UTF-8's worst case).
        self.assertEqual(calls["max_bytes"], 4_000)

    def test_read_website_never_exceeds_the_hard_fetch_ceiling(self):
        calls = {}

        def fake_fetch(url, timeout=None, max_bytes=None):
            calls["max_bytes"] = max_bytes
            return "body"

        with mock.patch.object(tools, "_fetch", new=fake_fetch):
            tools.read_website("http://example.invalid", max_chars=10_000_000)
        self.assertEqual(calls["max_bytes"], tools.DEFAULT_MAX_FETCH_BYTES)
        with mock.patch.object(tools, "_fetch", new=fake_fetch):
            tools.read_website("http://example.invalid", max_chars=0)
        self.assertEqual(calls["max_bytes"], tools.DEFAULT_MAX_FETCH_BYTES)


if __name__ == "__main__":
    unittest.main()
