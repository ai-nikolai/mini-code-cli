"""Tests for the tool implementations: paged reads, atomic writes, fetch caps.

These are the small correctness properties the model relies on: a paged read
returns exactly the requested lines, an interrupted write cannot leave a
truncated file, and a fetch cannot pull an unbounded body into memory.
"""
import os
import shutil
import sys
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

    def test_whole_file_read_is_bounded(self):
        path = self.write("x" * (tools.DEFAULT_MAX_TOOL_OUTPUT_CHARS + 1_000))
        result = tools.read_file(path)
        self.assertLessEqual(len(result),
                             tools.DEFAULT_MAX_TOOL_OUTPUT_CHARS + 200)
        self.assertIn("truncated", result)
        self.assertIn("offset/limit", result)

    def test_large_paged_read_does_not_scan_for_the_total(self):
        path = self.write(("line\n") * 100)
        with mock.patch.object(tools, "MAX_LINE_COUNT_SCAN_BYTES", 10):
            result = tools.read_file(path, offset=1, limit=1)
        self.assertTrue(result.startswith("[lines 1-1]"))
        self.assertIn("total line count not read", result)

    def test_paged_read_is_bounded(self):
        path = self.write(("x" * 1_000 + "\n") * 200)
        result = tools.read_file(path, offset=1)
        self.assertLessEqual(len(result),
                             tools.DEFAULT_MAX_TOOL_OUTPUT_CHARS + 200)
        self.assertIn("truncated", result)


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


class ToolOutputBoundsTest(ToolTestCase):
    def test_grep_streams_long_lines(self):
        path = self.path("long.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x" * (tools.MAX_GREP_LINE_CHARS * 3) + "\nmatch\n")
        result = tools.grep("x+", path)
        self.assertIn("[line truncated]", result)
        self.assertIn("match", tools.grep("match", path))

    def test_grep_does_not_skip_the_line_after_an_exact_cap_line(self):
        path = self.path("exact.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x" * tools.MAX_GREP_LINE_CHARS + "\nneedle\n")
        # The cap-width line was complete, so the next line must still be read.
        self.assertIn("2:needle", tools.grep("needle", path))
        # ... and the complete line must not claim it was truncated.
        self.assertNotIn("[line truncated]", tools.grep("^x+$", path))

    def test_grep_matches_past_the_display_cap(self):
        path = self.path("one-line.js")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x" * (tools.MAX_GREP_LINE_CHARS * 4) + "needle\n")
        result = tools.grep("needle", path)
        self.assertIn("1:", result)
        self.assertIn("match at column", result)

    def test_grep_output_is_bounded_even_with_unlimited_results(self):
        path = self.path("many.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("match\n" * 40_000)
        result = tools.grep("match", path, max_results=0)
        self.assertLessEqual(len(result),
                             tools.DEFAULT_MAX_TOOL_OUTPUT_CHARS + 300)
        self.assertIn("truncated", result)

    def test_glob_output_is_bounded(self):
        fake = (f"/tmp/file-{index}.py"
                for index in range(tools.MAX_GLOB_RESULTS + 10))
        with mock.patch.object(tools.glob_module, "iglob", return_value=fake):
            result = tools.glob("*.py")
        self.assertIn("truncated", result)
        self.assertLessEqual(len(result.splitlines()),
                             tools.MAX_GLOB_RESULTS + 2)


class WebSearchTest(unittest.TestCase):
    HTML = """
    <a rel="nofollow" class="result__a" href="https://example.test/one">
      First <b>result</b> &amp; more</a>
    <a class="result__snippet" href="#">First snippet.</a>
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.test%2Ftwo">
      Second result</a>
    <a class="result__snippet" href="#">Second snippet.</a>
    """

    def test_parses_ddg_html_results_and_unwraps_redirects(self):
        with mock.patch.object(tools, "_post", return_value=self.HTML):
            results = tools._ddg_html_results("query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "First result & more")
        self.assertEqual(results[0][1], "https://example.test/one")
        self.assertEqual(results[0][2], "First snippet.")
        self.assertEqual(results[1][1], "https://example.test/two")

    def test_parses_ddg_lite_results(self):
        body = """
        <a rel="nofollow" href="https://example.test/one"
           class='result-link'>First result</a>
        <td class='result-snippet'>First snippet.</td>
        """
        with mock.patch.object(tools, "_post", return_value=body):
            results = tools._ddg_lite_results("query")
        self.assertEqual(results, [("First result", "https://example.test/one",
                                    "First snippet.")])

    def test_challenge_page_raises_so_callers_can_fall_back(self):
        with mock.patch.object(
                tools, "_post",
                return_value="<html>anomaly challenge</html>"):
            with self.assertRaises(RuntimeError):
                tools._ddg_html_results("query")

    def test_web_search_falls_back_to_answers_when_html_is_blocked(self):
        with mock.patch.object(tools, "_ddg_lite_results",
                               side_effect=RuntimeError("blocked")), \
                mock.patch.object(tools, "_ddg_html_results",
                                  side_effect=RuntimeError("blocked")), \
                mock.patch.object(tools, "_ddg_instant_results",
                                  return_value=[("Example", "https://example.test",
                                                 "An example abstract.")]), \
                mock.patch.object(tools, "_wikipedia_results",
                                  side_effect=AssertionError("not needed")):
            text = tools.web_search("example")
        self.assertIn("Example", text)
        self.assertIn("https://example.test", text)
        self.assertIn("An example abstract.", text)

    def test_web_search_rejects_empty_queries(self):
        self.assertIn("must not be empty", tools.web_search("  "))


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

    def test_read_website_rejects_non_http_schemes(self):
        result = tools.read_website("file:///etc/passwd")
        self.assertIn("unsupported URL scheme", result)
        self.assertNotIn("root:", result)
        with self.assertRaises(ValueError):
            tools._fetch("file:///etc/passwd")

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


class ShellOutputCapTest(ToolTestCase):

    def test_small_output_is_returned_without_a_file(self):
        result = tools.run_shell_command("echo hello", cwd=self.directory)
        self.assertIn("hello", result)
        self.assertFalse(os.path.exists(
            os.path.join(self.directory, ".mini", "tool-output")))

    def test_large_output_is_capped_and_saved_in_full(self):
        size = tools.DEFAULT_MAX_SHELL_OUTPUT_CHARS * 2
        command = f"{sys.executable} -c \"print('x' * {size})\""
        result = tools.run_shell_command(command, cwd=self.directory)

        self.assertLess(len(result), tools.DEFAULT_MAX_SHELL_OUTPUT_CHARS + 2_000)
        self.assertIn("truncated", result)
        self.assertIn("full output saved to", result)

        directory = os.path.join(self.directory, ".mini", "tool-output")
        files = os.listdir(directory)
        self.assertEqual(len(files), 1)
        path = os.path.join(directory, files[0])
        self.assertIn(path, result)
        with open(path, encoding="utf-8") as handle:
            self.assertGreaterEqual(len(handle.read()), size)

    def test_a_missing_worktree_still_returns_a_bounded_result(self):
        # A read-only cwd must not make the tool raise just because the full
        # output cannot be saved; the heading/exit code still comes back.
        with mock.patch.object(tools, "_save_shell_output", return_value=None):
            result = tools.run_shell_command(
                f"{sys.executable} -c \"print('y' * 60000)\"",
                cwd=self.directory)
        self.assertIn("full output could not be saved", result)
        self.assertLess(len(result), tools.DEFAULT_MAX_SHELL_OUTPUT_CHARS + 2_000)


if __name__ == "__main__":
    unittest.main()
