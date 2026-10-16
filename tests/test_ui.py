"""Tests for the dependency-free terminal UI (ui.Console).

The interesting behaviour here is *graceful degradation*: colour must be
stripped (and unicode fall back to ASCII) when output is not a TTY or when the
NO_COLOR convention is honoured, and styling must appear when colour is forced
on. We instantiate Consoles directly with explicit flags so the tests do not
depend on the ambient terminal.
"""
import io
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli import ui


class TestConsoleRendering(unittest.TestCase):
    def setUp(self):
        # Capturing to a StringIO lets us inspect exactly what gets written.
        self.buf = io.StringIO()

    # -- capability detection ----------------------------------------------
    def test_no_color_disabled(self):
        c = ui.Console(stream=self.buf, color=False)
        self.assertFalse(c.color_enabled)
        self.assertFalse(c.unicode_enabled)

    def test_forced_color_enabled(self):
        c = ui.Console(stream=self.buf, color=True)
        self.assertTrue(c.color_enabled)

    def test_no_color_env_is_honoured(self):
        os.environ["NO_COLOR"] = "1"
        try:
            # Force unicode off explicitly so the tty does not matter; only the
            # colour branch (which reads the env) is under test here.
            c = ui.Console(stream=self.buf, unicode=False)
            self.assertFalse(c.color_enabled)
        finally:
            os.environ.pop("NO_COLOR", None)

    def test_force_color_env_overrides_no_tty(self):
        os.environ["FORCE_COLOR"] = "1"
        try:
            c = ui.Console(stream=io.StringIO())
            self.assertTrue(c.color_enabled)
        finally:
            os.environ.pop("FORCE_COLOR", None)

    # -- styling ------------------------------------------------------------
    def test_style_strips_ansi_when_no_color(self):
        c = ui.Console(stream=self.buf, color=False)
        self.assertEqual(c.style("hi", fg="red", bold=True), "hi")

    def test_style_adds_ansi_when_color(self):
        c = ui.Console(stream=self.buf, color=True)
        out = c.style("hi", fg="red", bold=True)
        self.assertIn("\033[", out)
        self.assertTrue(out.endswith("\033[0m"))
        # The literal text is preserved between the escape codes.
        self.assertIn("hi", out)

    # -- box ----------------------------------------------------------------
    def test_box_has_no_ansi_when_no_color(self):
        c = ui.Console(stream=io.StringIO(), color=False)
        out = c.box(["line one", "line two"], title="Title")
        self.assertNotIn("\033", out)
        # Box is closed on both sides of the title line.
        self.assertIn("Title", out)
        # The corner/border characters are ASCII in fallback mode.
        self.assertIn("-", out)

    def test_box_adds_ansi_when_color(self):
        c = ui.Console(stream=io.StringIO(), color=True)
        out = c.box(["hi"], title="T")
        self.assertIn("\033[", out)

    def test_box_pads_lines_to_same_width(self):
        c = ui.Console(stream=io.StringIO(), color=False)
        out = c.box(["a", "longer line"])
        # Every rendered line should have identical visual width.
        widths = {len(line) for line in out.splitlines()}
        self.assertEqual(len(widths), 1)

    # -- semantic renderers -------------------------------------------------
    def test_usage_inline_handles_no_usage(self):
        c = ui.Console(stream=io.StringIO(), color=False)
        self.assertIn("n/a", c.usage_inline(None))
        self.assertIn("input=1", c.usage_inline({"prompt_tokens": 1}))

    def test_usage_summary_lines(self):
        c = ui.Console(stream=self.buf, color=False)
        c.usage_summary({"llm_calls": 2, "prompt_tokens": 10, "completion_tokens": 5,
                         "total_tokens": 15, "tool_calls": 3})
        text = self.buf.getvalue()
        for label in ("LLM calls", "Tool calls", "Total tokens"):
            self.assertIn(label, text)

    def test_note_no_ansi_without_color(self):
        c = ui.Console(stream=self.buf, color=False)
        c.note("hello", "warn")
        self.assertNotIn("\033", self.buf.getvalue())

    def test_tool_group_error_rendered_with_color(self):
        c = ui.Console(stream=self.buf, color=True)
        c.tool_group("read_file", {"file_path": "/x"}, "Error: boom")
        self.assertIn("\033[", self.buf.getvalue())

    def test_turn_renders_multiline_with_indent(self):
        c = ui.Console(stream=self.buf, color=False)
        c.turn("Assistant", "line1\nline2")
        text = self.buf.getvalue()
        self.assertIn("Assistant", text)
        self.assertIn("   line2", text)

    def test_short_divider_renders_without_ansi(self):
        c = ui.Console(stream=self.buf, color=False)
        c.short_divider()
        out = self.buf.getvalue().rstrip("\n")
        # A short run of dashes, not the full-width divider.
        self.assertTrue(set(out) <= {"-", " "})
        self.assertLess(len(out.strip("-")), 12)
        self.assertIn("-", out)

    def test_short_divider_shorter_than_full_divider(self):
        c = ui.Console(stream=self.buf, color=False)
        c.short_divider()
        short = len(self.buf.getvalue().rstrip("\n").strip())
        self.buf.seek(0); self.buf.truncate()
        c.divider()
        full = len(self.buf.getvalue().rstrip("\n").strip())
        self.assertLess(short, full)

    def test_welcome_renders_without_error(self):
        c = ui.Console(stream=self.buf, color=False)
        c.welcome("1.0", "id", "/cache", "/home", "http://x", "model",
                  "perms", "auto", "enabled")
        self.assertIn("MiniCodeCLI", self.buf.getvalue())


if __name__ == "__main__":
    unittest.main()


class TestConsoleParentInput(unittest.TestCase):
    """A swallowed console (parent_console set) routes input to the parent.

    Covers the sub-agent wiring: when a sub-agent's own console has its output
    redirected away from the terminal, its interactive input (permission
    answers, any further prompts) must be answered on the console that actually
    owns the screen -- the parent -- so the main agent responds in place of the
    sub-agent instead of the human or hanging on an answer nobody reads.
    """

    def setUp(self):
        # A real parent console (no colour so it stays clean/piped).
        self.parent_buf = io.StringIO()
        self.parent = ui.Console(stream=self.parent_buf, color=False, unicode=False)
        # The swallowed console: output goes to its own buffer, input routes to
        # the parent (the console that owns the terminal).
        self.child = ui.Console(stream=io.StringIO(), color=False, unicode=False,
                                parent_console=self.parent)

    def test_read_line_delegates_to_parent(self):
        # Feeding stdin lines, the child must not read them itself; it forwards
        # the whole prompt to the parent and returns what the parent read.
        with mock.patch("builtins.input", return_value="hello") as m:
            self.assertEqual(self.child.read_line("q: "), "hello")
        # The prompt was passed straight to the parent's input() call.
        self.assertEqual(m.call_args[0], ("q: ",))

    def test_read_line_none_when_parent_reads_none(self):
        # On EOF / Ctrl-C the parent returns None, and the child passes that
        # straight through (no crash, no swallowed-but-unread value).
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertIsNone(self.child.read_line("q: "))

    def test_read_line_without_parent_reads_itself(self):
        # A standalone console (no parent) keeps reading input() directly -- the
        # normal interactive path is unchanged.
        standalone = ui.Console(stream=io.StringIO(), color=False, unicode=False)
        with mock.patch("builtins.input", return_value="self") as m:
            self.assertEqual(standalone.read_line("x: "), "self")
        self.assertEqual(m.call_args[0], ("x: ",))

    def test_ask_permission_box_and_answer_on_parent(self):
        # The permission box must be drawn on the parent (the one on screen), and
        # the answer read from the parent. When the parent reads 'y', proceed; when
        # it reads 'n', refuse. This is what lets the main agent answer the
        # sub-agent's permission prompt.
        with mock.patch("builtins.input", return_value="y"):
            self.assertTrue(self.child.ask_permission("run_shell_command", "{}"))
        self.assertIn("Permission needed", self.parent_buf.getvalue())
        self.assertIn("Tool       : run_shell_command", self.parent_buf.getvalue())

        self.parent_buf.truncate(0)
        self.parent_buf.seek(0)
        with mock.patch("builtins.input", return_value="n"):
            self.assertFalse(self.child.ask_permission("read_file", "{}"))

