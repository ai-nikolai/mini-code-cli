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

    def test_welcome_renders_without_error(self):
        c = ui.Console(stream=self.buf, color=False)
        c.welcome("1.0", "id", "/cache", "/home", "http://x", "model",
                  "perms", "auto", "enabled")
        self.assertIn("MiniCodeCLI", self.buf.getvalue())


if __name__ == "__main__":
    unittest.main()
