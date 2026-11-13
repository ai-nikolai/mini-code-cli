"""The opt-in ``--tui`` prompt: history, editing and multiline input."""

import io
import os
import pty
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from mini_code_cli import agent_loop, tui, ui
from mini_code_cli.agent_loop import AgentSession

from test_e2e_agent_loop import make_args


class TuiCliTest(unittest.TestCase):
    def test_tui_modes_default_to_auto(self):
        with mock.patch.object(sys, "argv", ["mini-code"]):
            self.assertEqual(agent_loop.parse_args().tui, "auto")
        with mock.patch.object(sys, "argv", ["mini-code", "--tui"]):
            self.assertEqual(agent_loop.parse_args().tui, "on")
        with mock.patch.object(sys, "argv", ["mini-code", "--tui=off"]):
            self.assertEqual(agent_loop.parse_args().tui, "off")
        with mock.patch.object(sys, "argv", ["mini-code", "--tui=auto"]):
            self.assertEqual(agent_loop.parse_args().tui, "auto")
        with mock.patch.object(sys, "argv", ["mini-code", "--all"]):
            self.assertEqual(agent_loop.parse_args().tui, "auto")

    def test_auto_support_is_conservative(self):
        stdin, stdout = _FakeTty(), _FakeTty()
        with mock.patch.object(tui.sys, "stdin", stdin), \
                mock.patch.object(tui.sys, "stdout", stdout), \
                mock.patch.dict(os.environ, {"TERM": "xterm-256color"}):
            self.assertTrue(tui.TuiConsole.supported())
        with mock.patch.object(tui.sys, "stdin", stdin), \
                mock.patch.object(tui.sys, "stdout", stdout), \
                mock.patch.dict(os.environ, {"TERM": "dumb"}):
            self.assertFalse(tui.TuiConsole.supported())
        with mock.patch.object(tui.sys, "stdin", io.StringIO()), \
                mock.patch.object(tui.sys, "stdout", stdout), \
                mock.patch.dict(os.environ, {"TERM": "xterm-256color"}):
            self.assertFalse(tui.TuiConsole.supported())


class BaseMultilinePtyTest(unittest.TestCase):
    def test_plain_multiline_editor_still_accepts_shift_enter(self):
        child = r"""
import sys
from mini_code_cli.ui import Console
c = Console(stream=sys.stdout, color=False, unicode=False)
sys.stdout.write("READY\n")
sys.stdout.flush()
text = c.read_line("", multiline=True)
sys.stdout.write("\nRESULT:%r\n" % text)
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            time.sleep(0.3)
            os.write(master, b"a\x1b[13;2ub\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'a\\nb'", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_escape_then_a_letter_keeps_the_letter(self):
        child = r"""
import sys
from mini_code_cli.ui import Console
c = Console(stream=sys.stdout, color=False, unicode=False)
sys.stdout.write("READY\n")
sys.stdout.flush()
text = c.read_line("", multiline=True)
sys.stdout.write("\nRESULT:%r\n" % text)
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            time.sleep(0.3)
            os.write(master, b"\x1ba\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'a'", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)


class _FakeTty(io.StringIO):
    def isatty(self):
        return True


class _FakeStdin:
    def isatty(self):
        return True

    def fileno(self):
        return 0


class TuiConsoleTest(unittest.TestCase):
    def console(self, color=False):
        return tui.TuiConsole(stream=_FakeTty(), color=color, unicode=False)

    def _feed_input(self, console, data):
        """Drive the TUI editor with ``data`` bytes, then EOF."""
        calls = [bytes([byte]) for byte in data] + [b""]
        with mock.patch.object(tui.sys, "stdin", _FakeStdin()), \
                mock.patch.object(tui.os, "read", side_effect=calls), \
                mock.patch("select.select",
                           side_effect=lambda r, w, x, t: (list(r), [], [])), \
                mock.patch("termios.tcgetattr",
                           side_effect=lambda fd: [0, 0, 0, 0, 0, 0, [0] * 32]), \
                mock.patch("termios.tcsetattr"), \
                mock.patch("termios.tcflush"), \
                mock.patch.object(tui.TuiConsole, "start_scroll_listener"), \
                mock.patch.object(tui.TuiConsole, "stop_scroll_listener"):
            return console.read_line("", multiline=True, tui=True)

    @staticmethod
    def _caret(output):
        matches = re.findall(r"\x1b\[(\d+);(\d+)H\x1b\[\?25h", output)
        return tuple(map(int, matches[-1]))

    def test_clear_screen_keeps_the_conversation_and_clears_notices(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))), \
                mock.patch.object(tui.TuiConsole, "start_scroll_listener"):
            console.start_tui()
            console.turn("You", "hello")
            console.turn("Assistant", "hi there")
            console.note("watch out", "warn")
            console.status_box(["old command output"], title="help")
            console.scroll_body(2)

            console.clear_screen()

            content = "\n".join(console._content_lines(50))
            self.assertIn("hello", content)
            self.assertIn("hi there", content)
            self.assertEqual(list(console._turns), [
                {"role": "You", "content": "hello", "turn_id": 1,
                 "tools": [], "background": []},
                {"role": "Assistant", "content": "hi there", "turn_id": 1,
                 "tools": [], "background": []},
            ])
            self.assertEqual(console._notices, [])
            self.assertEqual(console._notice_line, "")
            self.assertEqual(console._overlay_lines, [])
            self.assertEqual(console._scroll_offset, 0)
            console.stop_tui()

    def test_eof_with_a_non_empty_buffer_returns_the_text(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))), \
                mock.patch.object(tui.TuiConsole, "start_scroll_listener"):
            console.start_tui()
            text = self._feed_input(console, b"abc")
            console.stop_tui()

        self.assertEqual(text, "abc")
        self.assertEqual(console._input_buffer, [])
        self.assertFalse(console._input_active)

    def test_caret_column_uses_display_width_for_cjk(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))), \
                mock.patch.object(tui.TuiConsole, "start_scroll_listener"):
            console.start_tui()
            console._input_buffer = list("日")
            console._input_cursor = 1
            console._input_active = True
            console.render()
            row, col = self._caret(console.stream.getvalue())
            console.stop_tui()

        # One wide glyph occupies two cells: text starts at column 3.
        self.assertEqual(col, 5)
        self.assertGreater(row, 0)

    def test_caret_row_follows_a_tall_input_into_view(self):
        console = self.console()
        lines = [f"line {i}" for i in range(30)]
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((40, 20))), \
                mock.patch.object(tui.TuiConsole, "start_scroll_listener"):
            console.start_tui()
            console._input_buffer = list("\n".join(lines))
            console._input_cursor = len(console._input_buffer)
            console._input_active = True
            console.render()
            output = console.stream.getvalue()
            row, _col = self._caret(output)
            frame = console._last_frame_lines
            top = next(i for i, line in enumerate(frame)
                       if line.startswith("+ You"))
            console.stop_tui()

        # 10 input rows are visible and the cursor is on the last one.
        self.assertEqual(row, top + 2 + 9)
        self.assertLess(row, 20)

    def test_dashboard_info_line_is_readable(self):
        session = object.__new__(AgentSession)
        session.total_usage = {"tool_calls": 189, "llm_calls": 138}
        session.background_tasks = mock.Mock()
        session.background_tasks.list.return_value = []

        self.assertEqual(
            session._tui_info_line(),
            "189 tool calls · 138 LLM calls · no background tasks")

        session.background_tasks.list.return_value = [
            {"status": "running"}, {"status": "done"},
            {"status": "cancelling"}, {"status": "cancelled"},
        ]
        self.assertEqual(
            session._tui_info_line(),
            "189 tool calls · 138 LLM calls · "
            "2 of 4 background tasks running")

        session.total_usage = {"tool_calls": 1, "llm_calls": 1}
        session.background_tasks.list.return_value = [{"status": "done"}]
        self.assertEqual(
            session._tui_info_line(),
            "1 tool call · 1 LLM call · 0 of 1 background task running")

    def test_tui_on_falls_back_on_an_unsupported_terminal(self):
        with tempfile.TemporaryDirectory() as workdir, \
                tempfile.TemporaryDirectory() as cache:
            args = make_args(workdir, cache)
            args.tui = "on"
            stream = io.StringIO()
            console = tui.TuiConsole(stream=stream, color=False, unicode=False)
            session = AgentSession(args, console=console)
            with mock.patch.object(AgentSession, "print_welcome"), \
                    mock.patch.object(AgentSession, "print_announcements"), \
                    mock.patch.object(AgentSession, "_warn_compaction_budget"), \
                    mock.patch.object(AgentSession, "_print_session_id"), \
                    mock.patch.object(AgentSession, "_emit_tip"), \
                    mock.patch.object(AgentSession, "_read_user_line",
                                      return_value=None):
                session.run()

        self.assertFalse(session.tui)
        self.assertIn("TUI is not available", stream.getvalue())

    def test_escape_then_a_letter_does_not_swallow_the_letter(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))), \
                mock.patch.object(tui.TuiConsole, "start_scroll_listener"):
            console.start_tui()
            text = self._feed_input(console, b"\x1ba\r")
            console.stop_tui()

        self.assertEqual(text, "a")

    def test_fixed_view_redraws_summary_body_and_input_box(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            self.assertTrue(console.start_tui())
            console.set_summary(["version 1.0", "model test"])
            console.turn("You", "hello")
            console.turn("Assistant", "hi there")
            console.note("watch out", "warn")
            console._input_buffer = list("abc")
            console._input_cursor = 3
            console._input_active = True
            console.render()
            console.stop_tui()
            output = console.stream.getvalue()

        self.assertIn("Mini-Code", output)
        self.assertIn("version 1.0", output)
        self.assertIn("hello", output)
        self.assertIn("hi there", output)
        self.assertIn("watch out", output)
        self.assertIn("abc", output)
        self.assertIn("Enter send", output)

    def test_dashboard_shows_the_ascii_mark(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((80, 24))):
            console.start_tui()
            console.set_summary(["version 1.0", "model test"])
            console.render()
            content = "\n".join(console._last_frame_lines)
            console.stop_tui()
        self.assertIn("__  __", content)
        self.assertIn("version 1.0", content)

    def test_the_middle_pane_scrolls(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((50, 14))):
            console.start_tui()
            console.set_summary(["summary"])
            console.turn("Assistant", "\n".join(f"line {i}" for i in range(40)))
            console.render()
            self.assertEqual(console._scroll_offset, 0)
            console.scroll_body(1)
            self.assertEqual(console._scroll_offset, 1)
            self.assertGreater(console._total_body_lines, 10)
            self.assertIn(f"scrolled 1/{console._total_body_lines}",
                          console.stream.getvalue())
            console.scroll_body(-1)
            self.assertEqual(console._scroll_offset, 0)
            console.stop_tui()

    def test_escape_returns_a_scrolled_view_to_the_bottom(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((50, 14))):
            console.start_tui()
            console.turn("Assistant", "\n".join(f"line {i}" for i in range(40)))
            console.render()
            console.scroll_body(5)
            self.assertEqual(console._scroll_offset, 5)
            console.dismiss_overlay()
            self.assertEqual(console._scroll_offset, 0)
            console.stop_tui()

    def test_info_and_warning_messages_are_separated_at_the_bottom(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((50, 16))):
            console.start_tui()
            console.note("first warning", "warn")
            console.note("second warning", "warn")
            console.note("third warning", "warn")
            console.note("latest info", "info")
            lines = console._content_lines(45)
            plain = [ui._ANSI_RE.sub("", line) for line in lines]
            text = "\n".join(plain)
            self.assertNotIn("first warning", text)
            for kept in ("second warning", "third warning", "latest info"):
                self.assertIn(kept, text)
            warning_index = next(i for i, line in enumerate(plain)
                                 if "latest info" in line)
            separator = console.glyph("h")
            self.assertTrue(any(set(line.strip()) == {separator}
                                for line in plain[:warning_index]))
            console.stop_tui()

    def test_escape_does_not_dismiss_an_active_question(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((50, 16))):
            console.start_tui()
            console.question_box("Question to User", ["Continue?"])
            console.dismiss_overlay()
            self.assertTrue(console._question_active)
            console.stop_tui()

    def test_assistant_blocks_show_turn_tool_and_background_counts(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((70, 22))):
            console.start_tui()
            console.turn("You", "inspect and test")
            console.turn("Assistant", "reading the file")
            console.start_tool_call(tool_name="read_file")
            console.tool_call_display("read_file", {"file_path": "x"}, "body")
            console.stop_tool_call()
            console.start_tool_call(tool_name="background")
            console.tool_call_display(
                "background", {"action": "start", "command": "pytest -q"},
                "started bg-1")
            console.stop_tool_call()
            console.turn("Assistant", "done")
            last_screen = "\n".join(
                ui._ANSI_RE.sub("", line)
                for line in console._last_frame_lines)
            console.stop_tui()
        self.assertIn("tools: read_file, background", last_screen)
        self.assertIn("background: pytest -q", last_screen)
        self.assertIn("messages: 2", last_screen)

    def test_core_view_keeps_multiple_turns_with_separators(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.turn("You", "first question")
            console.turn("Assistant", "first answer")
            console.turn("You", "second question")
            console.turn("Assistant", "second answer")
            console.render()
            output = console.stream.getvalue()
            console.stop_tui()
        for text in ("first question", "first answer",
                     "second question", "second answer"):
            self.assertIn(text, output)
        self.assertIn("═" if console.unicode_enabled else "-", output)

    def test_empty_core_view_shows_a_short_hint(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            output = console.stream.getvalue()
            console.stop_tui()
        self.assertIn("No conversation yet", output)

    def test_plan_is_appended_to_the_core_view(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.turn("You", "do the work")
            console.plan_box(["1. inspect", "2. fix"])
            console.turn("Assistant", "working on it")
            last_screen = "\n".join(
                ui._ANSI_RE.sub("", line)
                for line in console._last_frame_lines)
            console.stop_tui()
        self.assertIn("Plan", last_screen)
        self.assertIn("1. inspect", last_screen)
        self.assertIn("working on it", last_screen)

    def test_load_history_populates_the_core_view(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.load_history([("You", "old question"),
                                  ("Assistant", "old answer")])
            output = console.stream.getvalue()
            console.stop_tui()
        self.assertIn("old question", output)
        self.assertIn("old answer", output)

    def test_info_provider_updates_the_dashboard(self):
        console = self.console()
        console.set_info_provider(lambda: "tools 4 · background 1 running")
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((70, 20))):
            console.start_tui()
            output = console.stream.getvalue()
            console.stop_tui()
        self.assertIn("tools 4 · background 1 running", output)

    def test_escape_dismisses_a_command_overlay(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.rule("help")
            console.emit("HELP BODY")
            console.dismiss_overlay()
            last_screen = "\n".join(
                ui._ANSI_RE.sub("", line)
                for line in console._last_frame_lines)
            console.stop_tui()
        self.assertNotIn("HELP BODY", last_screen)

    def test_thinking_spinner_animates_and_stops(self):
        console = self.console()
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.start_thinking("Thinking")
            time.sleep(0.25)
            self.assertTrue(console._status_line)
            self.assertTrue(console._thinking_thread.is_alive())
            console.stop_thinking()
            self.assertEqual(console._status_line, "")
            console.stop_tui()

    def test_question_uses_a_yellow_answer_box(self):
        console = self.console(color=True)
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.question_box("Question to User", ["1) Resume"])
            output = console.stream.getvalue()
            self.assertTrue(console._question_active)
            self.assertIn("Answer", output)
            self.assertIn("Question to User", output)
            self.assertIn("33m", output)
            console.stop_tui()


class TuiConsolePtyTest(unittest.TestCase):
    def test_scroll_keys_work_while_the_agent_is_busy(self):
        child = r"""
import sys, time
from mini_code_cli.tui import TuiConsole
c = TuiConsole(stream=sys.stdout, color=False, unicode=False)
c.start_tui()
c.turn("Assistant", "\n".join("line %d" % i for i in range(60)))
c.render()
sys.stderr.write("READY\n")
sys.stderr.flush()
time.sleep(1.0)
offset = c._scroll_offset
c.stop_tui()
sys.stdout.write("\nRESULT:%d\n" % offset)
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            time.sleep(0.3)
            os.write(master, b"\x1b[1;2A")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:1", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_ctrl_c_during_llm_exits_gracefully(self):
        child = r"""
import argparse, sys, tempfile, time
from unittest import mock
from mini_code_cli import agent_loop, tui, ui, utils

args = argparse.Namespace(
    url="http://localhost:1", max_tokens=None, temperature=1.0,
    model="test-model", api_key=None, cache_dir=tempfile.mkdtemp(),
    restore_session=None, load_last_session=False, system_prompt=None,
    auto_mode=True, agent_md=None, prompt=None, ask_permission=False,
    sudo=False, allowed_dir=tempfile.mkdtemp(), enable_shell=False,
    enable_subagents=False, skills=False, llm_timeout=30, tool_timeout=30,
    max_context_tokens=None, plan=False, verbose=False, scratchpad=False,
    background=False, compaction_strategy="summarise", plan_strategy=None,
    tui="on")

def slow(messages, *a, **k):
    time.sleep(30)
    return ("never", [], None, None)

with mock.patch.object(utils, "call_openai_server", new=slow), \
        mock.patch.object(utils, "determine_vllm_model", return_value=None):
    console = tui.TuiConsole(stream=sys.stdout, color=False, unicode=False)
    session = agent_loop.AgentSession(args, console=console)
    sys.stderr.write("READY\n")
    sys.stderr.flush()
    session.run()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            time.sleep(0.5)
            os.write(master, b"hello\r")
            time.sleep(0.5)
            proc.send_signal(signal.SIGINT)
            deadline = time.time() + 10
            while proc.poll() is None and time.time() < deadline:
                drain(deadline)
            self.assertIsNotNone(proc.poll(), "session did not exit on Ctrl+C")
            drain(time.time() + 1)
            self.assertIn(b"Interrupting", output)
            self.assertIn(b"\x1b[?1049l", output)
            # The cancellation still ends like a normal session: save note and
            # good-bye box on the restored screen.
            self.assertIn(b"Saved session to", output)
            self.assertIn(b"goodbye", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_full_session_turn_and_quit_in_a_pty(self):
        child = r"""
import argparse, sys, tempfile
from unittest import mock
from mini_code_cli import agent_loop, tui, ui, utils

args = argparse.Namespace(
    url="http://localhost:1", max_tokens=None, temperature=1.0,
    model="test-model", api_key=None, cache_dir=tempfile.mkdtemp(),
    restore_session=None, load_last_session=False, system_prompt=None,
    auto_mode=True, agent_md=None, prompt=None, ask_permission=False,
    sudo=False, allowed_dir=tempfile.mkdtemp(), enable_shell=False,
    enable_subagents=False, skills=False, llm_timeout=30, tool_timeout=30,
    max_context_tokens=None, plan=False, verbose=False, scratchpad=False,
    background=False, compaction_strategy="summarise", plan_strategy=None,
    tui="on")

def fake(messages, *a, **k):
    return ("Hello from stub.", [], {"prompt_tokens": 1,
                                     "completion_tokens": 1,
                                     "total_tokens": 2,
                                     "tool_calls": 0}, None)

with mock.patch.object(utils, "call_openai_server", new=fake), \
        mock.patch.object(utils, "determine_vllm_model", return_value=None):
    console = tui.TuiConsole(stream=sys.stdout, color=False, unicode=False)
    session = agent_loop.AgentSession(args, console=console)
    sys.stderr.write("READY\n")
    sys.stderr.flush()
    session.run()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"READY", output)
            time.sleep(0.5)
            os.write(master, b"hello\r")
            time.sleep(0.8)
            os.write(master, b"/quit\r")
            deadline = time.time() + 10
            while proc.poll() is None and time.time() < deadline:
                drain(deadline)
            self.assertIsNotNone(proc.poll(), "session did not exit")
            drain(time.time() + 1)
            self.assertIn(b"Hello from", output)
            self.assertIn(b"\x1b[?1049h", output)
            self.assertIn(b"\x1b[?1049l", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_kitty_shift_enter_and_ctrl_c_sequences(self):
        child = r"""
import sys
from mini_code_cli.tui import TuiConsole
c = TuiConsole(stream=sys.stdout, color=False, unicode=False)
c.start_tui()
sys.stderr.write("READY\n")
sys.stderr.flush()
first = c.read_line("")
second = c.read_line("")
c.stop_tui()
sys.stdout.write("\nRESULT:%r|%r\n" % (first, second))
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            time.sleep(0.3)
            os.write(master, b"a\x1b[13;2ub\r")
            time.sleep(0.25)
            os.write(master, b"\x1b[99;5u")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'a\\nb'|None", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_question_box_disappears_after_the_answer(self):
        child = r"""
import sys
from mini_code_cli.tui import TuiConsole
c = TuiConsole(stream=sys.stdout, color=False, unicode=False)
c.start_tui()
c.question_box("Question to User", ["Continue?"])
sys.stderr.write("READY\n")
sys.stderr.flush()
answer = c.read_line("")
active = c._question_active
c.stop_tui()
sys.stdout.write("\nRESULT:%r|%r\n" % (answer, active))
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            time.sleep(0.3)
            os.write(master, b"yes\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'yes'|False", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_shift_enter_multiline_and_shift_arrow_scroll(self):
        child = r"""
import sys
from mini_code_cli.tui import TuiConsole
c = TuiConsole(stream=sys.stdout, color=False, unicode=False)
c.start_tui()
c.set_summary(["test"])
sys.stderr.write("READY\n")
sys.stderr.flush()
first = c.read_line("")
c.turn("Assistant", "\n".join("line %d" % i for i in range(40)))
second = c.read_line("")
offset = c._scroll_offset
c.stop_tui()
sys.stdout.write("\nRESULT:%r|%r|%d\n" % (first, second, offset))
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"READY", output)
            time.sleep(0.3)
            os.write(master, b"line1\x1b[27;2;13~line2\r")
            time.sleep(0.25)
            os.write(master, b"\x1b[1;2A\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'line1\\nline2'|''|", output)
            marker = output.split(b"RESULT:", 1)[1]
            self.assertNotEqual(marker.rsplit(b"|", 1)[-1].strip(), b"0")
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_tui_console_history_in_a_real_pty(self):
        child = r"""
import sys
from mini_code_cli.tui import TuiConsole
c = TuiConsole(stream=sys.stdout, color=False, unicode=False)
c.start_tui()
c.set_summary(["test"])
sys.stderr.write("READY\n")
sys.stderr.flush()
first = c.read_line("")
second = c.read_line("")
c.stop_tui()
sys.stdout.write("\nRESULT:%r|%r\n" % (first, second))
sys.stdout.flush()
"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master, slave = pty.openpty()
        proc = subprocess.Popen([sys.executable, "-c", child], stdin=slave,
                                stdout=slave, stderr=slave, cwd=root,
                                close_fds=True)
        os.close(slave)
        output = b""

        def drain(deadline):
            nonlocal output
            while time.time() < deadline:
                ready, _w, _x = select.select([master], [], [], 0.1)
                if ready:
                    try:
                        chunk = os.read(master, 4096)
                    except OSError:
                        return False
                    if not chunk:
                        return False
                    output += chunk
                    return True
            return False

        try:
            deadline = time.time() + 10
            while b"READY" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"READY", output)
            time.sleep(0.3)
            os.write(master, b"hello\r")
            time.sleep(0.2)
            os.write(master, b"\x1b[A\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'hello'|'hello'", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)


class TuiClearTest(unittest.TestCase):
    def test_clear_empties_the_middle_pane_without_reprinting_the_last_turn(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            console = tui.TuiConsole(stream=_FakeTty(), color=False,
                                     unicode=False)
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args, console=console)
            session.start_new_session()
            with mock.patch.object(
                    tui.shutil, "get_terminal_size",
                    return_value=os.terminal_size((60, 20))):
                console.start_tui()
                session.messages += [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
                session._clear_and_show_last()
                console.stop_tui()
            self.assertEqual(console._last_user, "")
            self.assertEqual(console._last_assistant, "")


class TuiDashboardLayoutTest(unittest.TestCase):
    def test_settings_have_labels_and_vertical_separators(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            console = tui.TuiConsole(stream=_FakeTty(), color=False,
                                     unicode=False)
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args, console=console)
            header = "\n".join(session._tui_header_lines())

        for label in ("version:", "session:", "permissions:", "shell:",
                      "compaction:", "planning:"):
            self.assertIn(label, header)
        # ASCII console: the unicode vertical bar falls back to |.
        self.assertIn(" | ", header)
        self.assertNotIn("compact summarise", header)


class TuiHeaderRefreshTest(unittest.TestCase):
    def test_context_bar_updates_after_a_recorded_turn(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            console = tui.TuiConsole(stream=_FakeTty(), color=False,
                                     unicode=False)
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args, console=console)
            session.start_new_session()
            with mock.patch.object(
                    tui.shutil, "get_terminal_size",
                    return_value=os.terminal_size((80, 24))), \
                    mock.patch.object(tui.TuiConsole,
                                      "start_scroll_listener"):
                console.start_tui()
                console.set_summary(session._tui_header_lines())
                before = "\n".join(console._summary_lines)

                session.messages.append(
                    {"role": "user", "content": "x" * 20_000})
                session._record_turn("user", session.messages)
                after = "\n".join(console._summary_lines)
                console.stop_tui()

        self.assertIn("context:", before)
        self.assertIn("context:", after)
        self.assertNotEqual(
            next(line for line in before.splitlines()
                 if line.startswith("context:")),
            next(line for line in after.splitlines()
                 if line.startswith("context:")),
        )


class TuiInfoAndColorTest(unittest.TestCase):
    def test_an_untitled_rule_does_not_clear_command_output(self):
        console = tui.TuiConsole(stream=_FakeTty(), color=False, unicode=False)
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.rule("help")
            console.emit("HELP BODY")
            console.rule(char="=")          # closing separator
            last_screen = "\n".join(
                ui._ANSI_RE.sub("", line)
                for line in console._last_frame_lines)
            console.stop_tui()
        self.assertIn("HELP BODY", last_screen)

    def test_warnings_are_rendered_with_status_colour_and_icon(self):
        console = tui.TuiConsole(stream=_FakeTty(), color=True, unicode=True)
        with mock.patch.object(
                tui.shutil, "get_terminal_size",
                return_value=os.terminal_size((70, 20))):
            console.start_tui()
            console.note("something needs attention", "warn")
            output = console.stream.getvalue()
            console.stop_tui()
        self.assertIn("something needs attention", output)
        self.assertIn("\x1b[", output)


class TuiExitCleanupTest(unittest.TestCase):
    def test_system_exit_at_resume_restores_the_alternate_screen(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            console = tui.TuiConsole(stream=_FakeTty(), color=False,
                                     unicode=False)
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args, console=console)
            with mock.patch.object(AgentSession, "start_session",
                                   side_effect=SystemExit):
                with self.assertRaises(SystemExit):
                    session.run()
            output = console.stream.getvalue()
            self.assertIn("\x1b[?1049h", output)
            self.assertIn("\x1b[?1049l", output)


class TuiWidthAndDocstringTest(unittest.TestCase):
    def test_visual_width_handles_wide_and_combining_characters(self):
        self.assertEqual(ui._visual_len("aa"), 2)
        self.assertEqual(ui._visual_len("\u6f22"), 2)
        self.assertEqual(ui._visual_len("e\u0301"), 1)
        self.assertEqual(ui._visual_len("\U0001f642"), 2)
        self.assertEqual(ui._visual_len("\x1b[31mred\x1b[0m"), 3)

    def test_tool_call_display_keeps_its_docstring(self):
        self.assertIsNotNone(ui.Console.tool_call_display.__doc__)
        self.assertIsNotNone(tui.TuiConsole.tool_call_display.__doc__)


class TuiRestoredHistoryTest(unittest.TestCase):
    def test_restored_header_and_history_turns(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            console = tui.TuiConsole(stream=_FakeTty(), color=False,
                                     unicode=False)
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args, console=console)
            session.start_new_session()
            session.restored_session = True
            session.messages += [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "Current plan for the goal (from CURRENT_PLAN.md):\n1. x"},
            ]
            header = "\n".join(session._tui_header_lines())
            self.assertIn("[restored]", header)
            self.assertEqual(session._tui_history_turns(),
                             [("user", "old question"),
                              ("assistant", "old answer")])


class TuiSessionWiringTest(unittest.TestCase):
    def test_session_uses_tui_console_when_flag_is_on(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args)
            self.assertIsInstance(session.console, tui.TuiConsole)


class TuiExpandCommandTest(unittest.TestCase):
    def test_expand_command_reaches_the_console(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            console = ui.Console(stream=io.StringIO(), color=False,
                                 unicode=False, verbose=False)
            session = AgentSession(make_args(work, cache), console=console)
            session.start_new_session()
            console.tool_call_display(
                "read_file", {"file_path": "x"}, "tool result body")
            self.assertTrue(session._handle_command("/expand"))
            self.assertIn("tool result body",
                          console.stream.getvalue())


class TuiTipTest(unittest.TestCase):
    def test_startup_tip_advertises_tui_keys(self):
        class Session:
            plan_mode = False
            tui = True

            def __init__(self):
                self.console = ui.Console(stream=io.StringIO(), color=False,
                                          unicode=False, verbose=False)

            _emit_tip = agent_loop.AgentSession._emit_tip

        session = Session()
        session._emit_tip()
        out = session.console.stream.getvalue()
        self.assertIn("TUI:", out)
        self.assertIn("↑/↓ history", out)
