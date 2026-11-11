"""The opt-in ``--tui`` prompt: history, editing and multiline input."""

import io
import os
import pty
import select
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from mini_code_cli import agent_loop, ui
from mini_code_cli.agent_loop import AgentSession

from test_e2e_agent_loop import make_args


class TuiCliTest(unittest.TestCase):
    def test_tui_is_off_by_default_and_not_part_of_all(self):
        with mock.patch.object(sys, "argv", ["mini-code"]):
            self.assertFalse(agent_loop.parse_args().tui)
        with mock.patch.object(sys, "argv", ["mini-code", "--tui"]):
            self.assertTrue(agent_loop.parse_args().tui)
        with mock.patch.object(sys, "argv", ["mini-code", "--all"]):
            self.assertFalse(agent_loop.parse_args().tui)


class TuiPtyTest(unittest.TestCase):
    def test_history_editing_and_shift_enter_in_a_real_pty(self):
        """Drive the real termios editor through a pseudo-terminal."""
        child = r"""
import sys
from mini_code_cli.ui import Console
console = Console(stream=sys.stdout, color=False, unicode=False)
sys.stdout.write("READY\n")
sys.stdout.flush()
first = console.read_line("", multiline=True, tui=True)
second = console.read_line("", multiline=True, tui=True)
third = console.read_line("", multiline=True, tui=True)
fourth = console.read_line("", multiline=True, tui=True)
sys.stdout.write("\nRESULT:%r|%r|%r|%r\n"
                 % (first, second, third, fourth))
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
            # Enter sends; Up recalls the previous history entry; Left + X
            # edits in the middle; Shift+Enter inserts a newline.
            os.write(master, b"hello\r")
            time.sleep(0.15)
            os.write(master, b"\x1b[A\r")
            time.sleep(0.15)
            os.write(master, b"abc\x1b[DX\r")
            time.sleep(0.15)
            os.write(master, b"line1\x1b[13;2uline2\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(
                b"RESULT:'hello'|'hello'|'abXc'|'line1\\nline2'", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)


class _FakeTty(io.StringIO):
    def isatty(self):
        return True


class TuiConsoleTest(unittest.TestCase):
    def console(self, color=False):
        return ui.TuiConsole(stream=_FakeTty(), color=color, unicode=False)

    def test_fixed_view_redraws_summary_body_and_input_box(self):
        console = self.console()
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
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

    def test_the_middle_pane_scrolls(self):
        console = self.console()
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
                return_value=os.terminal_size((50, 14))):
            console.start_tui()
            console.set_summary(["summary"])
            console.turn("Assistant", "\n".join(f"line {i}" for i in range(40)))
            console.render()
            self.assertEqual(console._scroll_offset, 0)
            console.scroll_body(1)
            self.assertGreater(console._scroll_offset, 0)
            self.assertIn("scrolled up", console.stream.getvalue())
            console.scroll_body(-1)
            self.assertEqual(console._scroll_offset, 0)
            console.stop_tui()

    def test_core_view_keeps_multiple_turns_with_separators(self):
        console = self.console()
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
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

    def test_info_provider_updates_the_dashboard(self):
        console = self.console()
        console.set_info_provider(lambda: "tools 4 · background 1 running")
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
                return_value=os.terminal_size((70, 20))):
            console.start_tui()
            output = console.stream.getvalue()
            console.stop_tui()
        self.assertIn("tools 4 · background 1 running", output)

    def test_escape_dismisses_a_command_overlay(self):
        console = self.console()
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.rule("help")
            console.emit("HELP BODY")
            console.dismiss_overlay()
            output = console.stream.getvalue()
            console.stop_tui()
        last_screen = output.rsplit("\x1b[H", 1)[-1]
        self.assertNotIn("HELP BODY", last_screen)

    def test_thinking_spinner_animates_and_stops(self):
        console = self.console()
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
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
                ui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.question_box("Resume previous session?", ["1) Resume"])
            output = console.stream.getvalue()
            self.assertTrue(console._question_active)
            self.assertIn("Answer", output)
            self.assertIn("Resume previous session?", output)
            self.assertIn("\x1b[33m", output)
            console.stop_tui()


class TuiConsolePtyTest(unittest.TestCase):
    def test_shift_enter_multiline_and_shift_arrow_scroll(self):
        child = r"""
import sys
from mini_code_cli.ui import TuiConsole
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
            os.write(master, b"line1\x1b[13;2uline2\r")
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
from mini_code_cli.ui import TuiConsole
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
            console = ui.TuiConsole(stream=_FakeTty(), color=False,
                                     unicode=False)
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args, console=console)
            session.start_new_session()
            with mock.patch.object(
                    ui.shutil, "get_terminal_size",
                    return_value=os.terminal_size((60, 20))):
                session.tui_active = console.start_tui()
                session.messages += [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
                session._clear_and_show_last()
                console.stop_tui()
            self.assertEqual(console._last_user, "")
            self.assertEqual(console._last_assistant, "")


class TuiInfoAndColorTest(unittest.TestCase):
    def test_an_untitled_rule_does_not_clear_command_output(self):
        console = ui.TuiConsole(stream=_FakeTty(), color=False, unicode=False)
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
                return_value=os.terminal_size((60, 20))):
            console.start_tui()
            console.rule("help")
            console.emit("HELP BODY")
            console.rule(char="=")          # closing separator
            output = console.stream.getvalue()
            console.stop_tui()
        last_screen = output.rsplit("\x1b[H", 1)[-1]
        self.assertIn("HELP BODY", last_screen)

    def test_warnings_are_rendered_with_status_colour_and_icon(self):
        console = ui.TuiConsole(stream=_FakeTty(), color=True, unicode=True)
        with mock.patch.object(
                ui.shutil, "get_terminal_size",
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
            console = ui.TuiConsole(stream=_FakeTty(), color=False,
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
        self.assertIsNotNone(ui.TuiConsole.tool_call_display.__doc__)


class TuiSessionWiringTest(unittest.TestCase):
    def test_session_uses_tui_console_when_flag_is_on(self):
        with tempfile.TemporaryDirectory() as work, \
                tempfile.TemporaryDirectory() as cache:
            args = make_args(work, cache)
            args.tui = True
            session = AgentSession(args)
            self.assertIsInstance(session.console, ui.TuiConsole)


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
