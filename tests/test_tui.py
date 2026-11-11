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


class TuiFrameTest(unittest.TestCase):
    def test_fixed_frame_draws_a_header_footer_and_alternate_screen(self):
        stream = _FakeTty()
        console = ui.Console(stream=stream, color=False, unicode=False)
        self.assertTrue(console.tui_enter(header_height=6, footer_height=3))
        console.tui_set_header(["HEADER ONE", "HEADER TWO"])
        console.emit("a transcript line")
        console.tui_begin_input("HISTORY HINT")
        console.tui_end_input()
        console.clear_screen()
        console.tui_leave()
        output = stream.getvalue()

        self.assertIn("\x1b[?1049h", output)
        self.assertIn("HEADER ONE", output)
        self.assertIn("a transcript line", output)
        self.assertIn("HISTORY HINT", output)
        self.assertIn("\x1b[?1049l", output)
        self.assertIsNone(console.tui_frame)

    def test_frame_is_a_noop_on_a_non_tty(self):
        console = ui.Console(stream=io.StringIO(), color=False, unicode=False)
        self.assertFalse(console.tui_enter())
        self.assertIsNone(console.tui_frame)
        console.tui_set_header(["ignored"])
        console.tui_begin_input()
        console.tui_end_input()
        console.tui_leave()

    def test_expand_tool_reprints_recent_tool_details(self):
        stream = io.StringIO()
        console = ui.Console(stream=stream, color=False, unicode=False)
        console.tool_call_display(
            "read_file", {"file_path": "sample.txt"}, "file body")
        console.expand_tool(1)
        output = stream.getvalue()
        self.assertIn("tool call 1 back", output)
        self.assertIn("read_file", output)
        self.assertIn("file body", output)


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
