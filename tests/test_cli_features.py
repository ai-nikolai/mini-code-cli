"""Tests for the four CLI features on ``feat/mini-code-cli-plan-multiline``.

* the generated plan is printed right after planning;
* ``/skip-plan PROMPT`` sends one prompt without authoring or reviewing a plan;
* the system prompt asks for short regular answers, not only tool calls;
* multi-line input falls back to a plain line on a non-TTY stdin and is
  advertised in the startup tip / help text.
"""

import io
import os
import pty
import select
import subprocess
import sys
import time
import unittest
from unittest import mock

from mini_code_cli import prompts, ui, utils
from mini_code_cli.agent_loop import AgentSession

from test_e2e_agent_loop import (
    USAGE, StubLLM, make_args, run_loop, tool_call,
)
from test_interactive import _SessionCase


class PlanDisplayAndSkipPlanTest(_SessionCase):
    def test_generated_plan_is_printed_without_asking_for_it(self):
        console = self.console()
        args = make_args(self.workdir, self.cache_dir, plan=True)
        session = AgentSession(args, console=console)
        target = os.path.join(self.workdir, "data.txt")
        with open(target, "w") as handle:
            handle.write("alpha\nbeta\n")

        stub = StubLLM([
            ("1. read data.txt\n2. summarise the lines", [], USAGE),  # plan gen
            ("", [tool_call("c1", "read_file", {"file_path": target})], USAGE),
            ("", [], USAGE),                                          # settle -> review
            ("ALL TASKS DONE. Go back to user.", [], USAGE),          # plan review
            ("Summary: alpha and beta.", [], USAGE),                  # follow-up
        ])
        run_loop(session, "read the file", stub)

        out = self.output()
        self.assertIn("Plan for the current goal", out)
        self.assertIn("read data.txt", out)
        self.assertIn("summarise the lines", out)

    def test_skip_plan_runs_one_turn_without_planning_or_review(self):
        session = self.build_session(plan=True)
        self.assertTrue(session.plan_mode)

        stub = StubLLM([("Short answer without a plan.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            self.assertTrue(session._handle_command(
                "/skip-plan answer this in one line"))

        # A planned turn would need plan-gen + step + review + follow-up; the
        # skipped turn is a single ordinary model call.
        self.assertEqual(len(stub.calls), 1)
        self.assertEqual(session.messages[-1]["content"],
                         "Short answer without a plan.")
        self.assertEqual(session.messages[-2]["content"],
                         "answer this in one line")
        # Plan mode itself is not turned off for later turns.
        self.assertTrue(session.plan_mode)

    def test_skip_plan_without_a_prompt_prints_usage(self):
        session = self.build_session(plan=True)
        self.assertTrue(session._handle_command("/skip-plan"))
        self.assertIn("Usage: /skip-plan", self.output())

    def test_skip_plan_works_when_plan_mode_is_off(self):
        session = self.build_session()
        stub = StubLLM([("Plain answer.", [], USAGE)])
        with mock.patch.object(utils, "call_openai_server", new=stub):
            self.assertTrue(session._handle_command("/skip-plan hello"))
        self.assertEqual(len(stub.calls), 1)
        self.assertIn("Plan mode is not enabled", self.output())

    def test_startup_tip_advertises_multiline_and_skip_plan(self):
        session = self.build_session(plan=True)
        session._emit_tip()
        out = self.output()
        self.assertIn("Shift+Enter", out)
        self.assertIn("/skip-plan", out)


class SystemPromptBrevityTest(unittest.TestCase):
    def test_tool_prompt_asks_for_short_answers_between_tool_calls(self):
        text = prompts.system_prompt({"read_file": {"description": "read"}})
        self.assertIn("When you are not calling a tool", text)
        self.assertIn("short, direct answer", text)

    def test_no_tools_prompt_asks_for_brief_answers(self):
        self.assertIn("Answer briefly and directly", prompts.system_prompt())


class MultilineInputTest(unittest.TestCase):
    def test_non_tty_stdin_falls_back_to_a_plain_line(self):
        console = ui.Console(stream=io.StringIO(), color=False, unicode=False,
                             verbose=False)
        with mock.patch.object(sys, "stdin", io.StringIO("hello\nworld\n")):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(console.read_line("You: ", multiline=True),
                                 "hello")
                self.assertEqual(console.read_line("You: ", multiline=True),
                                 "world")

    def test_tty_editor_enter_submits_and_shift_enter_inserts_a_newline(self):
        """Drive the real termios reader through a pseudo-terminal."""
        child = r"""
import sys
from mini_code_cli.ui import Console
console = Console(stream=sys.stdout, color=False, unicode=False)
sys.stdout.write("READY\n")
sys.stdout.flush()
first = console.read_line("", multiline=True)
second = console.read_line("", multiline=True)
sys.stdout.write("\nRESULT:%r,%r\n" % (first, second))
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
            # Let the child enter raw mode: anything sent while the pty is
            # still canonical would have its CR translated to NL.
            time.sleep(0.3)
            # Shift+Enter (CSI-u) splits the first turn; plain Enter submits.
            os.write(master, b"line1\x1b[13;2uline2\rafter\r")
            deadline = time.time() + 10
            while b"RESULT:" not in output and time.time() < deadline:
                drain(deadline)
            self.assertIn(b"RESULT:'line1\\nline2','after'", output)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
            os.close(master)

    def test_multiline_flag_is_forwarded_through_a_swallowed_console(self):
        parent = ui.Console(stream=io.StringIO(), color=False, unicode=False,
                            verbose=False)
        child = ui.Console(stream=io.StringIO(), color=False, unicode=False,
                           verbose=False, parent_console=parent)
        with mock.patch.object(sys, "stdin", io.StringIO("forwarded\n")):
            with mock.patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(child.read_line("You: ", multiline=True),
                                 "forwarded")
