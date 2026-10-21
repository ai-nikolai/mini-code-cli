"""Tests for sub-agent delegation, and for the *unattended* guarantee.

The point of this module is one property: **a sub-agent must never need a human.**
A delegated session is started in *background mode*, which means it

  * never consults the ``.mini`` resume pointer and never shows the resume
    prompt (so it cannot block waiting for a choice at startup),
  * never reads a "You:" line (nothing to type, no EOF loop), and
  * auto-denies any tool call that would otherwise ask for permission, rather
    than forwarding the prompt to a console nobody is watching.

Those three paths are each pinned below by patching the input functions to
*raise*: if any of them is reached, the test fails loudly instead of hanging.

The remaining tests cover the delegation contract around that: the ``max_steps``
cap, degradation to an ``Error: ...`` tool result, and the ``report.md``
hand-off (including its exemption from the ``--allowed-dir`` sandbox).

Everything still runs against the real loop, the real tools and the real
sandbox; only the LLM is stubbed.
"""

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from mini_code_cli import ui, utils
from mini_code_cli.agent_loop import AgentSession
from mini_code_cli.subagents import SubAgent, clear_registry, register
from mini_code_cli.persistence import save_history, save_summary

from test_e2e_agent_loop import USAGE, StubLLM, make_args, run_loop, silent_console, tool_call


class SubAgentTestCase(unittest.TestCase):
    """Shared tmpdirs + helpers for driving one delegated task."""

    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        clear_registry()
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def make_session(self, **overrides):
        args = make_args(self.workdir, self.cache_dir, enable_subagents=True)
        for key, value in overrides.items():
            setattr(args, key, value)
        return AgentSession(args, console=silent_console())

    def tool_messages(self, session):
        return [m for m in session.messages if m.get("role") == "tool"]

    def assert_no_prompting(self):
        """Patch every human-input entry point to explode if it is called.

        Returned as a context manager list; use via :meth:`start_no_prompt_hooks`.
        """
        return [
            mock.patch.object(AgentSession, "_read_user_line",
                              side_effect=AssertionError("a sub-agent asked for input")),
            mock.patch.object(AgentSession, "show_resume_choice",
                              side_effect=AssertionError("a sub-agent offered the resume prompt")),
            mock.patch.object(ui.Console, "ask_permission",
                              side_effect=AssertionError("a sub-agent asked for permission")),
        ]


# ---------------------------------------------------------------------------
# the unattended guarantee
# ---------------------------------------------------------------------------

class NoUserInputTest(SubAgentTestCase):
    """A delegated sub-agent never reaches any human-input code path."""

    class EchoSubAgent(SubAgent):
        name = "echo_bg"
        description = "Writes a file, then finishes."
        tools = ["write_file"]
        max_steps = 5

    def test_startup_and_run_never_prompt(self):
        marker = os.path.join(self.workdir, "bg.txt")
        register(self.EchoSubAgent())

        # A stale .mini pointer exists: a naive child would find it and offer to
        # resume. background_mode must ignore it entirely.
        pointed = os.path.join(self.cache_dir, "old_session")
        os.makedirs(pointed, exist_ok=True)
        save_history([{"role": "system", "content": "OLD"},
                      {"role": "user", "content": "old question"}],
                     self.cache_dir, "old_session", 0)
        os.makedirs(os.path.join(self.workdir, ".mini"), exist_ok=True)
        with open(os.path.join(self.workdir, ".mini", "last_session.json"), "w") as f:
            json.dump({"session_dir": pointed}, f)

        stub = StubLLM([
            ("", [tool_call("p1", "echo_bg", {"task": f"Write {marker}"})], USAGE),
            ("", [tool_call("c1", "write_file", {"file_path": marker, "content": "x"})], USAGE),
            ("child done", [], USAGE),
            ("parent done", [], USAGE),
        ])

        session = self.make_session()
        patches = self.assert_no_prompting()
        for p in patches:
            p.start()
        try:
            stub = run_loop(session, "delegate please", stub)
        finally:
            for p in patches:
                p.stop()

        # Nothing raised: no input path was hit. The child really did the work.
        self.assertTrue(os.path.isfile(marker))
        self.assertEqual(len(stub.calls), 4)      # 2 parent + 2 child
        self.assertEqual(session._last_assistant_text(), "parent done")

        # The child ran in its OWN fresh session, not the pointed-at one.
        child_root = os.path.join(self.cache_dir, session.unique_id, "subagents", "echo_bg")
        child_ids = [d for d in os.listdir(child_root)
                     if os.path.isdir(os.path.join(child_root, d))]
        self.assertEqual(len(child_ids), 1)
        self.assertNotEqual(child_ids[0], "old_session")

    def test_background_session_skips_pointer_and_prompt(self):
        """start_session() on a background session ignores the .mini pointer."""
        pointed = os.path.join(self.cache_dir, "old_session")
        os.makedirs(pointed, exist_ok=True)
        save_history([{"role": "system", "content": "OLD"}], self.cache_dir, "old_session", 0)
        os.makedirs(os.path.join(self.workdir, ".mini"), exist_ok=True)
        with open(os.path.join(self.workdir, ".mini", "last_session.json"), "w") as f:
            json.dump({"session_dir": pointed}, f)

        args = make_args(self.workdir, self.cache_dir)
        background = AgentSession(args, console=silent_console(), background_mode=True)
        with mock.patch.object(AgentSession, "show_resume_choice",
                               side_effect=AssertionError("prompted")), \
             mock.patch.object(AgentSession, "_detect_last_session",
                               side_effect=AssertionError("consulted the .mini pointer")):
            background.start_session()

        self.assertNotEqual(background.unique_id, "old_session")
        self.assertNotIn("OLD", [m.get("content") for m in background.messages])

    def test_foreground_session_still_offers_the_pointer(self):
        """Contrast: the same pointer DOES prompt when not in background mode."""
        pointed = os.path.join(self.cache_dir, "old_session")
        os.makedirs(pointed, exist_ok=True)
        save_history([{"role": "system", "content": "OLD"},
                      {"role": "user", "content": "old question"}], self.cache_dir, "old_session", 0)
        os.makedirs(os.path.join(self.workdir, ".mini"), exist_ok=True)
        with open(os.path.join(self.workdir, ".mini", "last_session.json"), "w") as f:
            json.dump({"session_dir": pointed}, f)

        args = make_args(self.workdir, self.cache_dir)
        session = AgentSession(args, console=silent_console())
        with mock.patch.object(AgentSession, "show_resume_choice",
                               return_value="resume") as prompt:
            session.start_session()

        self.assertEqual(prompt.call_count, 1)
        self.assertEqual(session.unique_id, "old_session")   # resumed in place

    def test_child_args_drop_parent_restore_flags(self):
        """The nested namespace never inherits restore/prompt-related parent state."""
        sub = self.EchoSubAgent()
        parent = self.make_session(load_last_session=True, ask_permission=True)
        parent.start_session()          # gives the parent its session dir/id
        # Even with these set on the parent's args, the child must not inherit
        # anything resumable or interactive.
        parent.args.restore_session = "/some/old/session"
        parent.restore_session = "/some/old/session"
        child = sub._build_child_args(parent, "do a thing")

        self.assertIsNone(child.restore_session)
        self.assertFalse(child.load_last_session)
        self.assertEqual(child.prompt, "do a thing")
        self.assertTrue(child.auto_mode)          # unattended by construction
        self.assertEqual(sub.background_mode, True)

    def test_background_mode_is_overridable(self):
        class Interactive(SubAgent):
            name = "interactive_one"
            background_mode = False

        self.assertFalse(Interactive().background_mode)
        self.assertTrue(Interactive(background_mode=True).background_mode)
        self.assertTrue(self.EchoSubAgent().background_mode)   # class default


# ---------------------------------------------------------------------------
# per-call permission inside a background session
# ---------------------------------------------------------------------------

class BackgroundPermissionTest(SubAgentTestCase):
    """A prompt-requiring tool call is refused, never prompted for."""

    class Writer(SubAgent):
        name = "writer_bg"
        description = "Writes a file."
        tools = ["write_file"]

    def test_permission_requiring_call_is_auto_denied(self):
        marker = os.path.join(self.workdir, "should_not_exist.txt")
        register(self.Writer())

        stub = StubLLM([
            ("", [tool_call("p1", "writer_bg", {"task": "write it"})], USAGE),
            # The child tries a write; --ask-permission (inherited) would prompt.
            ("", [tool_call("c1", "write_file", {"file_path": marker, "content": "x"})], USAGE),
            ("child gave up", [], USAGE),
            ("parent done", [], USAGE),
        ])

        session = self.make_session(ask_permission=True)

        # The refusal is recorded in the *child's* transcript (the parent only
        # receives the sub-agent's summary), so capture the child as it finishes.
        captured = {}
        original = SubAgent._build_result

        def spy(self_, child, final_text, report_path):
            captured["child"] = child
            return original(self_, child, final_text, report_path)

        with mock.patch.object(ui.Console, "ask_permission",
                               side_effect=AssertionError("prompted the user")), \
             mock.patch.object(SubAgent, "_build_result", new=spy):
            run_loop(session, "delegate", stub)

        # The write was refused, explained to the model, and nothing was created.
        self.assertFalse(os.path.exists(marker))
        child_tools = [m.get("content", "") for m in captured["child"].messages
                       if m.get("role") == "tool"]
        self.assertEqual(len(child_tools), 1)
        self.assertIn("unattended (background) session", child_tools[0])
        # The child was not blocked: it saw the refusal and finished normally.
        self.assertEqual(captured["child"].stop_reason, "finished")
        self.assertEqual(session._last_assistant_text(), "parent done")


# ---------------------------------------------------------------------------
# delegation contract: cap, failure, report hand-off
# ---------------------------------------------------------------------------

class DelegationContractTest(SubAgentTestCase):

    class Looper(SubAgent):
        name = "looper"
        description = "Never stops on its own."
        tools = ["write_file"]
        max_steps = 2

    def test_max_steps_caps_the_child_loop(self):
        register(self.Looper())
        step = lambda i: ("", [tool_call(f"s{i}", "write_file",
                                        {"file_path": os.path.join(self.workdir, f"f{i}.txt"),
                                         "content": "x"})], USAGE)
        stub = StubLLM([
            ("", [tool_call("p1", "looper", {"task": "loop forever"})], USAGE),
            step(0), step(1),                      # exactly max_steps child calls
            ("parent done", [], USAGE),
        ])

        session = self.make_session()
        stub = run_loop(session, "delegate", stub)

        # The child stopped at its cap instead of consuming the queue forever
        # (a third child step would have exhausted the stub and raised).
        self.assertEqual(len(stub.calls), 4)
        results = [m.get("content", "") for m in self.tool_messages(session)
                   if "sub-agent" in m.get("content", "")]
        self.assertEqual(len(results), 1)
        # A capped run is explained rather than reported as "no output".
        self.assertIn("max_steps", results[0])
        self.assertEqual(session._last_assistant_text(), "parent done")

    def test_child_failure_becomes_an_error_tool_result(self):
        """An exploding sub-agent degrades; the parent session keeps going."""
        class Broken(SubAgent):
            name = "broken"
            description = "Always fails."

            def run(self, task, parent, max_steps=None):
                raise RuntimeError("kaboom")

        register(Broken())
        stub = StubLLM([
            ("", [tool_call("p1", "broken", {"task": "x"})], USAGE),
            ("parent recovered", [], USAGE),
        ])

        session = self.make_session()
        run_loop(session, "delegate", stub)

        tool_msgs = self.tool_messages(session)
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("Error: sub-agent 'broken' failed: kaboom", tool_msgs[0]["content"])
        self.assertEqual(session._last_assistant_text(), "parent recovered")

    def test_report_md_is_returned_and_exempt_from_sandbox(self):
        """The child may write report.md under the cache dir, outside --allowed-dir."""
        class Reporter(SubAgent):
            name = "reporter"
            description = "Writes its report."
            tools = ["write_file"]

        register(Reporter())

        # The child's report path is decided by the sub-agent itself (it lives
        # under the parent's cache dir, i.e. outside --allowed-dir). Capture it
        # so the scripted child can write exactly there.
        written = []
        original = SubAgent._report_path

        def spy(self_, parent):
            path = original(self_, parent)
            if not written:
                written.append(path)
            return path

        write_report = lambda messages: (
            "", [tool_call("c1", "write_file",
                           {"file_path": written[0], "content": "all good"})], USAGE)

        stub = StubLLM([
            ("", [tool_call("p1", "reporter", {"task": "write the report"})], USAGE),
            write_report,                              # child writes report.md
            ("child finished", [], USAGE),             # child stops
            ("parent done", [], USAGE),                # parent stops
        ])

        session = self.make_session()
        with mock.patch.object(SubAgent, "_report_path", new=spy):
            run_loop(session, "delegate", stub)

        # The report really landed on disk, outside the allowed dir.
        self.assertTrue(written)
        self.assertTrue(os.path.isfile(written[0]))
        self.assertFalse(written[0].startswith(os.path.abspath(self.workdir)))
        with open(written[0]) as f:
            self.assertEqual(f.read().strip(), "all good")

        results = [m.get("content", "") for m in self.tool_messages(session)]
        self.assertTrue(any("Result from sub-agent 'reporter'" in r for r in results))
        self.assertTrue(any("Report (" in r for r in results))
        # Crucially the write was NOT blocked by the sandbox.
        self.assertFalse(any("not allowed outside allowed directory" in r for r in results))


if __name__ == "__main__":
    unittest.main()
