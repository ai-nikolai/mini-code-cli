"""Tests for the interactive "resume last session" offer.

When a previous session exists (a recorded ``.mini`` pointer, or the most
recent session in the cache dir) but the user did *not* explicitly ask to
resume one, the agent offers a choice before starting the loop:

  1) Resume  -- splice the previous session's history into the current one.
  2) New     -- start a fresh (empty) session.
  3) Exit    -- stop before anything is written or the loop enters.

The tests drive the mocked console through a couple of turns and verify each
choice, plus that a session with no previous session never shows the offer.
"""
import os
import sys
import io
import json
import types
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mini_code_cli.agent_loop as agent_loop
from mini_code_cli.agent_loop import AgentSession
from mini_code_cli import ui, persistence


def _make_args(**kw):
    a = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None, api_key=None,
        system_prompt=None, agent_md=None, prompt=None, enable_shell=False,
        ask_permission=False, auto_mode=False, restore_session=None, llm_timeout=30,
        tool_timeout=30, load_last_session=False, verbose=False, max_tokens=None,
        temperature=1.0, plan=False, enable_subagents=False, enable_skills=False,
        sudo=False,
    )
    a.__dict__.update(kw)
    return a


class TestResumeOffer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.allowed = self.tmp.name
        self.cache = os.path.join(self.tmp.name, "cache")
        os.makedirs(self.cache)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)

    def _prev_session(self, content):
        """Create a previous session with the given messages, and mark it the
        last session via the .mini pointer."""
        d = os.path.join(self.cache, "prev")
        os.makedirs(d)
        with open(os.path.join(d, "history.json"), "w") as f:
            json.dump(content, f)
        persistence.save_last_session_pointer(self.allowed, d)
        return d

    def _fake_empty_llm(self):
        def fake(messages, **kw):
            # No answer, no tool calls -> the manual-mode loop just re-prompts.
            return "", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return fake

    def _run_until_quit(self, args, answers):
        """Build a session and run() it, feeding `answers` to the console's
        read_line (resume choice, then /quit), with the network calls mocked.

        Extra ``/quit`` lines are appended so the mock is never exhausted: the
        resume offer consumes the first answer, and the loop keeps asking until
        it sees ``/quit``.
        """
        session = AgentSession(args, console=self.console)
        answers = list(answers) + ["/quit"] * 5
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               self._fake_empty_llm()), \
             mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None), \
             mock.patch.object(self.console, "read_line", side_effect=iter(answers)):
            session.run()
        return session

    def test_no_previous_session_does_not_offer(self):
        # No previous session at all -> the offer is a no-op and run() just
        # enters the loop (which we end via a /quit read_line).
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        session = self._run_until_quit(args, ["/quit"])
        self.assertNotIn("A previous session was found", self.buf.getvalue())
        # No restore happened.
        self.assertIsNone(session.loaded_last_session_dir)

    def test_offer_prompt_appears_for_previous_session(self):
        self._prev_session([{"role": "system", "content": "sys"},
                            {"role": "user", "content": "remember 42"}])
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        session = self._run_until_quit(args, ["1", "2"])
        out = self.buf.getvalue()
        self.assertIn("A previous session was found", out)
        self.assertIn("Resume", out)

    def test_resume_choice_splices_history_and_announces(self):
        prev = self._prev_session([{"role": "system", "content": "sys"},
                                   {"role": "user", "content": "remember 42"}])
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        session = self._run_until_quit(args, ["1"])
        # The previous conversation was spliced into the restored session.
        joined = "\n".join(m.get("content", "") for m in session.messages)
        self.assertIn("remember 42", joined)
        # run() advertised that it was continuing the previous session.
        self.assertIn("continuing", self.buf.getvalue())
        # The candidate was consumed (no leftover restore pending).
        self.assertIsNone(session.last_session_candidate)
        self.assertEqual(session.loaded_last_session_dir, os.path.abspath(prev))

    def test_new_choice_starts_fresh(self):
        self._prev_session([{"role": "system", "content": "sys"},
                            {"role": "user", "content": "remember 42"}])
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        session = self._run_until_quit(args, ["2"])
        # Only the freshly-built system message remains; the old history is not
        # spliced in.
        contents = [m.get("content", "") for m in session.messages]
        self.assertNotIn("remember 42", contents)
        self.assertIsNone(session.loaded_last_session_dir)

    def test_exit_choice_stops_before_loop(self):
        self._prev_session([{"role": "system", "content": "sys"},
                            {"role": "user", "content": "remember 42"}])
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        # The LLM must never be called on the exit path. Track whether it is.
        state = {"called": False}

        def boom(*a, **k):
            state["called"] = True
            return "", [], {}

        session = AgentSession(args, console=self.console)
        with mock.patch.object(agent_loop.utils, "call_openai_server", boom), \
             mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None), \
             mock.patch.object(self.console, "read_line", side_effect=iter(["3"])):
            session.run()  # exits cleanly via the offer
        # The LLM was never reached; the candidate was consumed.
        self.assertFalse(state["called"])
        self.assertIsNone(session.last_session_candidate)

    def test_bad_input_reprompts_then_new(self):
        self._prev_session([{"role": "system", "content": "sys"},
                            {"role": "user", "content": "remember 42"}])
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        session = self._run_until_quit(args, ["banana", "2"])
        out = self.buf.getvalue()
        self.assertIn("Please choose 1, 2, or 3", out)

    def test_exit_via_eof(self):
        # EOF / Ctrl-C at the offer (read_line returns None) is a graceful exit.
        self._prev_session([{"role": "system", "content": "sys"},
                            {"role": "user", "content": "remember 42"}])
        args = _make_args(cache_dir=self.cache, allowed_dir=self.allowed)
        state = {"called": False}

        def boom(*a, **k):
            state["called"] = True
            return "", [], {}

        session = AgentSession(args, console=self.console)
        with mock.patch.object(agent_loop.utils, "call_openai_server", boom), \
             mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None), \
             mock.patch.object(self.console, "read_line", return_value=None):
            session.run()
        self.assertFalse(state["called"])
        self.assertIsNone(session.last_session_candidate)


if __name__ == "__main__":
    unittest.main()
