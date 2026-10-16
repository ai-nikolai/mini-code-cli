"""Regression tests for the agent-loop hardening:

  * runner.call_function_with_timeout shields the loop from a tool that raises
    (returns a friendly Error string instead of propagating), and still returns
    a distinct message on timeout.
  * utils.call_openai_server degrades to (None, None, None) on ANY failure
    (including malformed usage payloads) instead of raising.
  * an empty / blank user response just re-prompts: it does not append a user
    turn and does not call the LLM.
  * an LLM that returns no response and no tool calls leaves the loop intact
    (records a warning, re-prompts) rather than crashing.
"""
import io
import os
import sys
import types
import time
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mini_code_cli.agent_loop as agent_loop
from mini_code_cli import runner, utils, ui


# ---------------------------------------------------------------------------
# runner: a tool that raises must not crash the loop
# ---------------------------------------------------------------------------

class TestToolFailureRobustness(unittest.TestCase):
    def test_tool_that_raises_returns_error_string(self):
        def boom(**kw):
            raise RuntimeError("explode")
        with mock.patch.object(ui.console, "note") as noted:
            result = runner.call_function_with_timeout(boom, {}, "read_file", timeout_seconds=30)
        self.assertIsInstance(result, str)
        self.assertIn("Error", result)
        self.assertIn("read_file", result)
        self.assertTrue(any("raised an error" in str(c.args[0]) for c in noted.call_args_list))

    def test_timeout_still_reports_distinctly(self):
        def slow(**kw):
            time.sleep(3)
            return "never"
        result = runner.call_function_with_timeout(slow, {}, "read_file", timeout_seconds=1)
        self.assertIsInstance(result, str)
        self.assertIn("timed out", result)

    def test_normal_call_still_returns_value(self):
        def ok(**kw):
            return "fine"
        self.assertEqual(runner.call_function_with_timeout(ok, {}, "read_file"), "fine")


# ---------------------------------------------------------------------------
# utils: call_openai_server never raises
# ---------------------------------------------------------------------------

class TestLLMFailureRobustness(unittest.TestCase):
    def test_malformed_usage_is_coerced_not_raised(self):
        # A server that reports null token fields must not crash the caller's
        # accumulation logic. The success path coerces those to ints.
        self.assertEqual(utils._coerce_usage_int(None), 0)
        self.assertEqual(utils._coerce_usage_int("not-a-number"), 0)
        self.assertEqual(utils._coerce_usage_int(5), 5)

    def test_connection_error_degrades_to_none(self):
        with mock.patch("mini_code_cli.utils.requests.post",
                        side_effect=Exception("boom")):
            content, tool_calls, usage = utils.call_openai_server(
                [{"role": "user", "content": "hi"}], server_url="http://x")
        self.assertIsNone(content)
        self.assertIsNone(tool_calls)
        self.assertIsNone(usage)


# ---------------------------------------------------------------------------
# agent loop: empty response + no-response handling
# ---------------------------------------------------------------------------

def _make_args(**kw):
    a = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None, api_key=None,
        system_prompt=None, agent_md=None, prompt=None, enable_shell=False,
        ask_permission=False, auto_mode=False, restore_session=None, llm_timeout=30,
        tool_timeout=30, load_last_session=False, verbose=False, max_tokens=None,
        temperature=1.0, plan=False, enable_subagents=False, enable_skills=False, sudo=False,
    )
    a.__dict__.update(kw)
    return a


class _EmptyUserTurnBase(unittest.TestCase):
    """Run a session with a scripted console and a mocked LLM."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)
        self.llm_calls = 0

    def _counting_llm(self, answer="hi", tool_calls=None):
        def fake(messages, **kw):
            self.llm_calls += 1
            return answer, tool_calls or [], {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return fake

    def _session(self, **kw):
        # Use a fresh temp dir as --allowed-dir so there is no .mini last-session
        # pointer (which would otherwise trigger run()'s interactive resume
        # offer and consume the first scripted read_line).
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            return agent_loop.AgentSession(
                _make_args(cache_dir=self.tmp.name, allowed_dir=self.tmp.name),
                console=self.console)


class TestEmptyUserResponse(_EmptyUserTurnBase):
    def test_blank_response_reprompts_without_calling_llm(self):
        # A blank line (and one with only spaces) must be ignored and re-prompted,
        # never sent to the model. /quit then ends the loop.
        sess = self._session()
        with mock.patch.object(agent_loop.utils, "call_openai_server", self._counting_llm()), \
             mock.patch.object(self.console, "read_line",
                               side_effect=["", "   ", "/quit"]):
            sess.run()
        # No user turn was ever worth an LLM call.
        self.assertEqual(self.llm_calls, 0)
        # And no empty user message was appended.
        self.assertFalse(any(m.get("role") == "user" and not m.get("content")
                             for m in sess.messages))

    def test_real_input_still_sent_to_llm(self):
        # Sanity check: a non-empty line does reach the model.
        sess = self._session()
        with mock.patch.object(agent_loop.utils, "call_openai_server", self._counting_llm()), \
             mock.patch.object(self.console, "read_line", side_effect=["hello", "/quit"]):
            sess.run()
        self.assertEqual(self.llm_calls, 1)
        self.assertTrue(any(m.get("role") == "user" and m.get("content") == "hello"
                            for m in sess.messages))


class TestNoResponseFromLLM(_EmptyUserTurnBase):
    def test_empty_response_and_no_tool_calls_keeps_loop_alive(self):
        # The LLM returns nothing (no text, no tool calls): the loop must record
        # a warning and keep going (re-prompt), not crash.
        sess = self._session()

        def empty(messages, **kw):
            self.llm_calls += 1
            return "", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        with mock.patch.object(agent_loop.utils, "call_openai_server", empty), \
             mock.patch.object(self.console, "read_line", side_effect=["go", "/quit"]):
            sess.run()  # must not raise
        # The session persisted and survived.
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name, sess.unique_id)))

    def test_llm_returns_none_everything_keeps_loop_alive(self):
        # A hard failure returns (None, None, None); the loop survives and can
        # still be driven to a clean /quit.
        sess = self._session()

        def boom(messages, **kw):
            return None, None, None

        with mock.patch.object(agent_loop.utils, "call_openai_server", boom), \
             mock.patch.object(self.console, "read_line", side_effect=["go", "/quit"]):
            sess.run()  # must not raise
        self.assertTrue(os.path.isdir(os.path.join(self.tmp.name, sess.unique_id)))


if __name__ == "__main__":
    unittest.main()
