"""Tests for the agent loop changes:
  #1 memory compaction wiring (token_stats threading), and
  #2 shell execution forces ask-permission.
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
from mini_code_cli.agent_loop import (
    maybe_compact,
    shell_forces_ask_permission,
    accumulate_usage,
)
from mini_code_cli import ui


class _Args:
    def __init__(self, **kw):
        self.enable_shell = False
        self.ask_permission = False
        self.__dict__.update(kw)


class TestShellForcesPermission(unittest.TestCase):
    def test_shell_enabled_forces_permission(self):
        self.assertTrue(shell_forces_ask_permission(_Args(enable_shell=True)))

    def test_shell_disabled_does_not_force(self):
        self.assertFalse(shell_forces_ask_permission(_Args(enable_shell=False)))

    def test_missing_attr_is_safe(self):
        # An args object without enable_shell should default to False (no crash).
        self.assertFalse(shell_forces_ask_permission(types.SimpleNamespace()))


class TestMaybeCompactWiring(unittest.TestCase):
    def test_signature_accepts_token_stats(self):
        import inspect
        params = inspect.signature(maybe_compact).parameters
        self.assertIn("token_stats", params)

    def test_no_history_change_when_small(self):
        # A short history must pass through unchanged and never touch the network
        # (so we can test the wiring without a live LLM server). maybe_compact
        # returns (messages, did_compact, history_index); on a short history the
        # messages are unchanged, no compaction happened, and the segment index
        # is passed straight through.
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        # token_stats is below the 100-token budget, so nothing is compacted.
        stats = {"total_tokens": 0}
        result = maybe_compact(list(msgs), "http://x", "m", "k", 60, stats, max_total_tokens=100)
        messages, did_compact, history_index = result
        self.assertEqual(messages, msgs)
        self.assertFalse(did_compact)
        self.assertEqual(history_index, 0)

    def test_compaction_triggers_on_tokens_not_message_count(self):
        # Compaction is decided by total accumulated tokens, never by how many
        # messages there are: over a 100-token budget the 3 short messages must
        # compact; under it they must not. We stub the summarization call so no
        # network is needed (and use a full token_stats dict, since
        # accumulate_usage does in-place += on its keys).
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"}]
        stats_over = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 150, "llm_calls": 0}
        stats_under = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 90, "llm_calls": 0}
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               return_value=("summary", [], usage)):
            _, did_over, _ = maybe_compact(msgs, "http://x", "m", "k", 60, stats_over, max_total_tokens=100)
            _, did_under, _ = maybe_compact(msgs, "http://x", "m", "k", 60, stats_under, max_total_tokens=100)
        self.assertTrue(did_over)
        self.assertFalse(did_under)
        # A missing total_tokens means 0 -> under budget -> no compaction.
        _, did_missing, _ = maybe_compact(msgs, "http://x", "m", "k", 60, {})
        self.assertFalse(did_missing)

    def test_accumulate_still_works(self):
        stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        accumulate_usage(stats, {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
        self.assertEqual(stats["prompt_tokens"], 3)
        self.assertEqual(stats["llm_calls"], 1)


def _run_turn(sess, user_text, answer, tool_calls=None):
    """Feed one user turn through a single (mocked) LLM step.

    Patches the network call so no server is needed, then appends the user
    message, runs one step and persists. ``answer`` is the assistant reply and
    ``tool_calls`` (a list, or None) is what the mock LLM "wants" to run.

    The tool call is recorded in the history exactly as the real loop would
    (assistant message with ``tool_calls`` + a tool result), but we stub the
    actual execution so a mocked tool like ``read_file`` never runs for real.
    """
    def fake_call(messages, **kw):
        return answer, tool_calls or [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    with mock.patch.object(agent_loop.utils, "call_openai_server", fake_call):
        with mock.patch.object(sess, "_run_tool_calls", side_effect=lambda tcs: [
                sess._append_tool.computed for _ in tcs] if False else
                [sess._append_tool_result(tc.get("id", ""), "mocked tool output") for tc in (tool_calls or [])]):
            sess.messages.append({"role": "user", "content": user_text})
            sess._step()
            sess._persist()


def _make_args(**kw):
    """A full args object for AgentSession, overriding a couple of defaults."""
    args = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None,
        api_key=None, system_prompt=None, agent_md=None, prompt=None,
        enable_shell=False, ask_permission=False, auto_mode=False,
        restore_session=None, llm_timeout=30, tool_timeout=30,
        load_last_session=False, verbose=False, max_tokens=None, temperature=1.0,
    )
    args.__dict__.update(kw)
    return args


class TestConversationHistory(unittest.TestCase):
    """The high-level recorder + /history + /reset navigation.

    A session records one flat, high-level entry per turn (turn number, what
    happened, and which history_{n}.json + index the turn lives in). These tests
    drive a mocked LLM through a few turns and check that history can be listed,
    inspected and rewound, and that the recorder survives compaction.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            self.sess = agent_loop.AgentSession(
                _make_args(cache_dir=self.tmp.name), console=self.console)

    def _read(self):
        text = self.buf.getvalue()
        self.buf.truncate(0)
        self.buf.seek(0)
        return text

    def _load_summary(self):
        with open(os.path.join(self.tmp.name, self.sess.unique_id, "summary_history.json")) as f:
            return json.load(f)

    def test_history_list_renders_without_error(self):
        _run_turn(self.sess, "hello", "hi there")
        self._read()  # drop startup + usage output emitted before the command
        handled = self.sess._handle_command("/history")
        self.assertTrue(handled)
        text = self._read()
        self.assertIn("Conversation history", text)
        self.assertIn("assistant", text)
        self.assertIn("history_0.json", text)

    def test_history_list_handles_tool_call_rows(self):
        # A tool call must render as a single well-formed row, not crash the
        # 3-tuple unpacking in Console.history_list (a previous bug stringified
        # each row tuple first, which broke the `for turn, what, where` unpack).
        _run_turn(self.sess, "q0", "thinking",
                  tool_calls=[{"id": "c1", "function": {"name": "read_file",
                                                        "arguments": "{}"}}])
        self._read()
        handled = self.sess._handle_command("/history")
        self.assertTrue(handled)
        self.assertIn("tool call", self._read())

    def test_inspect_history_shows_messages_of_that_turn(self):
        _run_turn(self.sess, "remember 42", "noted: 42")
        _run_turn(self.sess, "what did i say", "you said 42")
        self._read()
        self.sess._inspect_history("1")
        text = self._read()
        # Turn 1 is the first assistant reply; its snapshot must include the
        # first user turn and that reply, but not the second.
        self.assertIn("remember 42", text)
        self.assertIn("noted: 42", text)
        self.assertNotIn("what did i say", text)

    def test_history_numbers_are_1_based_on_turns(self):
        # The recorder is 1-based on real turns (assistant replies), not on
        # message index, so turn 1 must be the first assistant answer.
        _run_turn(self.sess, "q0", "a0")
        self._read()
        with open(os.path.join(self.tmp.name, self.sess.unique_id, "summary_history.json")) as f:
            rec = json.load(f)
        self.assertEqual(rec[0]["turn"], 1)
        self.assertEqual(rec[0]["what"], "assistant")

    def test_inspect_history_bad_turn_is_reported(self):
        _run_turn(self.sess, "q", "a")
        self._read()
        self.sess._inspect_history("999")
        self.assertIn("No such turn", self._read())

    def test_reset_drops_later_messages_and_records(self):
        _run_turn(self.sess, "q0", "a0")
        _run_turn(self.sess, "q1", "a1")
        _run_turn(self.sess, "q2", "a2")
        self._read()
        before = len(self.sess.messages)
        self.sess._do_reset("1")  # rewind to end of first assistant reply
        self.assertLess(len(self.sess.messages), before)
        # Messages now end at the first assistant answer.
        self.assertIn("a0", self.sess.messages[-1]["content"])
        # Recorder was truncated to just the kept turns. One assistant reply
        # ("a0") remains, so the recorder holds exactly one record.
        self.assertEqual(len(self.sess.summary), 1)

    def test_reset_then_continue_appends_from_reset_state(self):
        _run_turn(self.sess, "q0", "a0")
        _run_turn(self.sess, "q1", "a1")
        self._read()
        self.sess._do_reset("1")
        _run_turn(self.sess, "after reset", "resumed")
        self.assertIn("resumed", self.sess.messages[-1]["content"])
        joined = "\n".join(m["content"] for m in self.sess.messages)
        self.assertIn("a0", joined)
        # 'a1' should have been discarded by the reset.
        self.assertNotIn("a1", joined)

    def test_reset_no_arg_keeps_all(self):
        _run_turn(self.sess, "q0", "a0")
        _run_turn(self.sess, "q1", "a1")
        self._read()
        n = len(self.sess.messages)
        self.sess._do_reset("")  # reset to most recent turn: a no-op
        self.assertEqual(len(self.sess.messages), n)

    def test_reset_history_survives_compaction(self):
        # Force compaction to roll segments (history_0 -> history_1), then make
        # sure /history still lists every turn across the segment boundary and
        # a reset to a mid-segment turn reconstructs a faithful prefix.
        #
        # We wrap (not replace) the real maybe_compact so the real segment logic
        # runs, just with a tiny threshold. Patching the module name directly
        # would recurse (the wrapper calls the name it replaced), so we call the
        # private _maybe_compact_impl via the real function object captured first.
        real_compact = agent_loop.maybe_compact
        # Force a tiny token budget regardless of what the loop passes, so
        # compaction rolls the segment within a handful of turns.
        def compact(messages, server_url, model, api_key, llm_timeout, token_stats,
                    history_index=0, max_total_tokens=5):
            return real_compact(
                messages, server_url, model, api_key, llm_timeout, token_stats,
                history_index, max_total_tokens=5)
        with mock.patch.object(agent_loop, "maybe_compact", compact):
            for i in range(4):
                _run_turn(self.sess, f"q{i}", f"a{i}")
        self._read()
        # history_1.json should now exist (a new segment was written).
        self.assertTrue(os.path.exists(
            os.path.join(self.tmp.name, self.sess.unique_id, "history_1.json")))
        # /history lists all turns across segments without error.
        handled = self.sess._handle_command("/history")
        self.assertTrue(handled)
        self.assertIn("Conversation history", self._read())
        # Reset to the second turn (still in the first segment) works.
        self.sess._do_reset("2")
        self.assertIn("a0", "\n".join(m["content"] for m in self.sess.messages))
        self.assertEqual(self.sess.history_index, 0)


if __name__ == "__main__":
    unittest.main()
