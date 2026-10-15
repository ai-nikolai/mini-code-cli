"""Tests for the agent loop changes:
  #1 memory compaction wiring (token_stats threading), and
  #2 shell execution forces ask-permission.
"""
import os
import sys
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
    format_usage,
)


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
        # (so we can test the wiring without a live LLM server).
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        result = maybe_compact(list(msgs), "http://x", "m", "k", 60, {}, max_messages=100)
        self.assertEqual(result, msgs)

    def test_accumulate_still_works(self):
        stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}
        accumulate_usage(stats, {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
        self.assertEqual(stats["prompt_tokens"], 3)
        self.assertEqual(stats["llm_calls"], 1)

    def test_format_usage_handles_none(self):
        # Server may report no usage at all.
        self.assertIn("n/a", format_usage(None))
        self.assertIn("input=1", format_usage({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}))


if __name__ == "__main__":
    unittest.main()


def _make_args(**kw):
    """A full args object for AgentSession, overriding a couple of defaults."""
    args = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None,
        api_key=None, system_prompt=None, agent_md=None, prompt=None,
        enable_shell=False, ask_permission=False, auto_mode=False,
        restore_session=None, llm_timeout=30, tool_timeout=30,
    )
    args.__dict__.update(kw)
    return args


class TestRestoreSession(unittest.TestCase):
    """Restoring a previous session continues its history under a brand-new id."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = os.path.join(self.tmp.name, "1792028090823027-a9d77653")
        os.makedirs(self.dir)
        self.history = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "remember this?"},
            {"role": "assistant", "content": "I remember that."},
        ]
        with open(os.path.join(self.dir, "history.json"), "w") as f:
            json.dump(self.history, f)
        # Patch out the network call AgentSession makes to discover the model.
        self._patcher = unittest.mock.patch.object(
            agent_loop.utils, "determine_vllm_model", return_value=None)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def _write_usage(self, **stats):
        with open(os.path.join(self.dir, "usage.json"), "w") as f:
            json.dump(stats, f)

    def test_restore_from_directory_path(self):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(restore_session=self.dir))
        self.assertEqual(sess.messages, self.history)
        # A brand-new unique id, distinct from the restored folder name.
        self.assertNotEqual(sess.unique_id, "1792028090823027-a9d77653")

    def test_restore_from_history_file_path(self):
        history_path = os.path.join(self.dir, "history.json")
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(restore_session=history_path))
        self.assertEqual(sess.messages, self.history)

    def test_tilde_path_is_expanded(self):
        home = os.path.join(self.tmp.name, "home")
        os.makedirs(os.path.join(home, "abc-123"))
        with open(os.path.join(home, "abc-123", "history.json"), "w") as f:
            json.dump(self.history, f)
        with mock.patch.object(os.path, "expanduser",
                               side_effect=lambda p: p.replace("~", home, 1) if p.startswith("~") else p):
            with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
                sess = agent_loop.AgentSession(_make_args(restore_session="~/abc-123"))
        self.assertEqual(sess.messages, self.history)

    def test_usage_is_seeded_from_saved(self):
        self._write_usage(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                          llm_calls=2, tool_calls=1)
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(restore_session=self.dir))
        self.assertEqual(sess.token_stats["prompt_tokens"], 10)
        self.assertEqual(sess.token_stats["completion_tokens"], 5)
        self.assertEqual(sess.token_stats["total_tokens"], 15)
        self.assertEqual(sess.token_stats["llm_calls"], 2)
        self.assertEqual(sess.token_stats["tool_calls"], 1)

    def test_missing_usage_is_ok(self):
        # Restoring with only a history.json (no usage.json) must still work.
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(restore_session=self.dir))
        self.assertEqual(sess.messages, self.history)
        self.assertEqual(sess.token_stats["llm_calls"], 0)

    def test_missing_restore_falls_back_to_fresh_system_prompt(self):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(restore_session="/nope/does/not/exist"))
        self.assertTrue(sess.messages)
        self.assertEqual(sess.messages[0]["role"], "system")
        self.assertIn("tools", sess.messages[0]["content"])

    def test_none_restore_falls_back_to_fresh(self):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(restore_session=None))
        self.assertEqual(sess.messages[0]["role"], "system")
        self.assertIn("tools", sess.messages[0]["content"])

    def test_load_history_none_for_empty_input(self):
        self.assertIsNone(agent_loop.load_history(None))
        self.assertIsNone(agent_loop.load_history(""))

    def test_load_history_none_for_malformed_json(self):
        with open(os.path.join(self.dir, "broken.json"), "w") as f:
            f.write("{not valid json")
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            self.assertIsNone(agent_loop.load_history(os.path.join(self.dir, "broken.json")))

    def test_load_history_none_for_non_list(self):
        with open(os.path.join(self.dir, "obj.json"), "w") as f:
            json.dump({"role": "system"}, f)
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            self.assertIsNone(agent_loop.load_history(os.path.join(self.dir, "obj.json")))
