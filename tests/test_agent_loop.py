"""Tests for the agent loop changes:
  #1 memory compaction wiring (token_stats threading), and
  #2 shell execution forces ask-permission.
"""
import os
import sys
import types
import unittest

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
