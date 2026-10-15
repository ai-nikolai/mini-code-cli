"""End-to-end integration test for the agent loop (fully offline, mocked LLM).

This is the former ``verify_integration.py`` smoke test, now a proper
``unittest``. It drives a whole :class:`AgentSession` through the interactive
loop with a mocked LLM, exercising the parts a unit test alone cannot:

  * preset-prompt handling and the main loop,
  * real tool execution (the LLM "asks" to write the plan / goal files and the
    ``write_file`` tool actually writes them -- plan-file writes are allowed
    even though the session dir is normally outside ``--allowed-dir``),
  * persistence and clean shutdown,
  * the plan-mode goal / plan files on disk.

There is no network access: both ``call_openai_server`` and
``determine_vllm_model`` are patched.
"""
import io
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mini_code_cli.agent_loop as agent_loop
from mini_code_cli import ui, plan as plan_utils


def _make_args(**kw):
    a = types.SimpleNamespace(
        url="http://x", model="m", allowed_dir=".", cache_dir=None, api_key=None,
        system_prompt=None, agent_md=None, prompt=None, enable_shell=False,
        ask_permission=False, auto_mode=False, restore_session=None, llm_timeout=30,
        tool_timeout=30, load_last_session=False, verbose=False, max_tokens=None,
        temperature=1.0, plan=False,
    )
    a.__dict__.update(kw)
    return a


class TestAgentLoopIntegration(unittest.TestCase):
    """Drive a full session end-to-end with a mocked LLM (plan mode)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)

    def _fake_llm(self, session):
        """A tiny stand-in for the LLM server.

        On the first assistant turn it asks the model to write the plan and
        goal files; on every later turn it just says "done" with no tool calls.
        This drives a *real* tool execution inside the loop.
        """
        def fake_call(messages, **kw):
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if n == 0:
                sdir = session.plan_manager.session_dir
                calls = [
                    {"id": "w1", "function": {"name": "write_file",
                     "arguments": json.dumps({"file_path": os.path.join(sdir, plan_utils.PLAN_FILENAME),
                                              "content": "# Plan\n1. do A\n2. do B\n"})}},
                    {"id": "w2", "function": {"name": "write_file",
                     "arguments": json.dumps({"file_path": os.path.join(sdir, plan_utils.GOAL_FILENAME),
                                              "content": "Solve the puzzle\n"})}},
                ]
                return "Planning...", calls, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            return "done", [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        return fake_call

    def _read_file(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_plan_mode_end_to_end(self):
        args = _make_args(cache_dir=self.tmp.name, allowed_dir=".", plan=True,
                          prompt="Solve the puzzle")
        sess = agent_loop.AgentSession(args, console=self.console)

        # /quit ends the interactive loop after the preset prompt has run.
        inputs = iter(["/quit"])
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value=None), \
             patch.object(agent_loop.utils, "call_openai_server", self._fake_llm(sess)), \
             patch.object(self.console, "read_line", side_effect=lambda *a, **k: next(inputs)):
            sess.run(initial_prompt=args.prompt)

        sdir = sess.plan_manager.session_dir
        plan = self._read_file(os.path.join(sdir, plan_utils.PLAN_FILENAME))
        goal = self._read_file(os.path.join(sdir, plan_utils.GOAL_FILENAME))

        # The LLM's write_file tool calls actually ran and produced the files.
        self.assertIn("do A", plan)
        self.assertIn("do B", plan)
        self.assertIn("Solve the puzzle", goal)
