"""Tests for the sub-agent feature (subagents/ + its wiring into agent_loop.py).

Covers:
  * the SubAgent base class in isolation (schema, prompt, arg building),
  * the registry (register / unregister / get_registry),
  * the loop wiring: a sub-agent shows up as a tool, and a real delegation
    spins up a nested AgentSession that runs to completion and returns text.

All tests are fully offline -- the LLM is always mocked, so no server is needed.
"""
import io
import json
import os
import sys
import types
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli import agent_loop, subagents, ui
from mini_code_cli.subagents.base import SubAgent
from mini_code_cli.subagents import builtin


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


class TestSubAgentBase(unittest.TestCase):
    def test_tool_schema_shape(self):
        schema = builtin.ResearchSubAgent().tool_schema()
        self.assertEqual(schema["type"], "function")
        fn = schema["function"]
        self.assertEqual(fn["name"], "research")
        self.assertIn("task", fn["parameters"]["properties"])
        self.assertIn("max_steps", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["required"], ["task"])

    def test_extra_parameters_are_merged(self):
        schema = builtin.ResearchSubAgent().tool_schema()
        props = schema["function"]["parameters"]["properties"]
        self.assertIn("sources", props)

    def test_system_prompt_gets_preamble_and_report_path(self):
        sa = builtin.ResearchSubAgent()
        parent_args = _make_args(cache_dir="/tmp/x", allowed_dir=".", plan=True)
        # Patch the network call that determine_vllm_model would otherwise make.
        patcher = patch.object(agent_loop.utils, "determine_vllm_model", return_value="m")
        with patcher:
            parent2 = agent_loop.AgentSession(parent_args,
                                              console=ui.Console(stream=io.StringIO(), color=False, unicode=False))
        prompt = sa.system_prompt("explain os.walk", parent2)
        self.assertIn("SUB-AGENT", prompt)
        self.assertIn("explain os.walk", prompt)
        self.assertIn("report.md", prompt)

    def test_build_child_args_inherits_and_narrows(self):
        sa = builtin.ResearchSubAgent()
        parent_args = _make_args(cache_dir="/tmp/x", allowed_dir="/allowed",
                                 enable_shell=True, plan=True, model="parent-model")
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"):
            parent = agent_loop.AgentSession(parent_args,
                                             console=ui.Console(stream=io.StringIO(), color=False, unicode=False))
        child = sa._build_child_args(parent, "do the thing")
        # Inherited from the parent.
        self.assertEqual(child.url, parent.args.url)
        self.assertEqual(child.api_key, parent.args.api_key)
        self.assertEqual(child.allowed_dir, parent.allowed_dir)
        # Narrowed by the sub-agent.
        self.assertTrue(child.auto_mode)          # runs unattended
        self.assertFalse(child.enable_shell)      # research is read-only
        self.assertTrue(child.plan)               # inherits parent plan mode
        # The child is told where to write its report, and the child session's
        # report_path is set to exactly that file so it can exempt it from the
        # sandbox (the report lives under the parent's cache dir, which is
        # usually outside --allowed-dir).
        self.assertIn("report.md", child.system_prompt)
        self.assertEqual(child.report_path, os.path.join(child.cache_dir, "report.md"))
        # Tool selection is done via the system prompt (built by system_prompt),
        # not via an args field -- confirm the child system prompt only lists
        # the sub-agent's allowed (read-only web) tools.
        self.assertIn("web_search", child.system_prompt)
        self.assertNotIn("run_shell_command", child.system_prompt)

    def test_planner_forces_plan_mode(self):
        sa = builtin.PlanSubAgent()
        parent_args = _make_args(cache_dir="/tmp/x", allowed_dir=".", plan=False)
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"):
            parent = agent_loop.AgentSession(parent_args,
                                             console=ui.Console(stream=io.StringIO(), color=False, unicode=False))
        self.assertTrue(sa.wants_plan_mode(parent))


class TestSubAgentRegistry(unittest.TestCase):
    def setUp(self):
        self._saved = dict(subagents.REGISTRY)
        subagents.clear_registry()

    def tearDown(self):
        subagents.REGISTRY.clear()
        subagents.REGISTRY.update(self._saved)

    def test_register_and_unregister(self):
        class Dummy(SubAgent):
            name = "dummy"
            description = "desc"

        subagents.register(Dummy())
        self.assertIn("dummy", subagents.get_registry())
        subagents.unregister("dummy")
        self.assertNotIn("dummy", subagents.get_registry())

    def test_re_register_replaces(self):
        class A(SubAgent):
            name = "x"
            description = "first"

        class B(SubAgent):
            name = "x"
            description = "second"

        subagents.register(A())
        subagents.register(B())
        self.assertEqual(subagents.get_registry()["x"].description, "second")


class _BaseSession(unittest.TestCase):
    """Common setup: a session with sub-agents enabled and a mocked LLM."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False, verbose=False)
        subagents.clear_registry()
        self.addCleanup(lambda: subagents.REGISTRY.clear())

    def _session(self, **kw):
        args = _make_args(cache_dir=self.tmp.name, allowed_dir=self.tmp.name,
                          enable_subagents=True, plan=True, **kw)
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"):
            return agent_loop.AgentSession(args, console=self.console)

    def _child_like_session(self, **kw):
        # Like _session, but with the cache dir as a SIBLING of --allowed-dir,
        # mirroring the real sub-agent layout: the report lives under the cache
        # dir, which is OUTSIDE the working (--allowed-dir) directory.
        work = os.path.join(self.tmp.name, "work")
        os.makedirs(work, exist_ok=True)
        subagents.register(builtin.ResearchSubAgent())
        args = _make_args(cache_dir=self.tmp.name, allowed_dir=work,
                          enable_subagents=True, plan=True, **kw)
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"):
            return agent_loop.AgentSession(args, console=self.console)


class TestSubAgentWiring(_BaseSession):
    def test_subagent_becomes_a_tool(self):
        subagents.register(builtin.ResearchSubAgent())
        sess = self._session()
        self.assertIn("research", sess.subagents_registry)
        schema = next(t for t in sess.llm_tools if t.get("function", {}).get("name") == "research")
        self.assertEqual(schema["type"], "function")
        # The tools_dict entry is a sentinel (no real function) so the loop knows
        # to delegate rather than call something.
        self.assertIsNone(sess.tools_dict["research"]["function"])
        self.assertTrue(sess.tools_dict["research"]["subagent"])

    def test_delegation_runs_nested_session_and_returns_text(self):
        subagents.register(builtin.ResearchSubAgent())
        sess = self._session()

        def parent_llm(messages, **kw):
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if n == 0:
                return "I'll delegate this.", [
                    {"id": "r1", "function": {"name": "research",
                     "arguments": '{"task": "explain os.walk"}'}}
                ], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            return "all set", [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        # The sub-agent writes a report to its own session dir. We intercept
        # write_file so the test works without touching the real filesystem.
        orig_write = agent_loop.tools.write_file
        reports_written = []

        def spy_write_file(file_path, content, *a, **k):
            reports_written.append(file_path)
            return orig_write(file_path, content, *a, **k)

        with patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"), \
             patch("mini_code_cli.utils.call_openai_server", side_effect=parent_llm):
            # Run two loop turns: turn 1 delegates (sub-agent runs to completion
            # inside), turn 2 gets the parent's follow-up answer.
            sess.messages.append({"role": "user", "content": "please research os.walk"})
            sess._step()
            sess._persist()
            sess._step()
            sess._persist()

        # The sub-agent's tool result should contain the "Result from sub-agent"
        # text produced by _build_result.
        joined = "\n".join(m.get("content", "") for m in sess.messages)
        self.assertIn("Result from sub-agent 'research'", joined)

    def test_subagent_tool_result_recorded(self):
        subagents.register(builtin.ResearchSubAgent())
        sess = self._session()

        calls = [
            {"id": "r1", "function": {"name": "research", "arguments": '{"task": "x"}'}}
        ]
        sess.messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        with patch("mini_code_cli.utils.call_openai_server",
                   return_value=("done", [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})):
            # _run_one_tool_call should delegate and append a tool result.
            before = len(sess.messages)
            sess._run_one_tool_call(calls[0])
            self.assertEqual(len(sess.messages), before + 1)
            self.assertEqual(sess.messages[-1]["role"], "tool")
            self.assertIn("Result from sub-agent 'research'", sess.messages[-1]["content"])

    def test_report_write_is_blocked_when_not_the_childs_report(self):
        # A write to the cache dir that is NOT the report file must still be
        # blocked by the sandbox -- the exemption is scoped to report.md only.
        sess = self._child_like_session()
        sa = subagents.get_registry()["research"]
        sess.report_path = sa._report_path(sess)
        # A sibling file in the same dir as report.md (outside --allowed-dir, and
        # whose parent already exists) that is NOT report.md.
        sdir = os.path.join(os.path.dirname(sess.report_path), "subagents", "research")
        os.makedirs(sdir, exist_ok=True)
        other = os.path.join(sdir, "secret.txt")
        call = {"id": "w1", "function": {"name": "write_file",
                "arguments": json.dumps({"file_path": other, "content": "x"})}}
        with patch("mini_code_cli.utils.call_openai_server",
                   return_value=("ok", [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})):
            sess._run_one_tool_call(call)
        last = sess.messages[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("not allowed outside allowed directory", last["content"])
        # And it must not have been written.
        self.assertFalse(os.path.exists(other))

    def test_report_write_allowed_even_when_cache_dir_outside_allowed_dir(self):
        # The real-world layout: --allowed-dir is the working dir, the cache dir
        # (and hence the report) is elsewhere. The child's write of its own
        # report.md must pass the sandbox even though it lives outside
        # --allowed-dir.
        sess = self._child_like_session()
        sa = subagents.get_registry()["research"]
        report_path = sa._report_path(sess)
        sess.report_path = report_path
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        call = {"id": "w1", "function": {"name": "write_file",
                "arguments": json.dumps({"file_path": report_path,
                                         "content": "REPORT OK"})}}
        with patch("mini_code_cli.utils.call_openai_server",
                   return_value=("ok", [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})):
            sess._run_one_tool_call(call)
        last = sess.messages[-1]
        self.assertEqual(last["role"], "tool")
        self.assertNotIn("not allowed", last["content"])
        self.assertIn("REPORT OK", open(report_path, encoding="utf-8").read())

    def test_report_path_defaults_none_for_normal_session(self):
        # A plain (non-child) session must not carry a report exemption.
        sess = self._session()
        self.assertIsNone(sess.report_path)
        # With no report_path, _is_report_file is always False.
        self.assertFalse(sess._is_report_file("write_file", {"file_path": "/whatever/report.md"}))

    def test_child_loop_disables_subagents_and_exempts_its_report(self):
        # A sub-agent's child loop must not itself offer sub-agents (no recursion):
        # the built child args omit enable_subagents, which the loop treats as
        # disabled via getattr(..., False). It also carries its own report path
        # so the child can write report.md back to the parent.
        subagents.register(builtin.ResearchSubAgent())
        sess = self._child_like_session()
        sa = subagents.get_registry()["research"]
        child = sa._build_child_args(sess, "do x")
        # Child runs unattended, without shell, and without sub-agents.
        self.assertTrue(child.auto_mode)
        self.assertFalse(child.enable_shell)
        self.assertFalse(getattr(child, "enable_subagents", False))
        # Its report lives under the parent cache dir and is exactly the exempt
        # path the child session will recognise.
        self.assertEqual(child.report_path, sa._report_path(sess))
        # With the cache dir outside --allowed-dir, a plain sandbox check on the
        # report path would fail -- proving the exemption is genuinely needed.
        self.assertFalse(agent_loop.tool_paths_are_allowed("write_file",
                        {"file_path": child.report_path}, child.allowed_dir))
        # The child session built from these args recognises its report.md as the
        # single exempt file (and nothing else).
        with patch.object(agent_loop.utils, "determine_vllm_model", return_value="m"):
            c = agent_loop.AgentSession(child, console=self.console)
        self.assertEqual(c.report_path, child.report_path)
        self.assertTrue(c._is_report_file("write_file", {"file_path": child.report_path}))
        # ...but any other path is still sandboxed.
        other = os.path.join(os.path.dirname(child.report_path), "x.txt")
        self.assertFalse(c._is_report_file("write_file", {"file_path": other}))

    def test_subagent_report_is_read_back_and_returned(self):
        # Regression: the child must write report.md to exactly the path the
        # system prompt advertises (self._report_path(parent)), and the parent
        # must read it back from there. Here the child writes the report via the
        # real write_file tool (to its own session dir under the parent's
        # allowed dir), then the parent returns it alongside the final answer.
        subagents.register(builtin.ResearchSubAgent())
        sess = self._session()
        sa = subagents.get_registry()["research"]
        report_path = sa._report_path(sess)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        def child_llm(messages, **kw):
            # First turn: the sub-agent writes its report, then a second turn
            # returns the final assistant text with no further tool calls.
            n = sum(1 for m in messages if m.get("role") == "assistant")
            if n == 0:
                calls = [{"id": "w1", "function": {"name": "write_file",
                          "arguments": json.dumps({"file_path": report_path,
                                                   "content": "REPORT BODY HERE"})}}]
                return "working on it", calls, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            return "the final answer", [], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

        call = {"id": "r1", "function": {"name": "research", "arguments": '{"task": "x"}'}}
        with patch("mini_code_cli.utils.call_openai_server", side_effect=child_llm):
            # A sub-agent never sets enable_subagents, so it cannot call the
            # parent's "research" tool back -- the child has no such tool.
            sess._run_one_tool_call(call)

        result = sess.messages[-1]["content"]
        self.assertIn("Result from sub-agent 'research'", result)
        self.assertIn("the final answer", result)
        self.assertIn("REPORT BODY HERE", result)

    def test_subagent_failure_is_reported_as_error(self):
        subagents.register(builtin.ResearchSubAgent())
        sess = self._session()
        # Make the sub-agent blow up on run().
        def boom(task, parent, max_steps=None):
            raise RuntimeError("kaboom")
        with patch.object(subagents.get_registry()["research"], "run", side_effect=boom):
            call = {"id": "r1", "function": {"name": "research", "arguments": '{"task": "x"}'}}
            sess._run_one_tool_call(call)
            self.assertEqual(sess.messages[-1]["role"], "tool")
            self.assertIn("Error: sub-agent 'research' failed", sess.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
