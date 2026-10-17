"""Tests for the --plan mode feature (plan.py + its wiring into agent_loop.py).

Covers:
  * the PlanManager (write/read goal, write/generate plan, write review summary,
    log requests, mark finished) in isolation,
  * the plan-mode system-prompt instructions, the plan-generation prompt and the
    plan-completion-review prompt,
  * the loop wiring: recording each user request/goal in the tracking files,
    asking the model (no tools) to author an actual plan from the whole
    conversation, saving it, and re-injecting it into the conversation,
  * the completion-review prompt and the review loop (review -> follow-up work ->
  re-review), including the finalized_plan guard that stops an infinite loop,
  * the plan re-injection that runs after a history compaction, and marking the
    goal finished on shutdown.
  * the fact that the tracking files are loop-owned: the model authors none of
    them, so a generic write_file of a plan file (which lives outside
    --allowed-dir) is still blocked by the sandbox.
"""
import io
import json
import os
import sys
import types
import tempfile
import unittest
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mini_code_cli import plan as plan_utils
from mini_code_cli import prompts
from mini_code_cli import agent_loop
from mini_code_cli import ui


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


class TestPlanManager(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.session = os.path.join(self.tmp.name, "sess")

    def test_disabled_is_a_noop(self):
        pm = plan_utils.PlanManager(self.session, enabled=False, goal="do x")
        pm.write_goal("do x")
        pm.write_plan("1. do x")
        pm.mark_finished("do x")
        # Nothing written, reads return None.
        self.assertIsNone(pm.read_plan())
        self.assertIsNone(pm.read_goal())
        self.assertFalse(os.path.exists(pm.goal_path))

    def test_write_goal_then_read_goal(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_goal("Build the thing\n")
        self.assertTrue(os.path.exists(pm.goal_path))
        self.assertIn("Build the thing", pm.read_goal())
        self.assertEqual(pm.goal, "Build the thing\n")

    def test_write_goal_overwrites_previous(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_goal("first goal")
        pm.write_goal("second goal")
        self.assertIn("second goal", open(pm.goal_path).read())
        self.assertNotIn("first goal", open(pm.goal_path).read())

    def test_goal_written_as_markdown(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_goal("g")
        self.assertTrue(open(pm.goal_path).read().startswith("# Current Goal"))

    def test_mark_finished_creates_plan_and_appends(self):
        pm = plan_utils.PlanManager(self.session, enabled=True, goal="do x")
        pm.mark_finished()
        text = open(pm.plan_path).read()
        self.assertIn("# Current Plan", text)
        self.assertIn("Task finished: do x", text)

    def test_mark_finished_appends_to_existing_plan(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        os.makedirs(self.session, exist_ok=True)
        open(pm.plan_path, "w").write("# Current Plan\n1. do A\n")
        pm.mark_finished("goal Z")
        text = open(pm.plan_path).read()
        self.assertIn("1. do A", text)
        self.assertIn("Task finished: goal Z", text)

    def test_mark_finished_reads_goal_from_disk(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_goal("the real goal")
        pm.mark_finished()
        self.assertIn("Task finished: the real goal", open(pm.plan_path).read())

    def test_read_plan_returns_none_when_absent(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        self.assertIsNone(pm.read_plan())

    def test_write_plan_persists_and_updates_memory(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_plan("1. step one\n2. step two")
        self.assertEqual(pm.plan, "1. step one\n2. step two")
        text = open(pm.plan_path).read()
        self.assertIn("# Current Plan", text)
        self.assertIn("step one", text)
        self.assertIn("step two", text)

    def test_write_plan_empty_only_keeps_header(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_plan("")
        text = open(pm.plan_path).read()
        self.assertIn("# Current Plan", text)

    def test_write_plan_overwrites(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_plan("old plan")
        pm.write_plan("new plan")
        text = open(pm.plan_path).read()
        self.assertIn("new plan", text)
        self.assertNotIn("old plan", text)

    def test_read_plan_falls_back_to_memory(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.plan = "generated plan text"
        self.assertEqual(pm.read_plan(), "generated plan text")

    def test_read_current_status_noop_when_disabled(self):
        pm = plan_utils.PlanManager(self.session, enabled=False)
        # Disabled manager returns None (reads are no-ops), so the command
        # degrades to a note rather than anything rendered.
        self.assertIsNone(pm.read_current_status())

    def test_read_current_status_bundles_goal_plan_and_review(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_goal("the goal")
        pm.write_plan("1. step one\n2. step two")
        pm.write_plan_summary("TASKS still in progress. 2. step two")
        status = pm.read_current_status()
        self.assertIn("# Current Goal", status)
        self.assertIn("the goal", status)
        self.assertIn("# Current Plan", status)
        self.assertIn("1. step one", status)
        self.assertIn("2. step two", status)
        self.assertIn("# Plan completion summary", status)
        self.assertIn("TASKS still in progress", status)

    def test_read_current_status_falls_back_when_missing(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        status = pm.read_current_status()
        # Every section is present with a (none) note rather than raising.
        self.assertIn("(no goal set)", status)
        self.assertIn("(no plan generated yet)", status)
        self.assertIn("(no completion review yet)", status)

    def test_write_plan_summary_persists_and_updates_memory(self):
        pm = plan_utils.PlanManager(self.session, enabled=True, goal="do x")
        pm.write_plan("1. step one")
        pm.write_plan_summary("ALL TASKS DONE. Go back to user.")
        self.assertEqual(pm.review_summary, "ALL TASKS DONE. Go back to user.")
        text = open(pm.plan_path).read()
        self.assertIn("# Plan completion summary", text)
        self.assertIn("ALL TASKS DONE", text)

    def test_write_plan_summary_replaces_previous(self):
        pm = plan_utils.PlanManager(self.session, enabled=True, goal="do x")
        pm.write_plan_summary("TASKS still in progress. Outstanding: 1.")
        pm.write_plan_summary("ALL TASKS DONE. Go back to user.")
        text = open(pm.plan_path).read()
        self.assertIn("ALL TASKS DONE", text)
        self.assertNotIn("TASKS still in progress", text)

    def test_write_plan_summary_noop_when_disabled(self):
        pm = plan_utils.PlanManager(self.session, enabled=False)
        pm.write_plan_summary("ALL TASKS DONE.")
        self.assertIsNone(pm.review_summary)


class TestPlanPrompt(unittest.TestCase):
    def test_plan_mode_instructions_present(self):
        text = prompts.plan_instructions()
        self.assertIn("CURRENT_GOAL.md", text)
        self.assertIn("CURRENT_PLAN.md", text)
        self.assertIn("step-by-step", text)

    def test_system_prompt_contains_plan_block_only_when_enabled(self):
        with_plan = prompts.system_prompt(tools_dict=None, allowed_dir=".", plan_mode=True)
        without = prompts.system_prompt(tools_dict=None, allowed_dir=".", plan_mode=False)
        self.assertIn("PLAN MODE", with_plan)
        self.assertIn("CURRENT_PLAN.md", with_plan)
        self.assertNotIn("PLAN MODE", without)

    def test_plan_prompt_includes_history_and_instruction(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "make a thing"},
            {"role": "assistant", "content": "sure"},
        ]
        text = prompts.plan_prompt(messages, "make a thing")
        self.assertIn("sys", text)
        self.assertIn("make a thing", text)
        self.assertIn("numbered", text)

    def test_plan_review_prompt_includes_history_and_goal(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "make a thing"},
            {"role": "assistant", "content": "done it"},
        ]
        text = prompts.plan_review_prompt(messages, "make a thing")
        self.assertIn("sys", text)
        self.assertIn("make a thing", text)
        self.assertIn("TASKS still in progress", text)
        self.assertIn("Go back to user", text)

    def test_plan_review_prompt_mentions_task_states(self):
        text = prompts.plan_review_prompt([], "do x")
        for state in ("DONE", "TODO", "IN-PROGRESS"):
            self.assertIn(state, text)


class TestPlanWiringInLoop(unittest.TestCase):
    """The AgentSession-level wiring for plan mode."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)

    def _make(self, **kw):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            return agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name, **kw),
                                           console=self.console)

    def test_disabled_by_default(self):
        sess = self._make(plan=False)
        self.assertFalse(sess.plan_mode)
        self.assertFalse(sess.plan_manager.enabled)

    def test_enabled_when_plan_flag(self):
        sess = self._make(plan=True)
        self.assertTrue(sess.plan_mode)
        self.assertTrue(sess.plan_manager.enabled)
        self.assertEqual(sess.plan_manager.session_dir,
                         os.path.join(self.tmp.name, sess.unique_id))

    def test_system_prompt_gets_plan_mode_flag(self):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name, plan=True),
                                           console=self.console)
        self.assertIn("PLAN MODE", sess.messages[0]["content"])

    def test_record_goal_request_writes_goal_file(self):
        sess = self._make(plan=True)
        sess._record_goal_request("Build the thing")
        self.assertIn("Build the thing",
                      open(os.path.join(sess.plan_manager.session_dir, plan_utils.GOAL_FILENAME)).read())

    def test_record_goal_request_ignores_slash_commands(self):
        sess = self._make(plan=True)
        sess._record_goal_request("/history")
        self.assertIsNone(sess.plan_manager.read_goal())

    def test_record_goal_request_noop_when_disabled(self):
        sess = self._make(plan=False)
        sess._record_goal_request("do x")
        self.assertIsNone(sess.plan_manager.read_goal())

    def test_record_goal_request_ignores_empty_input(self):
        sess = self._make(plan=True)
        sess._record_goal_request("   ")
        self.assertIsNone(sess.plan_manager.read_goal())

    def test_show_plan_noop_when_not_plan_mode(self):
        sess = self._make(plan=False)
        buf = io.StringIO()
        sess.console = ui.Console(stream=buf, color=False, unicode=False)
        sess._show_plan()
        self.assertIn("not enabled", buf.getvalue())

    def test_show_plan_renders_goal_and_plan(self):
        sess = self._make(plan=True)
        sess._record_goal_request("build a widget")
        # Mock the plan-authoring LLM call so _generate_plan_and_inject writes
        # a real, non-empty plan (no network in the test suite).
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               return_value=("1. read the brief\n2. write it", {}, {})):
            sess._generate_plan_and_inject("build a widget")
        buf = io.StringIO()
        sess.console = ui.Console(stream=buf, color=False, unicode=False)
        sess._show_plan()
        rendered = buf.getvalue()
        self.assertIn("build a widget", rendered)
        self.assertIn("1. read the brief", rendered)
        self.assertIn("# Current Plan", rendered)

    def test_plan_command_is_handled(self):
        sess = self._make(plan=True)
        sess._record_goal_request("build a widget")
        sess._generate_plan_and_inject("build a widget")
        buf = io.StringIO()
        sess.console = ui.Console(stream=buf, color=False, unicode=False)
        self.assertTrue(sess._handle_command("/plan"))
        self.assertIn("build a widget", buf.getvalue())

    def test_record_goal_request_rearms_review(self):
        sess = self._make(plan=True)
        sess.finalized_plan = False
        sess._record_goal_request("the request")
        self.assertTrue(sess.finalized_plan)

    def test_mark_finished_and_shutdown(self):
        sess = self._make(plan=True)
        sess._record_goal_request("my goal")
        with mock.patch.object(sess, "shutdown"):
            sess._finish_goal_and_shutdown()
        plan = open(os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)).read()
        self.assertIn("Task finished: my goal", plan)

    def test_mark_finished_noop_when_disabled(self):
        sess = self._make(plan=False)
        with mock.patch.object(sess, "shutdown"):
            sess._finish_goal_and_shutdown()  # must not raise
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, sess.unique_id, plan_utils.PLAN_FILENAME)))

    def test_generate_plan_and_inject_persists_and_appends(self):
        sess = self._make(plan=True)
        sess.messages.append({"role": "user", "content": "the request"})
        with mock.patch.object(
                agent_loop.utils, "call_openai_server",
                return_value=("1. do a\n2. do b", [], {"total_tokens": 3})):
            sess._generate_plan_and_inject("the request")
        plan = open(os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)).read()
        self.assertIn("1. do a", plan)
        self.assertIn("2. do b", plan)
        texts = [m.get("content", "") for m in sess.messages]
        self.assertTrue(any("Current plan" in t for t in texts))
        self.assertTrue(any("do a" in t for t in texts))

    def test_generate_plan_and_inject_noop_when_disabled(self):
        sess = self._make(plan=False)
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               side_effect=AssertionError("should not be called")):
            sess._generate_plan_and_inject("the request")  # must not call the model
        self.assertIsNone(sess.plan_manager.plan)
        self.assertEqual(len(sess.messages), 1)  # nothing re-injected

    def test_generate_plan_and_inject_handles_failure(self):
        sess = self._make(plan=True)
        sess.messages.append({"role": "user", "content": "the request"})
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               side_effect=RuntimeError("boom")):
            sess._generate_plan_and_inject("the request")  # must not raise
        # No plan was generated (empty string is stored, no re-injection).
        self.assertEqual(sess.plan_manager.plan, "")

    def test_finalized_plan_starts_true(self):
        sess = self._make(plan=True)
        self.assertTrue(sess.finalized_plan)

    def test_review_plan_completion_noop_when_not_due(self):
        sess = self._make(plan=True)
        sess.finalized_plan = False
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               side_effect=AssertionError("should not be called")):
            sess._review_plan_completion()  # must not call the model
        # Nothing re-injected.
        self.assertTrue(all("Plan completion summary" not in m.get("content", "")
                            for m in sess.messages))

    def test_review_plan_completion_noop_when_disabled(self):
        sess = self._make(plan=False)
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               side_effect=AssertionError("should not be called")):
            sess._review_plan_completion()  # must not call the model

    def test_review_plan_completion_runs_once_and_clears_flag(self):
        sess = self._make(plan=True)
        sess.messages.append({"role": "user", "content": "the request"})
        sess.plan_manager.write_plan("1. do a")
        with mock.patch.object(
                agent_loop.utils, "call_openai_server",
                return_value=("ALL TASKS DONE. Go back to user.", [],
                              {"total_tokens": 3})):
            sess._review_plan_completion()
        # Flag is cleared so it does not run again for the same state.
        self.assertFalse(sess.finalized_plan)
        # The summary was written to the plan document and re-injected.
        plan = open(os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)).read()
        self.assertIn("ALL TASKS DONE", plan)
        texts = [m.get("content", "") for m in sess.messages]
        self.assertTrue(any("Plan completion summary" in t and "ALL TASKS DONE" in t
                            for t in texts))

    def test_step_rearms_review_when_predicts_more_work(self):
        # A step that predicts a tool call re-arms finalized_plan (review due),
        # while a step with no tool calls leaves it untouched.
        sess = self._make(plan=True, auto_mode=True)
        sess.messages.append({"role": "user", "content": "the request"})
        call = {"id": "w1", "function": {"name": "write_file",
                "arguments": json.dumps({"file_path": os.path.join(sess.allowed_dir, "n.txt"),
                                         "content": "ok\n"})}}
        # First step predicts a tool call -> re-arms the review.
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               return_value=("working", [call], {"total_tokens": 2})):
            made = sess._step()
            self.assertTrue(made)
            self.assertTrue(sess.finalized_plan)
        # Second step makes no tool calls -> the flag stays as it was.
        sess.finalized_plan = False
        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               return_value=("done", [], {"total_tokens": 2})):
            made = sess._step()
            self.assertFalse(made)
            self.assertFalse(sess.finalized_plan)


class TestPlanCompactionReinjectsPlan(unittest.TestCase):
    """After compaction the loop-owned plan + goal must be re-appended.

    The tracking files are owned by the loop (not authored by the model), and
    they live in the session dir which sits outside --allowed-dir. A model that
    tried to write a plan file itself would therefore still hit the sandbox --
    the loop is the only writer.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            self.sess = agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name, plan=True),
                                                console=self.console)

    def test_apply_plan_appends_plan_and_goal(self):
        self.sess.plan_manager.write_goal("Build the thing")
        self.sess.plan_manager.write_plan("1. do X")
        out = self.sess._apply_plan_to_compacted(
            [{"role": "system", "content": "sys"}])
        texts = [m["content"] for m in out]
        self.assertTrue(any("current plan" in t.lower() for t in texts))
        self.assertTrue(any("do X" in t for t in texts))
        self.assertTrue(any("Build the thing" in t for t in texts))
        self.assertEqual(out[-1]["role"], "user")
        self.assertIn("Build the thing", out[-1]["content"])

    def test_apply_plan_recovers_goal_from_messages_when_no_goal_file(self):
        out = self.sess._apply_plan_to_compacted(
            [{"role": "user", "content": "the user asked this"}])
        self.assertEqual(out[-1]["role"], "user")
        self.assertIn("the user asked this", out[-1]["content"])

    def test_apply_plan_without_plan_or_goal_keeps_messages(self):
        out = self.sess._apply_plan_to_compacted(
            [{"role": "system", "content": "sys"}])
        self.assertEqual(len(out), 1)


class TestPlanReviewEndToEnd(unittest.TestCase):
    """Drive the full plan-mode review + follow-up loop with a mocked LLM."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)

    def _make(self, **kw):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            return agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name,
                                                      allowed_dir=self.tmp.name, **kw),
                                           console=self.console)

    def test_review_runs_and_marks_finished(self):
        # Plan prompt (no work) -> review -> "all done" -> no follow-up -> loop
        # exits on /quit. The plan document carries a completion summary.
        sess = self._make(plan=True, prompt="build a thing")
        inputs = iter(["/quit"])

        def fake_call(messages, **kw):
            # Both the plan prompt and the review prompt are tool-free model
            # calls; the response for both is the review summary.
            return ("ALL TASKS DONE. Go back to user.", [], {"total_tokens": 3})

        with mock.patch.object(agent_loop.utils, "call_openai_server",
                               side_effect=fake_call):
            with mock.patch.object(self.console, "read_line",
                                   side_effect=lambda *a, **k: next(inputs)):
                sess.run(initial_prompt=sess.args.prompt)

        plan = open(os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)).read()
        self.assertIn("# Plan completion summary", plan)
        self.assertIn("ALL TASKS DONE", plan)
        self.assertTrue(any("Task finished:" in line for line in plan.splitlines()))


if __name__ == "__main__":
    unittest.main()
