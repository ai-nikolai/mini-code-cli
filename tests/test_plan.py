"""Tests for the --plan mode feature (plan.py + its wiring into agent_loop.py).

Covers:
  * the PlanManager (write/read goal + plan, log requests, mark finished) in
    isolation,
  * the plan-mode system-prompt instructions,
  * the loop wiring: recording each user request/goal in the tracking files,
    the plan re-injection that runs after a history compaction, and marking the
    goal finished on shutdown.
  * the fact that the tracking files are loop-owned: the model authors none of
    them, so a generic write_file of a plan file (which lives outside
    --allowed-dir) is still blocked by the sandbox.
"""
import io
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
        # Agent-authored plan first (dir must exist for us to write into it).
        os.makedirs(self.session, exist_ok=True)
        open(pm.plan_path, "w").write("# Current Plan\n1. do A\n")
        pm.mark_finished("goal Z")
        text = open(pm.plan_path).read()
        self.assertIn("1. do A", text)
        self.assertIn("Task finished: goal Z", text)

    def test_mark_finished_reads_goal_from_disk(self):
        # If the loop never recorded a goal in memory, mark_finished still names
        # the right goal by reading CURRENT_GOAL.md off disk.
        pm = plan_utils.PlanManager(self.session, enabled=True)
        pm.write_goal("the real goal")
        pm.mark_finished()
        self.assertIn("Task finished: the real goal", open(pm.plan_path).read())

    def test_read_plan_returns_none_when_absent(self):
        pm = plan_utils.PlanManager(self.session, enabled=True)
        self.assertIsNone(pm.read_plan())


class TestPlanPrompt(unittest.TestCase):
    def test_plan_mode_instructions_present(self):
        text = prompts.plan_instructions()
        self.assertIn("CURRENT_GOAL.md", text)
        self.assertIn("CURRENT_PLAN.md", text)
        self.assertIn("Task finished", text)

    def test_system_prompt_contains_plan_block_only_when_enabled(self):
        with_plan = prompts.system_prompt(tools_dict=None, allowed_dir=".", plan_mode=True)
        without = prompts.system_prompt(tools_dict=None, allowed_dir=".", plan_mode=False)
        self.assertIn("PLAN MODE", with_plan)
        self.assertIn("CURRENT_PLAN.md", with_plan)
        self.assertNotIn("PLAN MODE", without)


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

    def test_record_goal_request_logs_to_plan(self):
        sess = self._make(plan=True)
        sess._record_goal_request("do the thing")
        plan = open(os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)).read()
        self.assertIn("do the thing", plan)

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
        # Simulate the agent having authored a plan on disk.
        open(self.sess.plan_manager.plan_path, "w").write("# Current Plan\n1. do X\n")
        out = self.sess._apply_plan_to_compacted(
            [{"role": "system", "content": "sys"}])
        texts = [m["content"] for m in out]
        self.assertTrue(any("current plan" in t.lower() for t in texts))
        self.assertTrue(any("Build the thing" in t for t in texts))
        # Goal must be the last injected instruction.
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
        self.assertEqual(len(out), 1)  # nothing injected, message preserved


if __name__ == "__main__":
    unittest.main()
