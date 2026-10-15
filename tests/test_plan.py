"""Tests for the --plan mode feature (plan.py + its wiring into agent_loop.py).

Covers:
  * the PlanManager (write/read goal + plan, mark finished) in isolation,
  * the plan-mode system-prompt instructions,
  * the loop wiring: goal seeding, plan-file sandbox exemption, and the
    plan re-injection that runs after a history compaction.
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

    def test_capture_goal_writes_goal_file(self):
        sess = self._make(plan=True)
        sess._capture_goal("Build the thing")
        self.assertIn("Build the thing",
                      open(os.path.join(sess.plan_manager.session_dir, plan_utils.GOAL_FILENAME)).read())

    def test_capture_goal_ignores_slash_commands(self):
        sess = self._make(plan=True)
        sess._capture_goal("/history")
        self.assertIsNone(sess.plan_manager.read_goal())

    def test_capture_goal_noop_when_disabled(self):
        sess = self._make(plan=False)
        sess._capture_goal("do x")
        self.assertIsNone(sess.plan_manager.read_goal())

    def test_capture_goal_ignores_empty_input(self):
        sess = self._make(plan=True)
        sess._capture_goal("   ")
        self.assertIsNone(sess.plan_manager.read_goal())

    def test_is_plan_file_true_for_plan_files(self):
        sess = self._make(plan=True)
        p = os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)
        self.assertTrue(sess._is_plan_file("write_file", {"file_path": p}))
        g = os.path.join(sess.plan_manager.session_dir, plan_utils.GOAL_FILENAME)
        self.assertTrue(sess._is_plan_file("read_file", {"file_path": g}))

    def test_is_plan_file_false_for_other_files(self):
        sess = self._make(plan=True)
        other = os.path.join(sess.plan_manager.session_dir, "notes.txt")
        self.assertFalse(sess._is_plan_file("write_file", {"file_path": other}))

    def test_is_plan_file_false_when_disabled(self):
        sess = self._make(plan=False)
        p = os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)
        self.assertFalse(sess._is_plan_file("write_file", {"file_path": p}))

    def test_is_plan_file_false_for_write_outside_session_dir(self):
        sess = self._make(plan=True)
        # Same filename, but a different directory must not be treated as a plan file.
        elsewhere = os.path.join(self.tmp.name, plan_utils.PLAN_FILENAME)
        self.assertFalse(sess._is_plan_file("write_file", {"file_path": elsewhere}))

    def test_mark_finished_and_shutdown(self):
        sess = self._make(plan=True)
        sess._capture_goal("my goal")
        with mock.patch.object(sess, "shutdown"):
            sess._finish_goal_and_shutdown()
        self.assertIn("Task finished: my goal",
                      open(os.path.join(sess.plan_manager.session_dir, plan_utils.PLAN_FILENAME)).read())

    def test_mark_finished_noop_when_disabled(self):
        sess = self._make(plan=False)
        sess._mark_current_goal_finished()  # must not raise
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, sess.unique_id, plan_utils.PLAN_FILENAME)))


class TestSandboxExemptsPlanFiles(unittest.TestCase):
    """A plan-file write must pass the sandbox even though the session dir is
    (usually) outside --allowed-dir."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.buf = io.StringIO()
        self.console = ui.Console(stream=self.buf, color=False, unicode=False)

    def _run_tool(self, sess, call):
        sess._run_one_tool_call(call)

    def test_plan_file_write_allowed_even_outside_allowed_dir(self):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name, allowed_dir="/tmp", plan=True),
                                           console=self.console)
        sdir = sess.plan_manager.session_dir
        call = {"id": "c1", "function": {"name": "write_file",
                "arguments": '{"file_path": "%s/%s", "content": "plan"}' %
                             (sdir, plan_utils.PLAN_FILENAME)}}
        with mock.patch.object(sess, "_run_one_tool_call", wraps=sess._run_one_tool_call) as spy:
            sess._run_one_tool_call(call)
        # The tool actually ran (file written) instead of being blocked by the sandbox.
        self.assertTrue(os.path.exists(os.path.join(sdir, plan_utils.PLAN_FILENAME)))

    def test_non_plan_file_write_still_blocked(self):
        with mock.patch.object(agent_loop.utils, "determine_vllm_model", return_value=None):
            sess = agent_loop.AgentSession(_make_args(cache_dir=self.tmp.name, allowed_dir=self.tmp.name, plan=True),
                                           console=self.console)
        outside = os.path.join(os.path.dirname(self.tmp.name), "secret.txt")
        call = {"id": "c1", "function": {"name": "write_file",
                "arguments": '{"file_path": "%s", "content": "x"}' % outside}}
        sess._run_one_tool_call(call)
        # A blocked write appends a refusal tool-result and writes nothing.
        last = sess.messages[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("not allowed outside allowed directory", last["content"])


class TestPlanCompactionReinjectsPlan(unittest.TestCase):
    """After compaction the plan + goal must be re-appended to the messages."""

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
