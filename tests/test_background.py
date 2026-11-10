"""Background shell/sub-agent tasks and the ``/background`` command."""

import io
import os
import shutil
import tempfile
import time
import unittest

from mini_code_cli import background, ui
from mini_code_cli.agent_loop import AgentSession
from mini_code_cli.subagents import SubAgent, clear_registry, register

from test_e2e_agent_loop import make_args, tool_call


class _Base(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp()
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        clear_registry()
        shutil.rmtree(self.workdir, ignore_errors=True)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def manager(self):
        return background.BackgroundTaskManager(
            os.path.join(self.workdir, "background"))


class BackgroundManagerTest(_Base):
    def test_shell_task_runs_to_completion_with_output_and_log(self):
        manager = self.manager()
        task = manager.start_shell("printf 'hello\\nworld\\n'")
        snapshot = manager.wait(task.id, timeout=5)

        self.assertEqual(snapshot["status"], "done")
        self.assertEqual(snapshot["exit_code"], 0)
        self.assertIn("hello", manager.tail(task.id))
        self.assertTrue(os.path.isfile(snapshot["log_path"]))
        manager.shutdown()

    def test_a_nonzero_exit_is_reported_as_failed(self):
        manager = self.manager()
        task = manager.start_shell("echo bad; exit 3")
        snapshot = manager.wait(task.id, timeout=5)

        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["exit_code"], 3)
        self.assertIn("bad", manager.tail(task.id))
        manager.shutdown()

    def test_cancel_stops_a_sleeping_shell_task(self):
        manager = self.manager()
        task = manager.start_shell("sleep 30")
        self.assertEqual(manager.wait(task.id, timeout=0.1)["status"], "running")

        manager.cancel(task.id)
        snapshot = manager.wait(task.id, timeout=5)
        self.assertEqual(snapshot["status"], "cancelled")
        manager.shutdown()

    def test_callable_task_success_and_cancellation(self):
        manager = self.manager()

        def success(should_stop):
            return "finished"

        task = manager.start_callable("quick", success)
        self.assertEqual(manager.wait(task.id, timeout=5)["result"], "finished")

        started = []

        def loop(should_stop):
            started.append(True)
            while not should_stop():
                time.sleep(0.01)
            return "stopped"

        task = manager.start_callable("loop", loop)
        deadline = time.time() + 2
        while not started and time.time() < deadline:
            time.sleep(0.01)
        manager.cancel(task.id)
        snapshot = manager.wait(task.id, timeout=5)
        self.assertEqual(snapshot["status"], "cancelled")
        manager.shutdown()

    def test_background_tool_formats_tasks_and_output(self):
        manager = self.manager()
        started = background.background_tool(
            manager, None, "start", command="printf 'one\\ntwo\\n'")
        self.assertIn("Started bg-1", started)

        listed = background.background_tool(manager, None, "list")
        self.assertIn("bg-1", listed)

        waited = background.background_tool(
            manager, None, "wait", task_id="bg-1", timeout=5)
        self.assertIn("done", waited)
        self.assertIn("two", waited)
        manager.shutdown()


class BackgroundSessionTest(_Base):
    def session(self, enable_shell=False, enable_subagents=False):
        args = make_args(self.workdir, self.cache_dir,
                         enable_subagents=enable_subagents)
        args.enable_shell = enable_shell
        console = ui.Console(stream=io.StringIO(), color=False, unicode=False,
                             verbose=False)
        session = AgentSession(args, console=console)
        session.start_new_session()
        return session

    def output(self, session):
        return session.console.stream.getvalue()

    def test_background_tool_is_only_registered_with_shell_or_subagents(self):
        self.assertNotIn("background", self.session().tools_dict)
        self.assertIn("background",
                      self.session(enable_shell=True).tools_dict)
        self.assertIn("background",
                      self.session(enable_subagents=True).tools_dict)

    def test_background_shell_task_flows_through_the_tool_and_command(self):
        session = self.session(enable_shell=True)
        session._run_one_tool_call(tool_call(
            "c1", "background", {"action": "start",
                                 "command": "printf 'hello\\n'"}))
        self.assertIn("Started bg-1", session.messages[-1]["content"])

        snapshot = session.background_tasks.wait("bg-1", timeout=5)
        self.assertEqual(snapshot["status"], "done")

        self.assertTrue(session._handle_command("/background"))
        out = self.output(session)
        self.assertIn("Background", out)
        self.assertIn("bg-1", out)

        self.assertTrue(session._handle_command("/background bg-1"))
        out = self.output(session)
        self.assertIn("hello", out)
        self.assertIn("output (tail):", out)
        session.shutdown()

    def test_background_shell_uses_the_safety_classifier(self):
        session = self.session(enable_shell=True)
        session._run_one_tool_call(tool_call(
            "c1", "background",
            {"action": "start", "command": "rm -rf /"}))
        result = session.messages[-1]["content"].lower()
        self.assertIn("refused", result)
        self.assertEqual(session.background_tasks.list(), [])
        session.shutdown()

    def test_background_subagent_task_returns_its_result(self):
        class Quick(SubAgent):
            name = "quick"
            description = "Returns immediately."

            def run(self, task, parent, max_steps=None, should_stop=None,
                    **options):
                return f"done: {task}"

        register(Quick())
        session = self.session(enable_subagents=True)
        session._run_one_tool_call(tool_call(
            "c1", "background",
            {"action": "start", "kind": "subagent", "subagent": "quick",
             "task": "do it"}))
        snapshot = session.background_tasks.wait("bg-1", timeout=5)

        self.assertEqual(snapshot["status"], "done")
        self.assertIn("done: do it", snapshot["result"])
        self.assertTrue(session._handle_command("/background bg-1"))
        self.assertIn("done: do it", self.output(session))
        session.shutdown()

    def test_background_command_shows_foreground_activity(self):
        session = self.session(enable_shell=True)
        session._begin_activity("waiting for LLM response")
        self.assertTrue(session._handle_command("/background"))
        out = self.output(session)
        self.assertIn("Foreground : waiting for LLM response", out)
        self.assertIn("Background : none", out)
        session._end_activity()
        session.shutdown()

    def test_shutdown_cancels_a_running_task(self):
        session = self.session(enable_shell=True)
        task = session.background_tasks.start_shell("sleep 30")
        session.shutdown()
        self.assertEqual(
            session.background_tasks.wait(task.id, timeout=5)["status"],
            "cancelled")
