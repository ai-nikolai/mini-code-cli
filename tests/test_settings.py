"""Tests for session-local settings and the /settings command."""
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from mini_code_cli import settings, utils
from mini_code_cli.agent_loop import AgentSession
from test_e2e_agent_loop import make_args, silent_console


class SettingsStoreTest(unittest.TestCase):
    def setUp(self):
        self.cache = tempfile.mkdtemp()
        self.session = os.path.join(self.cache, "session")
        os.makedirs(self.session)

    def tearDown(self):
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_round_trip_and_private_permissions(self):
        args = make_args(self.cache, self.cache)
        args.model = "m"
        args.temperature = 0.4
        value = settings.Settings.from_args(args)
        value.values["enable_shell"] = True
        value.api_key = "sk-secret"
        value.persist_api_key = True
        path = value.save(self.session)

        self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        loaded = settings.Settings.load(self.session)
        self.assertEqual(loaded.values["model"], "m")
        self.assertTrue(loaded.values["enable_shell"])
        self.assertEqual(loaded.api_key, "sk-secret")

    def test_env_key_is_stored_as_a_reference(self):
        args = make_args(self.cache, self.cache)
        with mock.patch.dict(os.environ, {"MINI_CODE_API_KEY": "env-secret"}):
            value = settings.Settings.resolve(args)
        self.assertEqual(value.api_key, "env-secret")
        self.assertEqual(value.api_key_env, "MINI_CODE_API_KEY")
        value.save(self.session)
        with open(os.path.join(self.session, "settings.json"),
                  encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertNotIn("api_key", payload)
        self.assertEqual(payload["api_key_env"], "MINI_CODE_API_KEY")

    def test_cli_key_wins_and_is_not_persisted_by_default(self):
        args = make_args(self.cache, self.cache)
        args.api_key = "cli-secret"
        with mock.patch.dict(os.environ, {"MINI_CODE_API_KEY": "env-secret"}):
            value = settings.Settings.resolve(args)
        self.assertEqual(value.api_key, "cli-secret")
        self.assertFalse(value.persist_api_key)

    def test_previous_session_fills_missing_values(self):
        saved = settings.Settings()
        saved.values["model"] = "saved-model"
        saved.values["temperature"] = 0.2
        args = make_args(self.cache, self.cache)
        args.model = None
        args.temperature = 1.0
        value = settings.Settings.resolve(args, saved)
        self.assertEqual(value.values["model"], "saved-model")
        self.assertEqual(value.values["temperature"], 0.2)

    def test_default_plan_does_not_imply_plan_mode(self):
        args = make_args(self.cache, self.cache)
        args.plan = False
        args.plan_strategy = None
        value = settings.Settings.resolve(args)
        value.apply_to_args(args)
        self.assertFalse(args.plan)
        self.assertIsNone(args.plan_strategy)

    def test_non_tty_startup_does_not_prompt(self):
        args = make_args(self.cache, self.cache)
        args.model = None
        with mock.patch("sys.stdin.isatty", return_value=False), \
                mock.patch.object(settings, "_first_run") as first_run:
            settings.resolve_for_startup(args)
        first_run.assert_not_called()

    def test_last_session_settings_are_imported(self):
        with tempfile.TemporaryDirectory() as cache:
            session = os.path.join(cache, "previous")
            os.makedirs(session)
            saved = settings.Settings()
            saved.values["model"] = "from-previous"
            saved.save(session)
            args = make_args(cache, cache)
            args.model = None
            with mock.patch.object(settings, "find_last_session_dir",
                                   return_value=session):
                value = settings.resolve_for_startup(args)
        self.assertEqual(value.values["model"], "from-previous")

    def test_localhost_wizard_auto_detects_model(self):
        args = make_args(self.cache, self.cache)
        args.model = None
        args.prompt = None
        console = silent_console()
        console.read_line = mock.Mock(return_value="1")
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(utils, "determine_vllm_model",
                                  return_value="detected-model"):
            value = settings.resolve_for_startup(args, console=console)
        self.assertEqual(value.values["model"], "detected-model")


class SettingsCommandTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.cache = tempfile.mkdtemp()
        args = make_args(self.work, self.cache)
        self.console = silent_console()
        self.session = AgentSession(args, console=self.console)
        self.session.start_new_session()

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_plain_setting_applies_without_cache_rebuild(self):
        before = self.session.messages[0]["content"]
        self.session._handle_settings_command("temperature 0.3")
        self.assertEqual(self.session.args.temperature, 0.3)
        self.assertEqual(self.session.messages[0]["content"], before)

    def test_tool_setting_rebuilds_prompt(self):
        self.console.read_line = mock.Mock(return_value="y")
        before = self.session.messages[0]["content"]
        self.session._handle_settings_command("enable-shell on")
        self.assertTrue(self.session.args.enable_shell)
        self.assertNotEqual(self.session.messages[0]["content"], before)
        self.assertIn("run_shell_command", self.session.messages[0]["content"])

    def test_plan_strategy_change_implies_plan_mode(self):
        self.console.read_line = mock.Mock(return_value="y")
        self.session._handle_settings_command("plan-strategy plan-verify-step")
        self.assertTrue(self.session.plan_mode)
        self.assertEqual(self.session.plan_strategy, "plan-verify-step")

    def test_immutable_setting_is_rejected(self):
        self.session._handle_settings_command("tui on")
        self.assertIn("unknown setting", self.console.stream.getvalue().lower())

    def test_settings_table_marks_cache_effects(self):
        rows = {row["name"]: row for row in self.session.settings.rows()}
        self.assertEqual(rows["temperature"]["cache"], "none")
        self.assertEqual(rows["enable-shell"]["cache"], "tools")
        self.assertEqual(rows["plan"]["cache"], "prompt")

    def test_settings_prompt_accepts_number_and_value(self):
        self.console.read_line = mock.Mock(side_effect=["4 0.4", "q"])
        self.session._handle_settings_command("")
        self.assertEqual(self.session.args.temperature, 0.4)

    def test_settings_file_lives_in_session_cache(self):
        self.session._handle_settings_command("temperature 0.3")
        path = os.path.join(self.session.session_dir, "settings.json")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["values"]["temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
