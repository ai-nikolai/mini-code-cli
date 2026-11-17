"""Tests for session-local settings and the /settings command."""
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from unittest import mock

from mini_code_cli import agent_loop, settings, tui, ui, utils
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

    def test_saved_strategy_does_not_reenable_plan_mode(self):
        saved = settings.Settings()
        saved.values["plan_strategy"] = "open-plan"
        saved.values["plan"] = False
        args = make_args(self.cache, self.cache)
        args.model = None
        args.plan = False
        args.plan_strategy = None
        value = settings.Settings.resolve(args, saved)
        self.assertFalse(value.values["plan"])
        self.assertEqual(value.values["plan_strategy"], "open-plan")

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
                                   return_value=session), \
                    mock.patch.object(utils, "list_served_models",
                                      return_value=["from-previous"]):
                value = settings.resolve_for_startup(args)
        self.assertEqual(value.values["model"], "from-previous")

    def test_list_served_models_parses_and_fails_silently(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{"id": "model-a"}, {"id": "model-b"}, {}]}
        with mock.patch.object(utils.requests, "get",
                               return_value=response):
            self.assertEqual(utils.list_served_models("http://server"),
                             ["model-a", "model-b"])
        with mock.patch.object(
                utils.requests, "get",
                side_effect=utils.requests.exceptions.RequestException("down")):
            self.assertIsNone(utils.list_served_models("http://server"))

    def test_model_is_valid_against_the_served_list(self):
        value = settings.Settings()
        value.values["model"] = "saved-model"
        with mock.patch.object(utils, "list_served_models",
                               return_value=["other", "saved-model"]):
            self.assertTrue(settings.model_is_valid(value))
        with mock.patch.object(utils, "list_served_models",
                               return_value=["other"]):
            self.assertFalse(settings.model_is_valid(value))
        with mock.patch.object(utils, "list_served_models", return_value=[]):
            self.assertFalse(settings.model_is_valid(value))
        # A failed probe is "unknown", not "invalid": keep the saved model.
        with mock.patch.object(utils, "list_served_models", return_value=None):
            self.assertTrue(settings.model_is_valid(value))

    def test_invalid_saved_model_reopens_the_wizard(self):
        saved = settings.Settings()
        saved.values["model"] = "stale-model"
        saved.save(self.session)
        args = make_args(self.cache, self.cache)
        args.model = None
        with mock.patch.object(settings, "find_last_session_dir",
                               return_value=self.session), \
                mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(utils, "list_served_models",
                                  return_value=["fresh-model"]), \
                mock.patch.object(settings, "first_run") as first_run:
            value = settings.resolve_for_startup(args, console=silent_console())
        first_run.assert_called_once()
        self.assertTrue(value.startup_asked)

    def test_valid_saved_model_does_not_reopen_the_wizard(self):
        saved = settings.Settings()
        saved.values["model"] = "saved-model"
        saved.save(self.session)
        args = make_args(self.cache, self.cache)
        args.model = None
        with mock.patch.object(settings, "find_last_session_dir",
                               return_value=self.session), \
                mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(utils, "list_served_models",
                                  return_value=["saved-model"]), \
                mock.patch.object(settings, "first_run") as first_run:
            value = settings.resolve_for_startup(args, console=silent_console())
        first_run.assert_not_called()
        self.assertFalse(value.startup_asked)

    def test_explicit_cli_model_is_not_probed_or_reasked(self):
        args = make_args(self.cache, self.cache)
        args.model = "cli-model"
        with mock.patch.object(settings, "find_last_session_dir",
                               return_value=None), \
                mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(utils, "list_served_models") as probe, \
                mock.patch.object(settings, "first_run") as first_run:
            settings.resolve_for_startup(args, console=silent_console())
        probe.assert_not_called()
        first_run.assert_not_called()

    def test_first_run_uses_the_active_tui_console(self):
        args = make_args(self.cache, self.cache)
        args.model = None
        console = tui.TuiConsole(stream=io.StringIO(), color=False,
                                 unicode=False)
        console._active = True
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(console, "choose",
                                  side_effect=[2, 1]) as choose, \
                mock.patch.object(console, "read_secret",
                                  return_value="sk-tui") as secret, \
                mock.patch.object(console, "read_line", return_value="n"), \
                mock.patch.object(utils, "list_served_models",
                                  return_value=None), \
                mock.patch.object(settings, "find_last_session_dir",
                                  return_value=None):
            value = settings.resolve_for_startup(args, console=console)

        self.assertEqual(choose.call_count, 2)
        secret.assert_called_once_with(
            "DeepSeek API key (hidden; Enter to skip): ")
        self.assertEqual(value.values["model"], "deepseek-chat")
        self.assertEqual(value.api_key, "sk-tui")

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


class _WizardConsole:
    """Records the wizard's question calls without touching a terminal."""

    def __init__(self, choice=2, key="sk-test", store="n", answers=None,
                 choice_answers=None):
        self.choice = choice
        self.key = key
        self.store = store
        self.answers = list(answers or [])
        self.choice_answers = list(choice_answers or [])
        self.choices = []
        self.secrets = []
        self.lines = []
        self.notes = []

    def choose(self, title, options, default=1, prompt="> "):
        self.choices.append((title, list(options), default))
        if self.choice_answers:
            return self.choice_answers.pop(0)
        return self.choice

    def read_secret(self, prompt_text="API key: "):
        self.secrets.append(prompt_text)
        return self.key

    def read_line(self, prompt_text="", multiline=False, tui=False):
        self.lines.append(prompt_text)
        if self.answers:
            return self.answers.pop(0)
        return self.store

    def note(self, msg, level="info"):
        self.notes.append((msg, level))


class FirstRunWizardTest(unittest.TestCase):
    def test_wizard_uses_the_console_choose_and_secret_hooks(self):
        console = _WizardConsole(choice_answers=[2, 1], key=" sk-test ",
                                 store="n")
        value = settings.Settings()
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings._first_run(value, console)

        self.assertEqual(value.values["url"], "https://api.deepseek.com")
        self.assertEqual(value.values["model"], "deepseek-chat")
        self.assertEqual(value.api_key, "sk-test")
        self.assertFalse(value.persist_api_key)
        self.assertEqual(console.choices[0][1],
                         ["localhost", "deepseek", "custom URL"])
        self.assertEqual(console.choices[1][0], "DeepSeek model to use")
        self.assertIn("deepseek-chat", console.choices[1][1])
        self.assertEqual(
            console.secrets,
            ["DeepSeek API key (hidden; Enter to skip): "])

    def test_typed_key_is_not_written_without_explicit_opt_in(self):
        console = _WizardConsole(choice_answers=[2, 1], key="sk-secret",
                                 store="n")
        value = settings.Settings()
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings._first_run(value, console)
        with tempfile.TemporaryDirectory() as session:
            path = value.save(session)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertNotIn("api_key", payload)

    def test_deepseek_model_choices_prefer_the_served_list(self):
        value = settings.Settings()
        value.api_key = "sk-test"
        with mock.patch.object(
                utils, "list_served_models",
                return_value=["deepseek-chat", "deepseek-v4"]):
            models = settings._deepseek_model_choices(value)
        self.assertEqual(models[:2], ["deepseek-chat", "deepseek-v4"])
        self.assertIn("deepseek-reasoner", models)

    def test_wizard_defaults_to_the_current_provider(self):
        value = settings.Settings()
        value.values["url"] = settings.DEEPSEEK_URL
        value.values["model"] = "deepseek-chat"
        console = _WizardConsole(choice_answers=[2, 1], key="existing")
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings._first_run(value, console)
        self.assertEqual(console.choices[0][2], 2)

    def test_custom_provider_keeps_current_values_when_blank(self):
        value = settings.Settings()
        value.values["url"] = "http://custom:8000"
        value.values["model"] = "my-model"
        value.api_key = "existing"
        console = _WizardConsole(choice=3, answers=["", ""])
        settings._first_run(value, console)
        self.assertEqual(console.choices[0][2], 3)
        self.assertEqual(value.values["url"], "http://custom:8000")
        self.assertEqual(value.values["model"], "my-model")

    def test_explicit_opt_in_stores_the_key(self):
        console = _WizardConsole(choice_answers=[2, 1], key="sk-secret",
                                 store="y")
        value = settings.Settings()
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings._first_run(value, console)
        with tempfile.TemporaryDirectory() as session:
            path = value.save(session)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        self.assertEqual(payload["api_key"], "sk-secret")


class ConsoleChooseTest(unittest.TestCase):
    def test_number_name_and_default(self):
        console = silent_console()
        console.read_line = mock.Mock(return_value="2")
        self.assertEqual(console.choose("Pick", ["a", "b"]), 2)
        console.read_line = mock.Mock(return_value="b")
        self.assertEqual(console.choose("Pick", ["a", "b"]), 2)
        console.read_line = mock.Mock(return_value="")
        self.assertEqual(console.choose("Pick", ["a", "b"], default=1), 1)
        console.read_line = mock.Mock(return_value="nonsense")
        self.assertEqual(console.choose("Pick", ["a", "b"], default=2), 2)
        self.assertIn("1. a", console.stream.getvalue())


class StartupConsoleTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.args = make_args(self.dir, self.dir)

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_build_console_follows_tui_mode(self):
        self.args.tui = "off"
        console, _mode = agent_loop.build_console(self.args)
        self.assertIs(console, ui.console)

        self.args.tui = "on"
        console, mode = agent_loop.build_console(self.args)
        self.assertIsInstance(console, tui.TuiConsole)
        self.assertEqual(mode, "on")

        self.args.tui = "auto"
        with mock.patch.object(tui.TuiConsole, "supported",
                               return_value=True):
            console, _mode = agent_loop.build_console(self.args)
        self.assertIsInstance(console, tui.TuiConsole)
        with mock.patch.object(tui.TuiConsole, "supported",
                               return_value=False):
            console, _mode = agent_loop.build_console(self.args)
        self.assertIs(console, ui.console)

    def test_main_starts_tui_before_resolving_settings(self):
        self.args.tui = "on"
        self.args.prompt = None
        self.args.no_settings = False
        events = []
        console = tui.TuiConsole(stream=io.StringIO(), color=False,
                                 unicode=False)
        console.start_tui = mock.Mock(
            side_effect=lambda: events.append("tui") or True)
        sentinel = mock.Mock()

        class FakeSession:
            def __init__(self, args, console=None, settings=None):
                self.console = console
                self.settings = settings

            def run(self, initial_prompt=None):
                events.append("run")

        with mock.patch.object(agent_loop, "parse_args",
                               return_value=self.args), \
                mock.patch.object(agent_loop, "build_console",
                                  return_value=(console, "on")), \
                mock.patch.object(agent_loop.settings_mod,
                                  "interactive_startup", return_value=True), \
                mock.patch.object(
                    agent_loop.settings_mod, "resolve_for_startup",
                    side_effect=lambda args, console=None:
                    events.append("settings") or sentinel), \
                mock.patch.object(agent_loop, "AgentSession", FakeSession):
            agent_loop.main()

        self.assertEqual(events, ["tui", "settings", "run"])

    def test_interrupt_during_first_run_stops_the_tui(self):
        self.args.tui = "on"
        self.args.prompt = None
        self.args.no_settings = False
        console = tui.TuiConsole(stream=io.StringIO(), color=False,
                                 unicode=False)
        console._active = True
        console.start_tui = mock.Mock(return_value=True)

        with mock.patch.object(agent_loop, "parse_args",
                               return_value=self.args), \
                mock.patch.object(agent_loop, "build_console",
                                  return_value=(console, "on")), \
                mock.patch.object(agent_loop.settings_mod,
                                  "interactive_startup", return_value=True), \
                mock.patch.object(agent_loop.settings_mod,
                                  "resolve_for_startup",
                                  side_effect=KeyboardInterrupt):
            agent_loop.main()

        self.assertFalse(console.active)

    def test_one_shot_prompt_never_starts_the_tui(self):
        self.args.tui = "on"
        self.args.prompt = "do it"
        self.args.no_settings = False
        console = tui.TuiConsole(stream=io.StringIO(), color=False,
                                 unicode=False)
        console.start_tui = mock.Mock(return_value=True)

        class FakeSession:
            def __init__(self, args, console=None, settings=None):
                pass

            def run(self, initial_prompt=None):
                pass

        with mock.patch.object(agent_loop, "parse_args",
                               return_value=self.args), \
                mock.patch.object(agent_loop, "build_console",
                                  return_value=(console, "on")), \
                mock.patch.object(agent_loop.settings_mod,
                                  "resolve_for_startup",
                                  return_value=mock.Mock()), \
                mock.patch.object(agent_loop, "AgentSession", FakeSession):
            agent_loop.main()

        console.start_tui.assert_not_called()


class NewSessionWizardTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.cache = tempfile.mkdtemp()
        args = make_args(self.work, self.cache)
        args.prompt = None
        self.console = silent_console()
        self.session = AgentSession(args, console=self.console)

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_fresh_session_asks_from_scratch(self):
        self.session.settings.model_explicit = False
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            self.session._ask_settings_from_scratch()
        first_run.assert_called_once()
        self.assertTrue(self.session.settings.startup_asked)

    def test_already_asked_session_does_not_ask_again(self):
        self.session.settings.model_explicit = False
        self.session.settings.startup_asked = True
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            self.session._ask_settings_from_scratch()
        first_run.assert_not_called()

    def test_explicit_model_skips_the_fresh_session_wizard(self):
        self.session.settings.model_explicit = True
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            self.session._ask_settings_from_scratch()
        first_run.assert_not_called()

    def test_background_session_never_asks(self):
        self.session.settings.model_explicit = False
        self.session.background_mode = True
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            self.session._ask_settings_from_scratch()
        first_run.assert_not_called()

    def test_fresh_session_wizard_applies_the_new_model(self):
        self.session.settings.model_explicit = False
        self.console.choose = mock.Mock(side_effect=[2, 1])
        self.console.read_secret = mock.Mock(return_value="sk-test")
        self.console.read_line = mock.Mock(return_value="n")
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(utils, "list_served_models",
                                  return_value=None):
            self.session._ask_settings_from_scratch()

        self.assertEqual(self.session.settings.values["url"],
                         settings.DEEPSEEK_URL)
        self.assertEqual(self.session.settings.values["model"],
                         "deepseek-chat")
        self.assertEqual(self.session.args.model, "deepseek-chat")
        self.assertEqual(self.session.model, "deepseek-chat")

    def test_start_new_session_calls_the_wizard(self):
        with mock.patch.object(self.session, "_ask_settings_from_scratch") \
                as ask:
            self.session.start_new_session()
        ask.assert_called_once()


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

    def test_plan_strategy_is_not_shown_as_derived(self):
        rows = {row["name"]: row for row in self.session.settings.rows()}
        self.assertNotEqual(rows["plan-strategy"]["value"], "(derived)")
        self.assertIn(rows["plan-strategy"]["raw"],
                      rows["plan-strategy"]["spec"].choices)

    def test_fixed_value_setting_uses_a_choice_not_free_text(self):
        plan_row = next(row for row in self.session.settings.rows()
                        if row["name"] == "plan")
        self.console.settings_choose = mock.Mock(return_value="on")
        self.assertEqual(self.session._read_setting_value("plan"), "on")
        args, _kwargs = self.console.settings_choose.call_args
        self.assertEqual(args[0], "plan")
        self.assertEqual(args[1], ["on", "off"])

    def test_choice_setting_uses_the_spec_choices(self):
        strategy = next(row for row in self.session.settings.rows()
                        if row["name"] == "plan-strategy")
        self.console.settings_choose = mock.Mock(return_value="open-plan")
        self.assertEqual(self.session._read_setting_value("plan-strategy"),
                         "open-plan")
        args, _kwargs = self.console.settings_choose.call_args
        self.assertEqual(args[1], list(strategy["spec"].choices))

    def test_setting_exit_returns_to_the_menu(self):
        self.console.settings_edit = mock.Mock(return_value="exit")
        self.assertIsNone(self.session._read_setting_value("temperature"))

    def test_inline_change_applies_and_refreshes_rows(self):
        rows = self.session.settings.rows()
        plan_row = next(row for row in rows if row["name"] == "plan")
        self.assertTrue(
            self.session._change_setting_inline(plan_row, True, rows))
        self.assertTrue(self.session.settings.values["plan"])
        self.assertTrue(next(row for row in rows
                             if row["name"] == "plan")["raw"])

    def test_api_key_edit_is_masked_and_not_stored_by_default(self):
        self.console.read_secret = mock.Mock(return_value="sk-new")
        self.console.settings_choose = mock.Mock(
            return_value="No — keep it in memory for this session only")
        self.console.read_line = mock.Mock(return_value="y")
        self.session._handle_settings_command("api-key")

        self.assertEqual(self.session.settings.api_key, "sk-new")
        self.assertFalse(self.session.settings.persist_api_key)
        self.console.read_secret.assert_called_once_with(
            "api-key (hidden; type 'exit' to go back): ")
        self.console.settings_choose.assert_called_once()
        path = os.path.join(self.session.session_dir, "settings.json")
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertNotIn("api_key", payload)

    def test_settings_file_lives_in_session_cache(self):
        self.session._handle_settings_command("temperature 0.3")
        path = os.path.join(self.session.session_dir, "settings.json")
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertEqual(payload["values"]["temperature"], 0.3)


if __name__ == "__main__":
    unittest.main()
