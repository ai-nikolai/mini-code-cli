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
        value.use_key("sk-secret", persist=True)
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
        kid = settings.key_id(value.values["url"], value.values["model"])
        self.assertNotIn("api_key", payload)
        self.assertEqual(payload["api_key_envs"][kid],
                         "MINI_CODE_API_KEY")

    def test_legacy_single_key_migrates_to_the_model_map(self):
        payload = {
            "version": 1,
            "values": {"url": "https://api.deepseek.com",
                       "model": "deepseek-chat"},
            "api_key": "legacy-key",
        }
        path = os.path.join(self.session, "settings.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        loaded = settings.Settings.load(self.session)
        kid = settings.key_id("https://api.deepseek.com", "deepseek-chat")
        self.assertEqual(loaded.api_key, "legacy-key")
        self.assertEqual(loaded.api_keys[kid], "legacy-key")
        self.assertIn(kid, loaded.persisted_key_ids)

    def test_keys_are_stored_per_endpoint_and_model(self):
        value = settings.Settings()
        value.values["url"] = "https://api.example.test"
        value.values["model"] = "model-a"
        value.use_key("key-a", persist=True)
        value.values["model"] = "model-b"
        value.use_key("key-b", persist=True)
        value.save(self.session)

        with open(os.path.join(self.session, "settings.json"),
                  encoding="utf-8") as handle:
            payload = json.load(handle)
        kid_a = settings.key_id("https://api.example.test", "model-a")
        kid_b = settings.key_id("https://api.example.test", "model-b")
        self.assertEqual(payload["api_keys"][kid_a], "key-a")
        self.assertEqual(payload["api_keys"][kid_b], "key-b")

        loaded = settings.Settings.load(self.session)
        loaded.values["model"] = "model-a"
        self.assertTrue(loaded.load_key_for())
        self.assertEqual(loaded.api_key, "key-a")
        loaded.values["model"] = "model-b"
        self.assertTrue(loaded.load_key_for())
        self.assertEqual(loaded.api_key, "key-b")

    def test_saved_key_is_selected_by_model(self):
        saved = settings.Settings()
        saved.values["url"] = "https://api.example.test"
        saved.values["model"] = "model-a"
        saved.use_key("key-a", persist=True)
        saved.values["model"] = "model-b"
        saved.use_key("key-b", persist=True)

        args = make_args(self.cache, self.cache)
        args.url = "https://api.example.test"
        args.model = "model-b"
        value = settings.Settings.resolve(args, saved)
        self.assertEqual(value.api_key, "key-b")

        args.model = "model-c"
        value = settings.Settings.resolve(args, saved)
        self.assertEqual(value.api_key, "")

    def test_save_does_not_copy_a_key_to_a_new_model(self):
        value = settings.Settings()
        value.values["url"] = "https://api.example.test"
        value.values["model"] = "model-a"
        value.use_key("key-a", persist=True)
        value.values["model"] = "model-b"
        value.save(self.session)

        with open(os.path.join(self.session, "settings.json"),
                  encoding="utf-8") as handle:
            payload = json.load(handle)
        kid_a = settings.key_id("https://api.example.test", "model-a")
        kid_b = settings.key_id("https://api.example.test", "model-b")
        self.assertIn(kid_a, payload["api_keys"])
        self.assertNotIn(kid_b, payload["api_keys"])

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

    def test_no_settings_does_not_prompt_or_save(self):
        args = make_args(self.cache, self.cache)
        args.no_settings = True
        args.prompt = None
        args.model = None
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            value = settings.resolve_for_startup(args,
                                                console=silent_console())
        first_run.assert_not_called()
        self.assertFalse(value.enabled)
        self.assertIsNone(value.save(self.session))

    def test_non_tty_startup_does_not_prompt(self):
        args = make_args(self.cache, self.cache)
        args.model = None
        with mock.patch("sys.stdin.isatty", return_value=False), \
                mock.patch.object(settings, "first_run") as first_run:
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

    def test_probe_api_key_recognizes_success_and_rejection(self):
        ok = mock.Mock(status_code=200)
        with mock.patch.object(utils.requests, "get", return_value=ok):
            self.assertTrue(utils.probe_api_key("http://server", "sk"))
        denied = mock.Mock(status_code=401)
        with mock.patch.object(utils.requests, "get", return_value=denied):
            self.assertFalse(utils.probe_api_key("http://server", "sk"))
        with mock.patch.object(
                utils.requests, "get",
                side_effect=utils.requests.exceptions.RequestException("down")):
            self.assertIsNone(utils.probe_api_key("http://server", "sk"))

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
                mock.patch.object(utils, "probe_api_key",
                                  return_value=True), \
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
    def setUp(self):
        # The wizard validates entered keys against /v1/models; tests do not
        # talk to a real server.
        patcher = mock.patch.object(utils, "probe_api_key",
                                    return_value=True)
        self.probe_api_key = patcher.start()
        self.addCleanup(patcher.stop)

    def test_wizard_uses_the_console_choose_and_secret_hooks(self):
        console = _WizardConsole(choice_answers=[2, 1], key=" sk-test ",
                                 store="n")
        value = settings.Settings()
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings.first_run(value, console)

        self.assertEqual(value.values["url"], "https://api.deepseek.com")
        self.assertEqual(value.values["model"], "deepseek-chat")
        self.assertEqual(value.api_key, "sk-test")
        self.assertFalse(value.persist_api_key)
        self.assertEqual(
            value.key_for(settings.DEEPSEEK_URL, "deepseek-chat")[0],
            "sk-test")
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
            settings.first_run(value, console)
        with tempfile.TemporaryDirectory() as session:
            path = value.save(session)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        self.assertNotIn("api_key", payload)
        self.assertNotIn("api_keys", payload)

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
            settings.first_run(value, console)
        self.assertEqual(console.choices[0][2], 2)

    def test_custom_provider_keeps_current_values_when_blank(self):
        value = settings.Settings()
        value.values["url"] = "http://custom:8000"
        value.values["model"] = "my-model"
        value.api_key = "existing"
        console = _WizardConsole(choice=3, answers=["", ""])
        settings.first_run(value, console)
        self.assertEqual(console.choices[0][2], 3)
        self.assertEqual(value.values["url"], "http://custom:8000")
        self.assertEqual(value.values["model"], "my-model")

    def test_deepseek_reuses_a_stored_key_for_the_selected_model(self):
        value = settings.Settings()
        kid = settings.key_id(settings.DEEPSEEK_URL, "deepseek-chat")
        value.api_keys[kid] = "stored-key"
        value.persisted_key_ids.add(kid)
        console = _WizardConsole(choice_answers=[2, 1],
                                 key="should-not-be-asked")
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings.first_run(value, console)

        self.assertEqual(value.api_key, "stored-key")
        self.assertEqual(console.secrets, [])

    def test_selecting_a_model_loads_its_stored_key(self):
        value = settings.Settings()
        value.values["url"] = settings.DEEPSEEK_URL
        value.values["model"] = "deepseek-chat"
        value.use_key("key-chat", persist=True)
        value.values["model"] = "deepseek-reasoner"
        value.use_key("key-reasoner", persist=True)
        value.values["model"] = "deepseek-chat"
        value.api_key = ""
        value.key_entered = False
        console = _WizardConsole(key="should-not-be-used")

        settings._finish_key_for_model(value, console, "deepseek")

        self.assertEqual(value.api_key, "key-chat")
        self.assertEqual(console.secrets, [])

    def test_a_different_model_without_a_stored_key_asks_again(self):
        value = settings.Settings()
        value.values["url"] = settings.DEEPSEEK_URL
        value.values["model"] = "deepseek-chat"
        value.use_key("key-chat", persist=True)
        value.values["model"] = "deepseek-v4"
        value.api_key = "key-chat"
        value.key_entered = False
        console = _WizardConsole(key="key-v4", store="n")

        settings._finish_key_for_model(value, console, "deepseek")

        self.assertEqual(value.api_key, "key-v4")
        self.assertEqual(
            value.key_for(settings.DEEPSEEK_URL, "deepseek-v4")[0],
            "key-v4")
        self.assertEqual(len(console.secrets), 1)

    def test_rejected_key_is_asked_again(self):
        value = settings.Settings()
        value.values["url"] = settings.DEEPSEEK_URL
        value.values["model"] = "deepseek-chat"
        value.use_key("bad-key", persist=True)
        console = _WizardConsole(key="good-key", store="n")
        with mock.patch.object(utils, "probe_api_key",
                               side_effect=[False, True]):
            settings._validate_key(console, value, "deepseek")

        self.assertEqual(value.api_key, "good-key")
        self.assertEqual(len(console.secrets), 1)
        self.assertEqual(
            value.key_for(settings.DEEPSEEK_URL, "deepseek-chat")[0],
            "good-key")

    def test_explicit_opt_in_stores_the_key(self):
        console = _WizardConsole(choice_answers=[2, 1], key="sk-secret",
                                 store="y")
        value = settings.Settings()
        with mock.patch.object(utils, "list_served_models",
                               return_value=None):
            settings.first_run(value, console)
        with tempfile.TemporaryDirectory() as session:
            path = value.save(session)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
        kid = settings.key_id(value.values["url"], value.values["model"])
        self.assertEqual(payload["api_keys"][kid], "sk-secret")


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

    def test_main_defers_settings_and_model_until_run(self):
        events = []

        class FakeSession:
            def __init__(self, args, defer_model=False):
                events.append(("init", defer_model))

            def run(self, initial_prompt=None):
                events.append(("run", initial_prompt))

        with mock.patch.object(agent_loop, "parse_args",
                               return_value=self.args), \
                mock.patch.object(agent_loop, "AgentSession", FakeSession), \
                mock.patch.object(
                    agent_loop.settings_mod, "resolve_for_startup",
                    side_effect=AssertionError(
                        "settings loaded before the session decision")):
            agent_loop.main()

        self.assertEqual(events, [("init", True), ("run", None)])


class NewSessionWizardTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp()
        self.cache = tempfile.mkdtemp()
        self.args = make_args(self.work, self.cache)
        self.args.prompt = None
        self.console = silent_console()

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)
        shutil.rmtree(self.cache, ignore_errors=True)

    def _saved_session(self, model="saved-model"):
        session = os.path.join(self.cache, "previous")
        os.makedirs(session)
        saved = settings.Settings()
        saved.values["model"] = model
        saved.save(session)
        return session

    def test_new_session_resolution_asks_from_scratch(self):
        self.args.model = None
        source = self._saved_session()
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            value = settings.resolve_for_startup(
                self.args, console=self.console, source_dir=source,
                new_session=True)
        first_run.assert_called_once()
        self.assertTrue(value.startup_asked)

    def test_new_session_resolution_skips_wizard_for_explicit_model(self):
        self.args.model = "explicit-model"
        with mock.patch("sys.stdin.isatty", return_value=True), \
                mock.patch.object(settings, "first_run") as first_run:
            settings.resolve_for_startup(
                self.args, console=self.console, source_dir=None,
                new_session=True)
        first_run.assert_not_called()

    def test_new_session_resets_settings_but_keeps_the_api_key(self):
        first_args = make_args(self.work, self.cache)
        first_args.model = "previous-model"
        first_args.temperature = 0.5
        first_args.max_tokens = 1000
        first_args.verbose = True
        first = AgentSession(first_args, console=self.console)
        first.start_new_session()
        first.settings.use_key("previous-key", persist=True)
        first._save_settings()

        second_args = make_args(self.work, self.cache)
        second_args.model = "previous-model"
        second_console = silent_console()
        second = AgentSession(second_args, console=second_console,
                              defer_model=True)
        self.assertFalse(second_console.verbose)

        second.start_new_session()
        self.assertEqual(second.model, "previous-model")
        self.assertEqual(second.settings.values["temperature"], 1.0)
        self.assertNotEqual(second.args.max_tokens, 1000)
        self.assertFalse(second.settings.values["verbose"])
        self.assertFalse(second_console.verbose)
        self.assertEqual(second.api_key, "previous-key")

    def test_restore_adopts_the_restored_session_settings(self):
        first_args = make_args(self.work, self.cache)
        first_args.model = "restored-model"
        first = AgentSession(first_args, console=self.console)
        first.start_new_session()

        second_args = make_args(self.work, self.cache)
        second_args.model = None
        second = AgentSession(second_args, console=self.console,
                              defer_model=True)
        self.assertTrue(second._try_restore(first.session_dir))
        self.assertEqual(second.model, "restored-model")

    def test_background_session_never_loads_previous_settings(self):
        session = AgentSession(self.args, console=self.console,
                               background_mode=True)
        with mock.patch.object(
                agent_loop, "find_last_session_dir",
                side_effect=AssertionError("background loaded previous settings")), \
                mock.patch.object(settings, "resolve_for_startup",
                                  return_value=mock.Mock()) as resolve, \
                mock.patch.object(session, "_adopt_settings"):
            session.start_new_session()

        kwargs = resolve.call_args.kwargs
        self.assertIsNone(kwargs["source_dir"])
        self.assertFalse(kwargs["ask"])
        self.assertTrue(kwargs["new_session"])

    def test_start_new_session_adopts_resolved_settings(self):
        session = AgentSession(self.args, console=self.console)
        sentinel = mock.Mock()
        with mock.patch.object(agent_loop, "find_last_session_dir",
                               return_value=None), \
                mock.patch.object(settings, "resolve_for_startup",
                                  return_value=sentinel) as resolve, \
                mock.patch.object(session, "_adopt_settings") as adopt:
            session.start_new_session()

        kwargs = resolve.call_args.kwargs
        self.assertTrue(kwargs["new_session"])
        self.assertIsNone(kwargs["source_dir"])
        adopt.assert_called_once_with(sentinel)


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

    def test_settings_default_preset_keeps_the_api_key(self):
        self.session.settings.use_key("keep-key", persist=False)
        self.session._handle_settings_command("temperature 0.3")
        with mock.patch.object(utils, "determine_vllm_model",
                               return_value=None):
            self.session._handle_settings_command("default")

        self.assertEqual(self.session.args.temperature,
                         settings.DEFAULT_TEMPERATURE)
        self.assertEqual(self.session.settings.api_key, "keep-key")

    def test_plan_strategy_change_does_not_enable_plan_mode(self):
        self.console.read_line = mock.Mock(return_value="y")
        self.session._handle_settings_command("plan-strategy plan-verify-step")
        self.assertFalse(self.session.plan_mode)
        self.assertEqual(self.session.settings.values["plan_strategy"],
                         "plan-verify-step")

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

    def test_default_values_are_marked_in_the_table(self):
        rows = {row["name"]: row for row in self.session.settings.rows()}
        for name in ("temperature", "plan", "plan-strategy"):
            self.assertTrue(rows[name]["is_default"], name)
            self.assertIn("(default)", rows[name]["display"], name)
        # "(not set)" is already explicit enough for an absent API key.
        self.assertNotIn("(default)", rows["api-key"]["display"])

    def test_inline_plan_strategy_change_keeps_plan_mode_off(self):
        snapshot = self.session._snapshot_settings()
        rows = self.session.settings.rows()
        strategy = next(row for row in rows
                        if row["name"] == "plan-strategy")
        self.session._change_setting_inline(strategy, "open-plan", rows)
        self.assertFalse(self.session.settings.values["plan"])
        self.assertFalse(self.session.plan_mode)

        self.session._commit_staged_settings(snapshot)
        self.assertFalse(self.session.plan_mode)
        self.assertEqual(self.session.settings.values["plan_strategy"],
                         "open-plan")

    def test_changing_a_value_removes_the_default_marker(self):
        rows = self.session.settings.rows()
        temperature = next(row for row in rows
                           if row["name"] == "temperature")
        self.session._change_setting_inline(temperature, 0.5, rows)
        fresh = next(row for row in rows
                     if row["name"] == "temperature")
        self.assertFalse(fresh["is_default"])
        self.assertNotIn("(default)", fresh["display"])

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

    def test_inline_change_stages_until_saved(self):
        snapshot = self.session._snapshot_settings()
        rows = self.session.settings.rows()
        plan_row = next(row for row in rows if row["name"] == "plan")
        self.assertTrue(
            self.session._change_setting_inline(plan_row, True, rows))
        self.assertTrue(self.session.settings.values["plan"])
        self.assertFalse(self.session.plan_mode)
        self.assertTrue(next(row for row in rows
                             if row["name"] == "plan")["raw"])

        self.assertTrue(self.session._commit_staged_settings(snapshot))
        self.assertTrue(self.session.plan_mode)

    def test_snapshot_after_save_keeps_the_saved_state_on_cancel(self):
        snapshot = self.session._snapshot_settings()
        rows = self.session.settings.rows()
        plan_row = next(row for row in rows if row["name"] == "plan")
        self.session._change_setting_inline(plan_row, True, rows)
        self.assertTrue(self.session._commit_staged_settings(snapshot))
        snapshot = self.session._snapshot_settings()

        rows = self.session.settings.rows()
        plan_row = next(row for row in rows if row["name"] == "plan")
        self.session._change_setting_inline(plan_row, False, rows)
        self.session._restore_settings(snapshot)
        self.assertTrue(self.session.settings.values["plan"])

    def test_cancelled_staged_changes_restore_the_snapshot(self):
        snapshot = self.session._snapshot_settings()
        rows = self.session.settings.rows()
        plan_row = next(row for row in rows if row["name"] == "plan")
        self.session._change_setting_inline(plan_row, True, rows)
        self.assertTrue(self.session.settings.values["plan"])

        self.session._restore_settings(snapshot)
        self.assertFalse(self.session.settings.values["plan"])
        self.assertFalse(self.session.plan_mode)

    def test_explicit_key_survives_a_model_change(self):
        self.session.settings.use_key("explicit-key", persist=False)
        kid = settings.key_id(self.session.settings.values["url"],
                              "different-model")
        self.session.settings.api_keys[kid] = "stored-key"
        self.session.settings.persisted_key_ids.add(kid)
        self.assertTrue(self.session._apply_settings(
            {"model": "different-model"}, confirm_cache=False,
            announce=False, stage=True))
        self.assertEqual(self.session.settings.api_key, "explicit-key")

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
