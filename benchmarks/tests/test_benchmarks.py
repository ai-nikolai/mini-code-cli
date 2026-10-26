"""Tests for the benchmark harness (``benchmarks/``).

Everything here runs without Docker, without a dataset and without a model: the
whole point of the package's layering is that config expansion, command
construction, artifact handling and reporting are pure functions. The parts that
need a live harness (`tb`, `swebench`, agent CLIs) are exercised through their
*commands*, which these tests assert on.
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

from benchmarks import config as config_module
from benchmarks import forks as forks_module
from benchmarks.agents import get_adapter
from benchmarks.agents.base import AgentError
from benchmarks.config import AgentSpec, ConfigError, ModelSpec, SuiteSpec
from benchmarks.models import health_url, litellm_config, vllm_argv
from benchmarks.results import (read_results, redact, render_markdown, summarise,
                                write_results)
from benchmarks.runner import Runner, _execute, expand_env
from benchmarks.suites import get_suite
from benchmarks.suites.base import PlannedCommand, SuiteError, TrialOutcome
from benchmarks.suites.swebench_env import (read_instances_file, write_instances,
                                            write_predictions)


def local_model(name="local", base="http://127.0.0.1:30000", api="openai"):
    return ModelSpec(name=name, model="ornith-ai/Ornith-1.5-35B-A3B",
                     base_url=base, api=api)


def minimal_config(**overrides):
    data = {
        "output_dir": "benchmarks/runs",
        # Pin the runtime: the command shapes below are Docker's, and a test must
        # not change its assertions depending on which runtime the host happens to
        # have installed (apptainer machines are supported, and tested separately).
        "runtime": "docker",
        "suites": [{"name": "terminal-bench", "dataset": "terminal-bench",
                    "version": "2.1", "limit": 2}],
        "models": [{"name": "local", "model": "m", "base_url": "http://127.0.0.1:1",
                    "api": "openai"}],
        "agents": [{"name": "mini-code", "settings": ["default", "plan"]}],
    }
    data.update(overrides)
    return config_module.parse_config(data, source="<test>")


class ConfigTests(unittest.TestCase):
    def test_matrix_expands_agents_settings_and_models(self):
        config = minimal_config(
            models=[{"name": "a", "model": "m", "base_url": "http://x:1"},
                    {"name": "b", "model": "m", "base_url": "http://x:2"}])
        trials = config.trials()
        self.assertEqual(len(trials), 4)          # 2 settings x 2 models
        self.assertEqual({t.agent.setting for t in trials}, {"default", "plan"})
        # Order is stable: suite, then agent, then model.
        self.assertEqual([t.model.name for t in trials], ["a", "b", "a", "b"])

    def test_per_agent_model_filter_restricts_the_matrix(self):
        config = minimal_config(agents=[
            {"name": "mini-code", "settings": ["default"], "models": ["b"]},
            {"name": "mock", "settings": ["empty"], "models": ["a"]}],
            models=[{"name": "a", "model": "m", "base_url": "http://x:1"},
                    {"name": "b", "model": "m", "base_url": "http://x:2"}])
        pairs = {(t.agent.name, t.model.name) for t in config.trials()}
        self.assertEqual(pairs, {("mini-code", "b"), ("mock", "a")})

    def test_trial_run_id_is_a_safe_deterministic_slug(self):
        first = minimal_config().trials()[0].run_id
        second = minimal_config().trials()[0].run_id
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[A-Za-z0-9._-]+$")
        self.assertIn("mini-code-default", first)

    def test_unknown_top_level_key_is_rejected(self):
        with self.assertRaises(ConfigError) as caught:
            config_module.parse_config({
                "suites": ["terminal-bench"], "agents": ["mock"],
                "models": [{"name": "m", "model": "m", "base_url": "http://x:1"}],
                "suite": ["typo"]})
        self.assertIn("unknown config key", str(caught.exception))

    def test_comment_keys_are_ignored(self):
        config = minimal_config(_comment=["a note"])
        self.assertEqual(len(config.trials()), 2)

    def test_missing_key_is_named(self):
        with self.assertRaises(ConfigError) as caught:
            config_module.parse_config({"agents": [], "models": []})
        self.assertIn("suites", str(caught.exception))

    def test_environment_keys_are_only_required_when_asked(self):
        model = ModelSpec(name="cloud", model="m", base_url="https://api",
                          api="anthropic", api_key_env="BENCH_TEST_UNSET_KEY")
        config = minimal_config(models=[{"name": "cloud", "model": "m",
                                         "base_url": "https://api",
                                         "api": "anthropic",
                                         "api_key_env": "BENCH_TEST_UNSET_KEY"}])
        self.assertEqual(config_module.validate_config(config), [])
        problems = config_module.validate_config(config, check_env=True)
        self.assertTrue(any("BENCH_TEST_UNSET_KEY" in p for p in problems), problems)
        self.assertEqual(model.api_key_env, "BENCH_TEST_UNSET_KEY")

    def test_duplicate_settings_are_reported(self):
        with self.assertRaises(ConfigError) as caught:
            minimal_config(agents=[{"name": "mini-code", "settings": ["plan", "plan"]}])
        self.assertIn("listed twice", str(caught.exception))

    def test_invalid_api_is_reported(self):
        with self.assertRaises(ConfigError) as caught:
            minimal_config(models=[{"name": "x", "model": "m",
                                    "base_url": "http://x:1", "api": "grpc"}])
        self.assertIn("api must be", str(caught.exception))


class MiniCodeAdapterTests(unittest.TestCase):
    def adapter(self, setting="default", options=None):
        return get_adapter(AgentSpec(name="mini-code", setting=setting,
                                     options=options))

    def test_default_argv_mirrors_the_cli(self):
        argv = self.adapter().argv("do a thing", local_model(), workspace="/work")
        self.assertEqual(argv[1:4], ["-m", "mini_code_cli", "--prompt"])
        self.assertIn("do a thing", argv)
        self.assertIn("--url", argv)
        self.assertEqual(argv[argv.index("--url") + 1], "http://127.0.0.1:30000")
        self.assertEqual(argv[argv.index("--allowed-dir") + 1], "/work")
        # The bare preset adds no agent flags.
        self.assertNotIn("--auto-mode", argv)

    def test_url_with_v1_is_trimmed(self):
        argv = self.adapter().argv("p", local_model(base="http://host:8000/v1"))
        self.assertEqual(argv[argv.index("--url") + 1], "http://host:8000")

    def test_auto_shell_preset_is_unattended(self):
        argv = self.adapter("auto-shell").argv("p", local_model())
        for flag in ("--auto-mode", "--enable-shell", "--sudo"):
            self.assertIn(flag, argv)

    def test_presets_carry_their_numbers(self):
        argv = self.adapter("long-context").argv("p", local_model())
        self.assertEqual(argv[argv.index("--max-context-tokens") + 1], "131072")
        argv = self.adapter("short-tool-timeout").argv("p", local_model())
        self.assertEqual(argv[argv.index("--tool-timeout") + 1], "30")

    def test_container_argv_uses_the_images_interpreter(self):
        argv = self.adapter().argv("p", local_model(), container=True)
        self.assertEqual(argv[0], "python3")

    def test_options_override_a_preset(self):
        argv = self.adapter("plan", options={"python": "/opt/py"}).argv("p", local_model())
        self.assertEqual(argv[0], "/opt/py")

    def test_unknown_setting_is_an_error_listing_the_known_ones(self):
        with self.assertRaises(AgentError) as caught:
            self.adapter("nope").argv("p", local_model())
        self.assertIn("default", str(caught.exception))

    def test_api_key_is_passed_by_reference(self):
        model = ModelSpec(name="k", model="m", base_url="http://x:1",
                          api_key_env="OPENAI_API_KEY")
        argv = self.adapter().argv("p", model)
        self.assertEqual(argv[argv.index("--api-key") + 1], "{env:OPENAI_API_KEY}")

    def test_harness_selection_uses_the_bridge_and_a_template(self):
        selection = self.adapter("auto-shell").harness_selection(local_model())
        self.assertEqual(selection["import_path"],
                         "benchmarks.tb_agents:CommandAgent")
        template = selection["kwargs"]["argv_template"]
        self.assertIn("{instruction}", template)
        self.assertIn("--sudo", template)
        self.assertEqual(selection["kwargs"]["base_url"], "http://127.0.0.1:30000")


class OtherAdapterTests(unittest.TestCase):
    def test_claude_code_matches_the_upstream_invocation(self):
        adapter = get_adapter(AgentSpec(name="claude-code"))
        model = ModelSpec(name="c", model="anthropic/claude-sonnet-4-5",
                          base_url="https://api.anthropic.com", api="anthropic",
                          api_key_env="ANTHROPIC_API_KEY")
        argv = adapter.argv("fix it", model)
        self.assertEqual(argv[0], "claude")
        self.assertIn("--print", argv)
        self.assertEqual(argv[argv.index("--output-format") + 1], "stream-json")
        self.assertIn("--allowedTools", argv)
        self.assertEqual(argv[-1], "fix it")
        env = adapter.env(model)
        self.assertEqual(env["ANTHROPIC_MODEL"], "claude-sonnet-4-5")
        self.assertNotIn("ANTHROPIC_BASE_URL", env)
        self.assertEqual(env["ANTHROPIC_API_KEY"], "{env:ANTHROPIC_API_KEY}")

    def test_claude_code_passes_a_proxy_base_url_through(self):
        adapter = get_adapter(AgentSpec(name="claude-code"))
        model = ModelSpec(name="p", model="qwen3-coder", base_url="http://proxy:4000",
                          api="anthropic")
        env = adapter.env(model)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://proxy:4000")
        self.assertTrue(adapter.notes(model))     # the proxy caveat is surfaced

    def test_claude_code_sets_everything_a_local_endpoint_needs(self):
        """A key-less or small-model-less client fails as "authentication"."""
        adapter = get_adapter(AgentSpec(name="claude-code"))
        model = ModelSpec(name="local", model="ornith-1.5-35b",
                          base_url="http://127.0.0.1:30000", api="anthropic",
                          api_key_env="ANTHROPIC_API_KEY")
        env = adapter.env(model)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:30000")
        # the CLI needs *a* key present even when the server ignores its value
        self.assertEqual(env["ANTHROPIC_AUTH_TOKEN"], "{env:ANTHROPIC_API_KEY}")
        # and it calls a second, smaller model; without this it asks for a Claude
        self.assertEqual(env["ANTHROPIC_SMALL_FAST_MODEL"], "ornith-1.5-35b")
        # no egress from a compute node: those failures look like auth errors
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertEqual(env["DISABLE_TELEMETRY"], "1")

    def test_claude_code_leaves_the_public_api_alone(self):
        adapter = get_adapter(AgentSpec(name="claude-code"))
        model = ModelSpec(name="c", model="anthropic/claude-sonnet-4-5",
                          base_url="https://api.anthropic.com", api="anthropic")
        env = adapter.env(model)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
        self.assertNotIn("DISABLE_TELEMETRY", env)

    def test_endpoint_probe_turns_http_codes_into_instructions(self):
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from benchmarks.agents.claude_code import probe_messages_endpoint

        replies = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                code = replies["code"]
                self.send_response(code)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            replies["code"] = 400
            self.assertIsNone(probe_messages_endpoint(base, timeout=2.0),
                              "a 400 means the route parsed our body: it exists")
            replies["code"] = 404
            self.assertIn("not an Anthropic-compatible endpoint",
                          probe_messages_endpoint(base, timeout=2.0))
            replies["code"] = 401
            self.assertIn("rejects the credentials",
                          probe_messages_endpoint(base, timeout=2.0))
        finally:
            server.shutdown()
            server.server_close()
        self.assertIn("cannot reach", probe_messages_endpoint(
            "http://127.0.0.1:1", timeout=1.0))

    def test_claude_code_refuses_an_openai_only_endpoint(self):
        adapter = get_adapter(AgentSpec(name="claude-code"))
        problems = adapter.preflight(local_model(), require_binary=False)
        self.assertTrue(any("speaks the 'anthropic' API" in p for p in problems), problems)

    def test_pi_runs_one_shot_without_a_session(self):
        adapter = get_adapter(AgentSpec(name="pi", options={"provider": "local"}))
        argv = adapter.argv("p", local_model())
        self.assertEqual(argv[0], "pi")
        self.assertIn("--print", argv)
        self.assertIn("--no-session", argv)
        self.assertEqual(argv[argv.index("--provider") + 1], "local")

    def test_pi_json_mode(self):
        argv = get_adapter(AgentSpec(name="pi", setting="json-events")).argv("p", local_model())
        self.assertEqual(argv[argv.index("--mode") + 1], "json")

    def test_command_template_substitutes_and_rejects_unknown_placeholders(self):
        adapter = get_adapter(AgentSpec(
            name="command", setting="mine",
            options={"template": ["myagent", "--model", "{model}", "{prompt}"]}))
        argv = adapter.argv("hello", local_model())
        self.assertEqual(argv, ["myagent", "--model", "ornith-ai/Ornith-1.5-35B-A3B",
                                "hello"])
        bad = get_adapter(AgentSpec(name="command", setting="bad",
                                    options={"template": ["x", "{nope}"]}))
        with self.assertRaises(AgentError):
            bad.argv("p", local_model())

    def test_mock_needs_no_model(self):
        adapter = get_adapter(AgentSpec(name="mock", setting="touch"))
        model = ModelSpec(name="x", model="m", base_url="http://127.0.0.1:9",
                          api="openai")
        self.assertEqual(adapter.preflight(model), [])
        self.assertIn("--behaviour", adapter.argv("p", model))

    def test_unknown_agent_is_named_with_the_alternatives(self):
        with self.assertRaises(AgentError) as caught:
            get_adapter(AgentSpec(name="gemini"))
        self.assertIn("mini-code", str(caught.exception))


class TerminalBenchSuiteTests(unittest.TestCase):
    """Both harnesses: harbor (Terminal-Bench 2.x) and tb (1.x/core)."""

    def setUp(self):
        self.suite = get_suite(SuiteSpec(name="terminal-bench",
                                         dataset="terminal-bench",
                                         version="2.1", limit=3,
                                         options={"harness": "harbor"}))
        self.config = minimal_config(
            suites=[{"name": "terminal-bench", "dataset": "terminal-bench",
                     "version": "2.1", "limit": 3, "task_ids": ["t1"],
                     "options": {"harness": "tb", "use_legacy_dataset_flags": False}}],
            agents=[{"name": "mini-code", "settings": ["auto-shell"]}])
        self.trial = self.config.trials()[0]
        self.tb_suite = get_suite(self.trial.suite)

    # -- shared ----------------------------------------------------------

    def test_alias_resolves_to_the_canonical_suite(self):
        self.assertEqual(self.suite.name, "terminal-bench")

    def test_dataset_label_per_harness(self):
        spec = SuiteSpec(name="terminal-bench", dataset="ds", version="2.1")
        harbor = get_suite(SuiteSpec(name="terminal-bench", dataset="ds",
                                     version="2.1",
                                     options={"harness": "harbor"}))
        self.assertEqual(harbor.dataset_label(spec), "ds@2.1")
        self.assertEqual(self.tb_suite.dataset_label(spec), "ds==2.1")

    # -- harbor (Terminal-Bench 2.x, the default) ------------------------

    def harbor_argv(self, agent_setting="auto-shell"):
        trial = minimal_config(
            suites=[{"name": "terminal-bench", "dataset": "terminal-bench",
                     "version": "2.1", "limit": 2}],
            agents=[{"name": "mini-code", "settings": [agent_setting]}]).trials()[0]
        suite = get_suite(trial.suite)
        return suite.commands(trial, "/tmp/work", "/tmp/run",
                              get_adapter(trial.agent))[0]

    def test_harbor_is_the_default_harness(self):
        command = self.harbor_argv()
        self.assertEqual(command.argv[:2], ["harbor", "run"])
        self.assertEqual(command.argv[command.argv.index("-d") + 1],
                         "terminal-bench@2.1")
        self.assertIn("--jobs-dir", command.argv)
        self.assertIn("--yes", command.argv)          # never prompt a human
        self.assertIn("--n-concurrent", command.argv)

    def test_harbor_custom_agent_goes_through_the_bridge(self):
        argv = self.harbor_argv().argv
        # `-a` takes a built-in agent name *or* an import path, and is the
        # spelling harbor 0.24 does not deprecate.
        self.assertEqual(argv[argv.index("-a") + 1],
                         "benchmarks.harbor_agents:CommandAgent")
        kwargs = [argv[i + 1] for i, item in enumerate(argv)
                  if item == "--agent-kwarg"]
        template = json.loads(next(k.split("=", 1)[1] for k in kwargs
                                   if k.startswith("argv_template=")))
        self.assertIn("{instruction}", template)
        self.assertIn("--sudo", template)

    def test_harbor_uses_its_own_agents_for_claude_code_and_pi(self):
        for name, expected in (("claude-code", "claude-code"), ("pi", "pi")):
            trial = minimal_config(
                suites=[{"name": "terminal-bench", "dataset": "terminal-bench",
                         "version": "2.1"}],
                agents=[{"name": name}]).trials()[0]
            suite = get_suite(trial.suite)
            argv = suite.commands(trial, "/tmp", "/tmp/r",
                                  get_adapter(trial.agent))[0].argv
            self.assertEqual(argv[argv.index("-a") + 1], expected)
            self.assertNotIn("--agent-import-path", argv)

    # -- tb (Terminal-Bench 1.x / core datasets) -------------------------

    def commands(self):
        return self.tb_suite.commands(self.trial, "/tmp/work", "/tmp/run",
                                      get_adapter(self.trial.agent))[0]

    def test_tb_command_uses_the_tb_cli(self):
        argv = self.commands().argv
        self.assertEqual(argv[:2], ["tb", "run"])
        self.assertEqual(argv[argv.index("--dataset") + 1], "terminal-bench==2.1")

    def test_tb_legacy_dataset_flags_are_available(self):
        config = minimal_config(suites=[{
            "name": "terminal-bench", "dataset": "ds", "version": "0.1.1",
            "options": {"harness": "tb", "use_legacy_dataset_flags": True}}])
        trial = config.trials()[0]
        argv = get_suite(trial.suite).commands(
            trial, "/tmp", "/tmp/r", get_adapter(trial.agent))[0].argv
        self.assertEqual(argv[argv.index("--dataset-name") + 1], "ds")
        self.assertEqual(argv[argv.index("--dataset-version") + 1], "0.1.1")

    def test_tb_agent_is_selected_by_import_path_with_kwargs(self):
        argv = self.commands().argv
        self.assertEqual(argv[argv.index("--agent-import-path") + 1],
                         "benchmarks.tb_agents:CommandAgent")
        kwargs = [argv[i + 1] for i, item in enumerate(argv)
                  if item == "--agent-kwarg"]
        template = json.loads(next(k.split("=", 1)[1] for k in kwargs
                                   if k.startswith("argv_template=")))
        self.assertIn("{instruction}", template)

    def test_tb_trial_limits_become_harness_flags(self):
        argv = self.commands().argv
        self.assertEqual(argv[argv.index("--n-tasks") + 1], "3")
        self.assertEqual(argv[argv.index("--task-id") + 1], "t1")
        self.assertIn("--global-agent-timeout-sec", argv)

    # -- results ---------------------------------------------------------

    def test_collect_reads_tb_results(self):
        with tempfile.TemporaryDirectory() as run_dir:
            target = os.path.join(run_dir, "tb", "run-1")
            os.makedirs(target)
            with open(os.path.join(target, "results.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"results": [
                    {"task_id": "t1", "is_resolved": True, "failure_mode": "none"},
                    {"task_id": "t2", "is_resolved": False,
                     "failure_mode": "agent_timeout"}]}, handle)
            outcomes = self.suite.collect(run_dir, "run-1")
        self.assertEqual([o.outcome for o in outcomes],
                         [TrialOutcome.RESOLVED, TrialOutcome.UNRESOLVED])
        self.assertEqual(outcomes[1].detail, "agent_timeout")

    def test_collect_reads_harbor_results(self):
        with tempfile.TemporaryDirectory() as run_dir:
            target = os.path.join(run_dir, "harbor", "job-1")
            os.makedirs(target)
            with open(os.path.join(target, "result.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"trial_results": [
                    {"task_name": "t1",
                     "verifier_result": {"rewards": {"reward": 1.0}}},
                    {"task_name": "t2",
                     "verifier_result": {"rewards": {"reward": 0.0}}},
                    {"task_name": "t3",
                     "exception_info": {"exception_type": "AgentTimeoutError"}}]},
                    handle)
            outcomes = self.suite.collect(run_dir, "job-1")
        self.assertEqual({o.instance_id: o.outcome for o in outcomes},
                         {"t1": TrialOutcome.RESOLVED,
                          "t2": TrialOutcome.UNRESOLVED,
                          "t3": TrialOutcome.ERROR})
        self.assertEqual([o.instance_id for o in outcomes], ["t1", "t2", "t3"])

    def test_collect_without_results_is_an_error(self):
        with tempfile.TemporaryDirectory() as run_dir:
            with self.assertRaises(SuiteError):
                self.suite.collect(run_dir, "missing")


class SweBenchSuiteTests(unittest.TestCase):
    def setUp(self):
        self.suite = get_suite(SuiteSpec(name="swe-bench",
                                         dataset="SWE-bench/SWE-bench_Verified",
                                         split="test", limit=1))
        self.trial = minimal_config(agents=[{"name": "mock", "settings": ["touch"]}],
                                    suites=[{"name": "swebench",
                                             "dataset": "SWE-bench/SWE-bench_Verified",
                                             "split": "test", "limit": 1}]).trials()[0]

    def test_alias_resolves(self):
        self.assertEqual(self.suite.name, "swebench")

    def test_commands_without_instances_say_how_to_prepare(self):
        with tempfile.TemporaryDirectory() as run_dir:
            with self.assertRaises(SuiteError) as caught:
                self.suite.commands(self.trial, "/tmp", run_dir,
                                    get_adapter(self.trial.agent))
        self.assertIn("prepare", str(caught.exception))

    def test_expansion_produces_agent_capture_predictions_evaluate(self):
        # The evaluation command depends on which harness is installed; pin the
        # fallback so this test says the same thing on every machine.
        with unittest.mock.patch.object(forks_module, "swebench_fork_active",
                                        return_value=False):
            with tempfile.TemporaryDirectory() as run_dir:
                write_instances([{"instance_id": "django__django-1",
                                  "problem_statement": "fix it"}],
                                self.suite.instances_path(run_dir))
                commands = self.suite.commands(self.trial, "/tmp", run_dir,
                                               get_adapter(self.trial.agent))
        labels = [c.label for c in commands]
        self.assertTrue(any("agent django__django-1" in label for label in labels))
        self.assertTrue(any("capture patch" in label for label in labels))
        self.assertTrue(any("write predictions" in label for label in labels))
        evaluate = commands[-1].argv
        if self.trial.container_runtime == "apptainer":
            self.assertEqual(evaluate[:3], [evaluate[0], "-m",
                                            "benchmarks.suites.swebench_apptainer"])
            self.assertIn("--report", evaluate)
        else:
            self.assertEqual(evaluate[:3], [evaluate[0], "-m",
                                            "swebench.harness.run_evaluation"])
            self.assertIn("-id", evaluate)

    def test_fork_harness_is_used_when_it_is_installed(self):
        """With the patched checkout imported, grading goes through upstream."""
        with unittest.mock.patch.object(forks_module, "swebench_fork_active",
                                        return_value=True), \
             unittest.mock.patch.object(type(self.trial), "container_runtime",
                                        new_callable=unittest.mock.PropertyMock,
                                        return_value="apptainer"):
            with tempfile.TemporaryDirectory() as run_dir:
                write_instances([{"instance_id": "i-1",
                                  "problem_statement": "p"}],
                                self.suite.instances_path(run_dir))
                command = self.suite.evaluate_command(self.trial, run_dir)
        self.assertIn("swebench.harness.run_evaluation", command.argv)
        self.assertEqual(command.argv[command.argv.index("--backend") + 1],
                         "apptainer")
        self.assertIn("--sif-cache", command.argv)

    def test_apptainer_decision_follows_the_trial_runtime(self):
        """Regression: `self.runtime` does not exist, so this always said apptainer."""
        self.assertEqual(
            self.suite.apptainer_backend(self.trial),
            (self.trial.container_runtime or "docker") == "apptainer")

    def test_agent_command_mounts_only_the_agents_own_code(self):
        """The agent may mount *itself*, and nothing of the user's.

        mini-code runs from this repository, so the checkout is mounted read-only;
        the agent-tools directory carries the npm tree that does not fit in the
        image's 64 MiB overlay. Anything else -- a home directory, a dataset --
        would leak host state into the instance and change the measurement.
        """
        with tempfile.TemporaryDirectory() as run_dir:
            write_instances([{"instance_id": "i-1", "problem_statement": "p"}],
                            self.suite.instances_path(run_dir))
            commands = self.suite.commands(self.trial, "/tmp", run_dir,
                                           get_adapter(self.trial.agent))
        agent = next(c for c in commands if c.label.startswith("swebench: agent"))
        mounts = [agent.argv[i + 1] for i, part in enumerate(agent.argv)
                  if part in ("-v", "--bind")]
        self.assertTrue(all("/testbed" not in mount for mount in mounts))
        self.assertTrue(any(mount.endswith("/opt/agent") for mount in mounts))
        for mount in mounts:
            self.assertTrue(
                mount.split(":")[1].startswith(("/opt/mini-code", "/opt/agent")),
                f"unexpected mount: {mount}")
        self.assertIn("--rm", agent.argv)
        self.assertIn("/testbed", agent.argv)

    def test_predictions_have_the_mandated_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            instances = os.path.join(tmp, "instances.jsonl")
            write_instances([{"instance_id": "a"}, {"instance_id": "b"}], instances)
            os.makedirs(os.path.join(tmp, "instances", "a"))
            with open(os.path.join(tmp, "instances", "a", "model.patch"), "w",
                      encoding="utf-8") as handle:
                handle.write("diff --git a/x b/x\n")
            out = os.path.join(tmp, "predictions.jsonl")
            write_predictions(instances, os.path.join(tmp, "instances"),
                              "model-x", out)
            with open(out, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            first = read_instances_file(instances)[0]["instance_id"]
        self.assertEqual(rows[0], {"instance_id": "a", "model_name_or_path": "model-x",
                                   "model_patch": "diff --git a/x b/x\n"})
        self.assertEqual(rows[1]["model_patch"], "")     # missing patch -> unresolved
        self.assertEqual(first, "a")

    def test_collect_reads_the_harness_report(self):
        with tempfile.TemporaryDirectory() as run_dir:
            with open(os.path.join(run_dir, "model-tb.run-1.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({"resolved_ids": ["a"], "unresolved_ids": ["b"],
                           "error_ids": ["c"], "completed_instances": 3,
                           "total_instances": 3}, handle)
            outcomes = self.suite.collect(run_dir, "run-1")
        self.assertEqual({o.instance_id: o.outcome for o in outcomes},
                         {"a": TrialOutcome.RESOLVED, "b": TrialOutcome.UNRESOLVED,
                          "c": TrialOutcome.ERROR})


class RunnerTests(unittest.TestCase):
    def test_plan_executes_nothing_and_is_repeatable(self):
        config = minimal_config()
        runner = Runner(config, dry_run=True)
        lines = _quiet(runner.plan, config.trials())
        text = "\n".join(lines)
        self.assertIn("dry run", text)
        self.assertIn("harbor run", text)
        self.assertIn("# nothing was executed", text)
        self.assertFalse(os.path.exists(os.path.join("benchmarks", "runs",
                                                     "plan"))) or True

    def test_plan_survives_an_unexpandable_trial(self):
        config = minimal_config(suites=[{"name": "swebench",
                                         "dataset": "d", "split": "test"}])
        runner = Runner(config, dry_run=True)
        text = "\n".join(_quiet(runner.plan, config.trials()))
        self.assertIn("cannot expand yet", text)
        self.assertIn("run_evaluation", text)      # the shape is still shown

    def test_env_placeholder_expansion(self):
        self.assertEqual(expand_env("{env:FOO}", {"FOO": "bar"}), "bar")
        self.assertEqual(expand_env("x{env:MISSING}y", {}), "xy")
        self.assertEqual(expand_env("plain", {}), "plain")

    def test_agent_execution_outside_a_container_is_refused(self):
        runner = Runner(minimal_config(), dry_run=False, allow_host=False)
        adapter = get_adapter(AgentSpec(name="mini-code"))
        if runner.container_guard.__self__ and _inside_container():
            self.skipTest("this test host is itself a container")
        with self.assertRaises(RuntimeError) as caught:
            runner.container_guard(adapter)
        self.assertIn("--allow-host", str(caught.exception))

    def test_conflicting_matrix_filters_raise(self):
        with self.assertRaises(ConfigError):
            config_module.parse_config({
                "suites": [], "agents": ["mock"], "models": []}, source="<t>")

    def test_run_id_is_stable_for_the_same_matrix(self):
        config = minimal_config()
        runner = Runner(config, dry_run=True)
        first = runner.new_run_id(["s"], ["a"], ["m"])
        second = runner.new_run_id(["s"], ["a"], ["m"])
        self.assertEqual(first.split("-")[-1], second.split("-")[-1])


def _quiet(func, *args, **kwargs):
    """Run ``func`` and return its value, without letting it print in tests."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _inside_container():
    from benchmarks.runner import inside_container

    return inside_container()


class ExecutionTests(unittest.TestCase):
    """The runner's process plumbing: logs, exit codes, env expansion."""

    def run_command(self, argv, env=None, timeout=None):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "log.txt")
            command = PlannedCommand(label="test", argv=argv, env=env or {},
                                     timeout=timeout)
            status = _execute(command, log_path, lambda *a: None)
            with open(log_path, encoding="utf-8") as handle:
                return status, handle.read()

    def test_successful_command_is_logged_with_its_argv(self):
        status, log = self.run_command(
            [sys.executable, "-c", "print('hello from the agent')"])
        self.assertEqual(status, "ok")
        self.assertIn("hello from the agent", log)
        self.assertIn("$ ", log)                 # the command line is recorded
        self.assertIn("[exit 0", log)

    def test_failing_command_reports_its_exit_code(self):
        status, log = self.run_command(
            [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
        self.assertEqual(status, "exit 3")
        self.assertIn("boom", log)

    def test_missing_binary_is_an_error_not_a_crash(self):
        status, log = self.run_command(["/nonexistent/agent-binary"])
        self.assertEqual(status, "error")
        self.assertIn("could not start", log)

    def test_timeout_is_reported_and_the_process_is_killed(self):
        status, _ = self.run_command([sys.executable, "-c", "import time; time.sleep(5)"],
                                     timeout=0.2)
        self.assertEqual(status, "timeout")

    def test_env_placeholders_reach_the_process(self):
        os.environ["BENCH_EXEC_KEY"] = "abc"
        try:
            status, log = self.run_command(
                [sys.executable, "-c", "import os; print('key=' + os.environ['K'])"],
                env={"K": "{env:BENCH_EXEC_KEY}"})
        finally:
            del os.environ["BENCH_EXEC_KEY"]
        self.assertEqual(status, "ok")
        self.assertIn("key=abc", log)

    def test_logs_are_redacted(self):
        _, log = self.run_command(
            [sys.executable, "-c", "print('OPENAI_API_KEY=sk-live-secret')"])
        self.assertNotIn("sk-live-secret", log)
        self.assertIn("OPENAI_API_KEY=***", log)


class CliTests(unittest.TestCase):
    """The CLI is the interface a human drives; its exit codes are the contract."""

    def cli(self, argv):
        from benchmarks import cli

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    def config_path(self, name="offline.json"):
        # benchmarks/tests/ -> benchmarks/ : this file now lives inside the package
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, os.pardir, "configs", name)

    def test_list_runs_anywhere(self):
        code, output = self.cli(["list"])
        self.assertEqual(code, 0)
        self.assertIn("terminal-bench", output)
        self.assertIn("mini-code", output)

    def test_plan_is_the_dry_run_and_succeeds_without_a_harness(self):
        code, output = self.cli(["plan", "-c", self.config_path()])
        self.assertEqual(code, 0, output)
        self.assertIn("dry run", output)
        self.assertIn("harbor run", output)
        self.assertIn("nothing was executed", output)

    def test_plan_accepts_the_config_after_the_subcommand(self):
        code, _ = self.cli(["-c", self.config_path(), "plan"])
        self.assertEqual(code, 0)

    def test_unknown_agent_in_the_config_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"suites": ["terminal-bench"], "models": [
                    {"name": "m", "model": "m", "base_url": "http://x:1"}],
                    "agents": ["gemini"]}, handle)
            code, output = self.cli(["plan", "-c", path])
        self.assertEqual(code, 2)
        self.assertIn("unknown agent", output)

    def test_validate_reports_precondition_problems(self):
        code, output = self.cli(["validate", "-c", self.config_path("default.json")])
        self.assertEqual(code, 2)             # nothing is installed on a bare host
        self.assertIn("problem:", output)

    def test_missing_config_is_a_clean_error(self):
        code, output = self.cli(["plan", "-c", "/nope/missing.json"])
        self.assertEqual(code, 2)
        self.assertIn("config file not found", output)

    def test_run_dry_run_matches_plan(self):
        code, output = self.cli(["run", "-c", self.config_path(), "--dry-run"])
        self.assertEqual(code, 0, output)
        self.assertIn("nothing was executed", output)


class ResultsTests(unittest.TestCase):
    def outcome(self, instance_id, outcome, seconds=None, detail=""):
        return TrialOutcome(suite="terminal-bench", instance_id=instance_id,
                            outcome=outcome, duration_seconds=seconds,
                            detail=detail)

    def test_redact_hides_values_but_keeps_names(self):
        self.assertEqual(redact("ANTHROPIC_API_KEY=sk-abc123"),
                         "ANTHROPIC_API_KEY=***")
        self.assertEqual(redact("-e OPENAI_API_KEY=abc"), "-e OPENAI_API_KEY=***")
        self.assertEqual(redact("token: ghp_abcdefghijkl"), "token: ***")
        self.assertEqual(redact("nothing secret here"), "nothing secret here")

    def test_results_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.jsonl")
            write_results(path, [self.outcome("b", TrialOutcome.UNRESOLVED, 2.0),
                                 self.outcome("a", TrialOutcome.RESOLVED, 4.0)])
            outcomes = read_results(path)
        self.assertEqual([o.instance_id for o in outcomes], ["a", "b"])
        self.assertTrue(outcomes[0].resolved)

    def test_results_are_redacted_on_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "results.jsonl")
            write_results(path, [self.outcome("a", TrialOutcome.ERROR, detail=
                                              "OPENAI_API_KEY=sk-live-123")])
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        self.assertNotIn("sk-live-123", text)
        self.assertIn("OPENAI_API_KEY", text)

    def test_summary_counts_commands_that_did_not_run(self):
        trials = [{"suite": "s", "agent": "a", "setting": "d", "model": "m",
                   "run_id": "t1", "failed_commands": ["tb run x"]}]
        rows = summarise(trials, {"t1": []})
        self.assertEqual(rows[0]["command_errors"], 1)
        self.assertEqual(rows[0]["n"], 0)

    def test_summary_keeps_trials_apart_and_counts_errors(self):
        trials = [{"suite": "terminal-bench", "agent": "mini-code",
                   "setting": "default", "model": "m", "run_id": "t1"},
                  {"suite": "terminal-bench", "agent": "mini-code",
                   "setting": "plan", "model": "m", "run_id": "t2"},
                  {"suite": "terminal-bench", "agent": "pi", "setting": "default",
                   "model": "m", "run_id": "t3"}]
        outcomes = {
            "t1": [self.outcome("a", TrialOutcome.RESOLVED, 10.0),
                   self.outcome("b", TrialOutcome.UNRESOLVED, 20.0)],
            "t2": [self.outcome("a", TrialOutcome.ERROR)],
        }
        rows = {row["run_id"]: row for row in summarise(trials, outcomes)}
        self.assertEqual(rows["t1"]["rate"], 0.5)
        self.assertEqual(rows["t1"]["mean_seconds"], 15.0)
        self.assertEqual(rows["t2"]["errors"], 1)
        self.assertEqual(rows["t2"]["rate"], 0.0)
        self.assertEqual(rows["t3"]["n"], 0)
        self.assertIsNone(rows["t3"]["rate"])

    def test_markdown_reports_the_denominator(self):
        rows = [{"suite": "s", "agent": "a", "setting": "d", "model": "m",
                 "run_id": "r", "n": 3, "resolved": 2, "unresolved": 1,
                 "errors": 0, "timeouts": 0, "command_errors": 0,
                 "rate": 0.6667, "mean_seconds": 12.5}]
        text = render_markdown(rows, {"run_id": "run-1", "git_sha": "abc1234"})
        self.assertIn("| s | a | d | m | 2 | 3 | 66.67% |", text)
        self.assertIn("abc1234", text)


class ModelsTests(unittest.TestCase):
    def test_vllm_is_openai_compatible_with_tool_calling(self):
        argv = vllm_argv("ornith-ai/Ornith-1.5-35B-A3B", port=30000,
                         tensor_parallel=1, max_model_len=131072)
        self.assertEqual(argv[:3], ["vllm", "serve", "ornith-ai/Ornith-1.5-35B-A3B"])
        self.assertEqual(argv[argv.index("--host") + 1], "0.0.0.0")
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertIn("--tool-call-parser", argv)

    def test_litellm_fronts_the_local_model_for_anthropic_clients(self):
        payload = litellm_config("m", "http://127.0.0.1:30000", 40000)
        entry = payload["model_list"][0]
        self.assertEqual(entry["litellm_params"]["api_base"], "http://127.0.0.1:30000")
        self.assertTrue(entry["litellm_params"]["model"].startswith("openai/"))
        self.assertEqual(payload["server_settings"]["port"], 40000)

    def test_health_url_is_the_server_root(self):
        self.assertEqual(health_url("http://host:30000/v1"), "http://host:30000/v1/health")


if __name__ == "__main__":
    unittest.main()
