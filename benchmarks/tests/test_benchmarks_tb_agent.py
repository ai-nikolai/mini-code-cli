"""Tests for the Terminal-Bench installed-agent bridge (``benchmarks/tb_agents.py``).

That module imports :mod:`terminal_bench` at module level on purpose (it is only
ever loaded inside a ``tb`` process), so these tests install a minimal stand-in
for the two symbols it needs. That is enough to test the part that actually
matters and that cannot be exercised any other way: how the task instruction is
quoted into a shell command, and which environment the agent gets in the
container.
"""
import importlib
import json
import os
import shlex
import sys
import types
import unittest


def _install_terminal_bench_stub():
    """Register the smallest ``terminal_bench`` surface ``tb_agents`` imports."""
    if "terminal_bench" in sys.modules:
        return

    package = types.ModuleType("terminal_bench")
    agents = types.ModuleType("terminal_bench.agents")
    installed = types.ModuleType("terminal_bench.agents.installed_agents")
    abstract = types.ModuleType(
        "terminal_bench.agents.installed_agents.abstract_installed_agent")
    terminal = types.ModuleType("terminal_bench.terminal")
    models = types.ModuleType("terminal_bench.terminal.models")

    class AbstractInstalledAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class TerminalCommand:
        """Mirrors the real dataclass' signature (command + session flags)."""

        def __init__(self, command, min_timeout_sec=0.0, max_timeout_sec=0.0,
                     block=False, append_enter=False):
            self.command = command
            self.min_timeout_sec = min_timeout_sec
            self.max_timeout_sec = max_timeout_sec
            self.block = block
            self.append_enter = append_enter

    abstract.AbstractInstalledAgent = AbstractInstalledAgent
    models.TerminalCommand = TerminalCommand

    modules = {
        "terminal_bench": package,
        "terminal_bench.agents": agents,
        "terminal_bench.agents.installed_agents": installed,
        "terminal_bench.agents.installed_agents.abstract_installed_agent": abstract,
        "terminal_bench.terminal": terminal,
        "terminal_bench.terminal.models": models,
    }
    sys.modules.update(modules)


class CommandAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_terminal_bench_stub()
        cls.module = importlib.import_module("benchmarks.tb_agents")

    def build(self, template=None, **kwargs):
        template = template or ["python3", "-m", "mini_code_cli",
                                "--prompt", "{instruction}"]
        return self.module.CommandAgent(argv_template=template, **kwargs)

    def test_instruction_is_quoted_not_interpreted(self):
        agent = self.build()
        (command,) = agent._run_agent_commands("rm -rf / ; echo $(whoami)")
        # Every argument is quoted by shlex.join, so the whole instruction stays
        # one argument to --prompt.
        self.assertIn("'rm -rf / ; echo $(whoami)'", command.command)
        self.assertEqual(shlex.split(command.command),
                         ["python3", "-m", "mini_code_cli", "--prompt",
                          "rm -rf / ; echo $(whoami)"])
        self.assertTrue(command.block)
        self.assertTrue(command.append_enter)

    def test_json_template_from_agent_kwarg_is_parsed(self):
        template = json.dumps(["claude", "-p", "{instruction}"])
        agent = self.build(template=template)
        (command,) = agent._run_agent_commands("do it")
        self.assertEqual(shlex.split(command.command), ["claude", "-p", "do it"])

    def test_template_without_placeholder_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.module.CommandAgent(argv_template=["true"])
        self.assertIn("{instruction}", str(caught.exception))

    def test_env_carries_endpoint_and_resolves_placeholders(self):
        os.environ["BENCH_STUB_KEY"] = "secret-value"
        try:
            agent = self.build(base_url="http://h200:30000/",
                               model="ornith-ai/Ornith-1.5-35B-A3B",
                               api_key_env="BENCH_STUB_KEY",
                               extra_env={"CUSTOM_TOKEN": "{env:BENCH_STUB_KEY}"})
            env = agent._env
        finally:
            del os.environ["BENCH_STUB_KEY"]
        self.assertEqual(env["BENCH_STUB_KEY"], "secret-value")
        self.assertEqual(env["CUSTOM_TOKEN"], "secret-value")
        self.assertEqual(env["OPENAI_BASE_URL"], "http://h200:30000/v1")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://h200:30000")
        self.assertEqual(env["ANTHROPIC_MODEL"], "ornith-ai/Ornith-1.5-35B-A3B")

    def test_install_script_contains_the_commands_and_is_executable(self):
        agent = self.build(install_commands=["pip install -e .", "echo ready"])
        path = agent._install_agent_script_path
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
            self.assertIn("pip install -e .", text)
            self.assertTrue(text.startswith("#!/bin/bash"))
            self.assertTrue(os.access(path, os.X_OK))
        finally:
            os.unlink(path)

    def test_empty_install_script_is_still_valid(self):
        path = self.build()._install_agent_script_path
        try:
            with open(path, encoding="utf-8") as handle:
                self.assertIn("set -e", handle.read())
        finally:
            os.unlink(path)

    def test_name_is_stable_for_terminal_bench(self):
        self.assertEqual(self.build().name(), "command")


if __name__ == "__main__":
    unittest.main()
