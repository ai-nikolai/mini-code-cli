"""Tests for the Harbor installed-agent bridge (``benchmarks/harbor_agents.py``).

Harbor is only installed on the machine that runs Terminal-Bench 2.x, so these
tests stand in for the three symbols the bridge imports. What is worth testing
here is exactly what a stub can check: the instruction is quoted into one argv
element, the endpoint environment is built (and keeps secrets out of the command
string), and a template without the placeholder is rejected.
"""
import asyncio
import importlib
import json
import os
import shlex
import sys
import types
import unittest


def _install_harbor_stub():
    """Register the minimal ``harbor`` surface ``harbor_agents`` imports."""
    if "harbor" in sys.modules:
        return

    package = types.ModuleType("harbor")
    agents = types.ModuleType("harbor.agents")
    installed = types.ModuleType("harbor.agents.installed")
    base = types.ModuleType("harbor.agents.installed.base")
    environments = types.ModuleType("harbor.environments")
    env_base = types.ModuleType("harbor.environments.base")
    models = types.ModuleType("harbor.models")
    agent_models = types.ModuleType("harbor.models.agent")
    context = types.ModuleType("harbor.models.agent.context")

    class BaseInstalledAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def exec_as_agent(self, environment, command=None):
            # Record instead of executing: that is the assertion surface.
            environment.commands.append(command)

    class BaseEnvironment:
        def __init__(self):
            self.commands = []

    class AgentContext:
        def __init__(self):
            self.metadata = {}

    base.BaseInstalledAgent = BaseInstalledAgent
    env_base.BaseEnvironment = BaseEnvironment
    context.AgentContext = AgentContext

    sys.modules.update({
        "harbor": package,
        "harbor.agents": agents,
        "harbor.agents.installed": installed,
        "harbor.agents.installed.base": base,
        "harbor.environments": environments,
        "harbor.environments.base": env_base,
        "harbor.models": models,
        "harbor.models.agent": agent_models,
        "harbor.models.agent.context": context,
    })


class HarborCommandAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_harbor_stub()
        cls.module = importlib.import_module("benchmarks.harbor_agents")

    def build(self, template=None, **kwargs):
        template = template or ["python3", "-m", "mini_code_cli",
                                "--prompt", "{instruction}"]
        return self.module.CommandAgent(argv_template=template, **kwargs)

    def run_agent(self, agent, instruction="do the task"):
        environment = self.module.BaseEnvironment()
        context = self.module.AgentContext()
        asyncio.run(agent.run(instruction, environment, context))
        return environment.commands[-1], context

    def test_instruction_stays_one_argument(self):
        command, context = self.run_agent(self.build(),
                                          "rm -rf / ; echo $(whoami)")
        self.assertEqual(shlex.split(command),
                         ["python3", "-m", "mini_code_cli", "--prompt",
                          "rm -rf / ; echo $(whoami)"])
        self.assertEqual(context.metadata["agent"], "command")

    def test_env_prefix_carries_the_endpoint_and_the_key(self):
        os.environ["BENCH_STUB_KEY"] = "secret-value"
        try:
            agent = self.build(base_url="http://h200:30000/",
                               model="ornith-ai/Ornith-1.5-35B-A3B",
                               api_key_env="BENCH_STUB_KEY")
            command, _ = self.run_agent(agent)
        finally:
            del os.environ["BENCH_STUB_KEY"]
        self.assertTrue(command.startswith("env "))
        self.assertIn("OPENAI_BASE_URL=http://h200:30000/v1", command)
        self.assertIn("ANTHROPIC_BASE_URL=http://h200:30000", command)
        self.assertIn("BENCH_STUB_KEY=secret-value", command)

    def test_unset_optional_env_is_omitted_rather_than_empty(self):
        agent = self.build(api_key_env="BENCH_STUB_UNSET")
        command, _ = self.run_agent(agent)
        self.assertNotIn("BENCH_STUB_UNSET", command)

    def test_json_template_and_install_commands(self):
        template = json.dumps(["claude", "-p", "{instruction}"])
        agent = self.build(template=template, install_commands=["pip install -e ."])
        environment = self.module.BaseEnvironment()
        asyncio.run(agent.install(environment))
        self.assertEqual(environment.commands, ["pip install -e ."])
        command, _ = self.run_agent(agent, "go")
        self.assertEqual(shlex.split(command), ["claude", "-p", "go"])

    def test_template_without_placeholder_is_rejected(self):
        with self.assertRaises(ValueError) as caught:
            self.module.CommandAgent(argv_template=["true"])
        self.assertIn("{instruction}", str(caught.exception))

    def test_name_and_version_follow_harbors_contract(self):
        agent = self.build()
        self.assertEqual(agent.name(), "command")
        self.assertIsNone(agent.version())


if __name__ == "__main__":
    unittest.main()
