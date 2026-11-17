"""Adapter for the deterministic mock agent (see :mod:`benchmarks.agents.mock`).

It launches *this* package's mock module rather than an external binary, so the
whole harness -- plan, run, capture, report -- can be exercised on a machine with
no agent installed, no GPU and no network.
"""
import sys

from .base import AgentAdapter


class MockAgent(AgentAdapter):
    name = "mock"
    speaks = "openai"
    install_hint = "built in (python -m benchmarks.agents.mock)"

    #: The behaviours are the "settings" here: a test asks for a known outcome.
    settings = {
        "empty": {"behaviour": "empty"},
        "touch": {"behaviour": "touch"},
        "fail": {"behaviour": "fail"},
        "timeout": {"behaviour": "timeout"},
    }

    def argv(self, prompt, model, workspace=None, container=False):
        # Inside a container the host's interpreter path does not exist: name
        # `python3` and let `options.install_script` make the package importable
        # there (the mock is normally run on the host, for plumbing checks).
        interpreter = "python3" if container else sys.executable
        argv = [
            interpreter, "-m", "benchmarks.agents.mock",
            "--prompt", prompt,
            "--behaviour", self.options().get("behaviour", "empty"),
        ]
        if workspace:
            argv += ["--workspace", str(workspace)]
        return argv

    def preflight(self, model, require_binary=True):
        # No model, no key, no binary: that is the whole point.
        return []

    def harbor_selection(self, model):
        return {"import_path": "benchmarks.harbor_agents:CommandAgent",
                "kwargs": {"argv_template": self.argv("{instruction}", model,
                                                     container=True),
                           "agent_name": "mock"}}

    def harness_selection(self, model):
        return {
            "import_path": "benchmarks.tb_agents:CommandAgent",
            "kwargs": {"argv_template": self.argv("{instruction}", model,
                                           container=True),
                       "agent_name": "mock"},
        }
