"""Agent registry.

Adding an agent means adding one module and one line here (or a ``command`` entry
in the config); nothing in the suites needs to know about it, which is what keeps
the benchmark side agent-agnostic.
"""
from .base import AgentAdapter, AgentError
from .claude_code import ClaudeCodeAgent
from .command import CommandAgent
from .mini_code import MiniCodeAgent
from .mock_agent import MockAgent
from .pi import PiAgent

#: ``--agent`` name -> adapter class.
ADAPTERS = {
    MiniCodeAgent.name: MiniCodeAgent,
    ClaudeCodeAgent.name: ClaudeCodeAgent,
    PiAgent.name: PiAgent,
    CommandAgent.name: CommandAgent,
    MockAgent.name: MockAgent,
}

__all__ = [
    "ADAPTERS", "AgentAdapter", "AgentError", "ClaudeCodeAgent",
    "CommandAgent", "MiniCodeAgent", "MockAgent", "PiAgent", "get_adapter",
]


def get_adapter(spec):
    """Build the adapter for an :class:`~benchmarks.config.AgentSpec`.

    Raises :class:`AgentError` for an unknown name, listing the known ones: the
    config is user-editable, so this is a normal error rather than a crash.
    """
    adapter_class = ADAPTERS.get(spec.name)
    if adapter_class is None:
        raise AgentError(
            f"unknown agent {spec.name!r}; known agents: "
            f"{', '.join(sorted(ADAPTERS))}")
    return adapter_class(spec)
