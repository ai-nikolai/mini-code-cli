"""The shared agent preamble: one file, every agent.

``benchmarks/agents/agent.md`` is the single source of the instructions an agent
is given before the task text. Each adapter delivers it through its own native
mechanism -- mini-code reads the file with ``--agent-md``, Claude Code gets it
appended to its system prompt, pi receives it at the head of its instruction --
so a comparison across agents cannot be explained by three different preambles.

This module lives apart from ``benchmarks.agents.__init__`` because the adapters
themselves import it (importing the registry from an adapter would be a cycle).
"""
import os

#: Absolute path of the preamble inside this repository.
AGENT_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.md")


def load_agent_preamble(path=None):
    """Return ``(path, text)`` for the shared preamble.

    A missing file yields ``(path, "")`` rather than an exception: a partial
    checkout must still be able to run, and the caller then simply passes no
    preamble instead of failing a whole benchmark run.
    """
    path = path or AGENT_MD
    try:
        with open(path, encoding="utf-8") as handle:
            return path, handle.read()
    except OSError:
        return path, ""


__all__ = ["AGENT_MD", "load_agent_preamble"]
