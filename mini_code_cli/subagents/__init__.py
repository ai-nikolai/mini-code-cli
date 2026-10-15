"""mini_code_cli.subagents -- registry of sub-agents the main agent can call.

Sub-agents are pluggable: each one is a :class:`~mini_code_cli.subagents.base.SubAgent`
instance. :func:`register` adds one; :func:`get_registry` returns the live dict
that the agent loop turns into extra tools when ``--enable-subagents`` is set.

Two worked examples live in :mod:`mini_code_cli.subagents.builtin`
(:class:`ResearchSubAgent`, :class:`PlanSubAgent`); import them here or write
your own and register it.
"""
from .base import SubAgent  # noqa: F401  (public re-export)

#: The live registry. Populated by :func:`register`.
REGISTRY = {}


def register(subagent, name=None):
    """Register a sub-agent instance so the agent loop can expose it.

    ``name`` defaults to ``subagent.name``. Registering with the same name
    replaces the previous entry (handy in tests). Returns the sub-agent.
    """
    key = name or subagent.name
    REGISTRY[key] = subagent
    return subagent


def unregister(name):
    """Remove a sub-agent from the registry by name (best-effort)."""
    REGISTRY.pop(name, None)


def get_registry():
    """Return the live {name: SubAgent} registry (a view, not a copy)."""
    return REGISTRY


def clear_registry():
    """Empty the registry (used by tests to get a clean slate)."""
    REGISTRY.clear()


# Import the built-in examples so they are registered and available. This keeps
# ``from mini_code_cli.subagents import get_registry`` returning a ready-to-use
# set out of the box. Importing here (after the helpers above) avoids a NameError
# when builtin.py calls register() at module load time. Re-export the module so
# the import reads as a genuine use (not an F401) and callers can do
# ``subagents.builtin``.
from . import builtin
__all__ = ["SubAgent", "builtin", "REGISTRY", "register", "unregister",
           "get_registry", "clear_registry"]
