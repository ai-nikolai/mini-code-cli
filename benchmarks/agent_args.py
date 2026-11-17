"""Shared parsing for agent bridges, config values and the runner.

An agent bridge receives ``--agent-kwarg`` values as strings even when the
caller wrote a list or a mapping, and any value may carry ``{env:NAME}``
placeholders. The same rules apply to the Terminal-Bench bridge, the Harbor
bridge and the runner, so they live here instead of in three copies.
"""
import json
import os

#: Placeholder inside an ``argv_template`` element that receives the instruction.
INSTRUCTION = "{instruction}"


def resolve_env(value, environ=None):
    """Resolve ``{env:NAME}`` placeholders; an unset variable expands to ``''``.

    Expanding to ``''`` rather than raising keeps the failure where it belongs:
    the agent reports "no API key", which is a result, instead of the runner
    dying with a ``KeyError`` halfway through a matrix.
    """
    environ = environ if environ is not None else os.environ
    text = str(value)
    while "{env:" in text:
        start = text.index("{env:")
        end = text.find("}", start)
        if end == -1:
            break
        name = text[start + 5:end]
        text = text[:start] + environ.get(name, "") + text[end + 1:]
    return text


def as_list(value):
    """Accept a list or its JSON spelling (``--agent-kwarg`` passes a string)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            return [text]
        return list(parsed) if isinstance(parsed, list) else [text]
    return [text]


def as_mapping(value):
    """Accept a mapping or its JSON spelling; anything else becomes ``{}``."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}
