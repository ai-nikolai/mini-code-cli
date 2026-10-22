"""Terminal-Bench installed-agent bridge: run our agents inside ``tb``.

Terminal-Bench selects an agent by name or by import path
(``--agent-import-path module:Class``), and an "installed agent" is a Python
class in the harness process that

1. writes an install script that is copied into the task container
   (``_install_agent_script_path``),
2. exports environment variables there (``_env``), and
3. returns the shell command that runs the agent inside the task's tmux session
   (``_run_agent_commands``).

That is all this module implements, generically: :class:`CommandAgent` takes an
argv *template* and the endpoint description, and every adapter that is not
already built into Terminal-Bench (mini-code, Pi, ``command``, ``mock``) uses it::

    tb run --agent-import-path benchmarks.tb_agents:CommandAgent \\
        --agent-kwarg 'argv_template=["python","-m","mini_code_cli",...]' ...

Two details matter for correctness:

* the instruction is substituted into its own argv element and the whole argv is
  quoted with :func:`shlex.join`, so a task instruction can never inject shell
  syntax;
* :mod:`terminal_bench` is imported at module level on purpose. This module is
  only ever imported by ``tb`` itself (via ``--agent-import-path``); importing it
  from our own CLI would drag the harness in as a dependency, which the package
  deliberately avoids.
"""
import json
import os
import shlex
import tempfile
from pathlib import Path

from terminal_bench.agents.installed_agents.abstract_installed_agent import (
    AbstractInstalledAgent,
)
from terminal_bench.terminal.models import TerminalCommand

#: Placeholder inside an ``argv_template`` element that receives the instruction.
INSTRUCTION = "{instruction}"


class CommandAgent(AbstractInstalledAgent):
    """Run any CLI agent described by an argv template inside a task container."""

    def __init__(self, argv_template=None, agent_name="command", model=None,
                 base_url=None, api_key_env=None, extra_env=None,
                 install_commands=None, **kwargs):
        super().__init__(**kwargs)
        template = _as_list(argv_template)
        if not template:
            raise ValueError("CommandAgent needs argv_template (a list of argv "
                             "elements, one of which is '{instruction}')")
        if not any(INSTRUCTION in str(element) for element in template):
            raise ValueError(f"CommandAgent argv_template has no {INSTRUCTION} "
                             f"placeholder, so the task instruction would be lost")
        self._argv_template = [str(element) for element in template]
        self._agent_name = str(agent_name)
        self._model = model
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._extra_env = _as_mapping(extra_env)
        self._install_commands = _as_list(install_commands)

    # -- Terminal-Bench interface ----------------------------------------

    @staticmethod
    def name():
        """Terminal-Bench requires a class-level name; the run id carries the rest."""
        return "command"

    def __str__(self):
        return self._agent_name

    @property
    def _env(self):
        """Environment for the agent *inside* the container.

        Values are resolved here, in the harness process, and handed to
        Terminal-Bench, which writes them with a ``sh -c`` outside the tmux
        session so they do not end up in the recorded transcript.
        """
        env = {}
        for key, value in self._extra_env.items():
            env[str(key)] = _resolve(str(value))
        if self._api_key_env:
            env[str(self._api_key_env)] = os.environ.get(str(self._api_key_env), "")
        if self._base_url:
            # Both spellings: OpenAI-style clients read OPENAI_BASE_URL, Anthropic
            # clients read ANTHROPIC_BASE_URL, and a container may use either.
            root = str(self._base_url).rstrip("/")
            env.setdefault("OPENAI_BASE_URL", f"{root}/v1")
            env.setdefault("ANTHROPIC_BASE_URL", root)
        if self._model:
            env.setdefault("ANTHROPIC_MODEL", str(self._model))
        return env

    @property
    def _install_agent_script_path(self):
        """A ``.sh`` that installs the agent (empty script when nothing is needed)."""
        body = ["#!/bin/bash", "set -e"]
        body += [str(command) for command in self._install_commands]
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
        handle.write("\n".join(body) + "\n")
        handle.close()
        os.chmod(handle.name, 0o755)
        return Path(handle.name)

    def _run_agent_commands(self, instruction):
        """One blocking command: the template with the (quoted) instruction in it."""
        argv = [element.replace(INSTRUCTION, str(instruction))
                for element in self._argv_template]
        return [TerminalCommand(
            command=shlex.join(argv),
            min_timeout_sec=0.0,
            max_timeout_sec=float("inf"),
            block=True,
            append_enter=True,
        )]

def _resolve(value):
    """Resolve ``{env:NAME}`` placeholders from this process's environment."""
    text = str(value)
    while "{env:" in text:
        start = text.index("{env:")
        end = text.find("}", start)
        if end == -1:
            break
        name = text[start + 5:end]
        text = text[:start] + os.environ.get(name, "") + text[end + 1:]
    return text


def _as_list(value):
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


def _as_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}
