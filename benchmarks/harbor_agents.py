"""Harbor installed-agent bridge: run our agents inside Terminal-Bench 2.x.

Harbor is the official framework for Terminal-Bench 2.0/2.1, and it selects an
agent by name (``-a claude-code``) or by import path
(``--agent-import-path module:Class``). An *installed* agent is a Python class in
the harbor process that

1. installs the agent inside the task container (``install``/``setup``),
2. runs it against the task instruction (``run``), and
3. reports what happened through the ``AgentContext``.

:class:`CommandAgent` implements that generically from an argv *template* plus the
endpoint description, so every adapter that harbor does not already ship
(mini-code, ``command``, ``mock``) can be benchmarked on the 2.x task sets::

    harbor run -d terminal-bench@2.1 \\
        --agent-import-path benchmarks.harbor_agents:CommandAgent \\
        --agent-kwarg 'argv_template=["python3","-m","mini_code_cli",...]' ...

Compatibility note: this module is written against harbor 0.24's
``harbor.agents.installed.base.BaseInstalledAgent`` (``name``/``version``/
``install``/``run``, ``exec_as_agent``). Harbor is a young, fast-moving framework,
so the first real run should start with ``plan`` (which prints the command) and a
single ``--limit 1`` trial. ``harbor`` is imported at module level on purpose: this
module is only ever loaded by harbor itself, and importing it from our CLI would
make the harness a hard dependency.
"""
import json
import os
import shlex

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

#: Placeholder inside an ``argv_template`` element that receives the instruction.
INSTRUCTION = "{instruction}"

#: Where the shared preamble lands inside a task container.
AGENT_MD_IN_CONTAINER = "/opt/agent/agent.md"


class CommandAgent(BaseInstalledAgent):
    """Run any CLI agent described by an argv template inside a task container."""

    def __init__(self, argv_template=None, agent_name="command", model=None,
                 base_url=None, api_key_env=None, extra_env=None,
                 install_commands=None, python_path=None, agent_md_file=None,
                 **kwargs):
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
        # A host directory holding the agent's own package. Harbor has no bind
        # mount on this path, so it is uploaded into the container and put on
        # PYTHONPATH -- without it `python3 -m mini_code_cli` dies with
        # "No module named mini_code_cli" and the trial looks like a model failure.
        self._python_path = str(python_path) if python_path else None
        self._uploaded_python_path = None
        # The host path of benchmarks/agents/agent.md. Harbor has no bind mount
        # here either, so the preamble travels with the agent: uploaded to the
        # same fixed path the template refers to (see AGENT_MD_IN_CONTAINER).
        self._agent_md_file = str(agent_md_file) if agent_md_file else None

    # -- harbor interface ------------------------------------------------

    @staticmethod
    def name():
        """Harbor requires a class-level name; the run id carries the setting."""
        return "command"

    def version(self):
        """No version to probe: the agent is whatever the image provides."""
        return None

    async def install(self, environment: BaseEnvironment) -> None:
        """Install the agent: upload its code, then run the declared commands."""
        if self._python_path:
            # Upload the package *into* a directory on PYTHONPATH: the tree handed
            # over is the package itself, so `python3 -m mini_code_cli` resolves
            # only if its parent is the one exported below.
            root = "/opt/mini-code"
            name = os.path.basename(str(self._python_path).rstrip("/"))
            await environment.upload_dir(source_dir=self._python_path,
                                         target_dir=f"{root}/{name}")
            self._uploaded_python_path = root
        if self._agent_md_file and os.path.isfile(self._agent_md_file):
            await environment.upload_file(source_path=self._agent_md_file,
                                          target_path=AGENT_MD_IN_CONTAINER)
        for command in self._install_commands:
            await self.exec_as_agent(environment, command=str(command))

    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        """Run the agent once, with the instruction quoted into its own argv slot.

        The environment is prefixed as ``KEY=value`` assignments resolved *here*
        (in the harbor process), so a credential is passed to the container
        without being written into the plan, the manifest or our logs.
        """
        argv = [element.replace(INSTRUCTION, str(instruction))
                for element in self._argv_template]
        assignments = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in self._env().items())
        command = f"env {assignments} {shlex.join(argv)}" if assignments \
            else shlex.join(argv)
        await self.exec_as_agent(environment, command=command)
        if hasattr(context, "metadata") and isinstance(context.metadata, dict):
            context.metadata.setdefault("agent", self._agent_name)
            if self._model:
                context.metadata.setdefault("model", str(self._model))

    # -- helpers ---------------------------------------------------------

    def _env(self):
        env = {str(key): _resolve(str(value))
               for key, value in self._extra_env.items()}
        if self._api_key_env:
            env[str(self._api_key_env)] = os.environ.get(str(self._api_key_env), "")
        if self._base_url:
            root = str(self._base_url).rstrip("/")
            env.setdefault("OPENAI_BASE_URL", f"{root}/v1")
            env.setdefault("ANTHROPIC_BASE_URL", root)
        if self._model:
            env.setdefault("ANTHROPIC_MODEL", str(self._model))
        if self._uploaded_python_path:
            env.setdefault("PYTHONPATH", self._uploaded_python_path)
        return {key: value for key, value in env.items() if value != ""}


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
