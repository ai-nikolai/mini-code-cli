"""Agent adapters: how each coding agent is driven, and nothing else.

An adapter answers four questions and owns no orchestration:

* **what to run** -- :meth:`AgentAdapter.argv` builds a shell-free ``argv`` for one
  non-interactive run against one workspace/instruction;
* **what it needs** -- :meth:`AgentAdapter.env` (never values that get recorded,
  only the environment the process needs) and :meth:`AgentAdapter.preflight`;
* **how it gets installed** -- :meth:`AgentAdapter.install_commands`, used when a
  container image has to be prepared;
* **how a harness that takes an agent by name is configured** --
  :meth:`AgentAdapter.harness_selection`, which is what lets the Terminal-Bench
  adapter hand ``tb`` either a built-in agent or our own installed-agent class.

Keeping this separate from the suites is what makes the matrix honest: the same
adapter object drives every suite, so "mini-code with plan mode" cannot quietly
mean two different things.
"""
import os
import shlex
import sys


class AgentError(RuntimeError):
    """An agent cannot be used as configured (missing binary, key, or option)."""


class AgentAdapter:
    """Base class for the agent adapters.

    Subclasses set :attr:`name`, the :attr:`settings` presets and implement
    :meth:`argv`. Everything else has a sensible default.
    """

    #: Registry name (``--agent`` on the CLI).
    name = ""

    #: ``setting name -> dict of options``. The first entry is the default.
    settings = {"default": {}}

    #: Wire protocol the agent's CLI speaks: ``"openai"`` or ``"anthropic"``.
    #: A model whose ``api`` differs needs a translating proxy (see models.py).
    speaks = "openai"

    #: Where to get the agent, for the preflight message.
    install_hint = ""

    def __init__(self, spec):
        self.spec = spec

    # -- configuration ---------------------------------------------------

    @property
    def setting(self):
        return self.spec.setting

    @property
    def label(self):
        return self.spec.label

    def options(self):
        """The preset's options merged with any per-agent ``options`` overrides."""
        if self.setting not in self.settings:
            raise AgentError(
                f"agent {self.name!r} has no setting {self.setting!r}; "
                f"known settings: {', '.join(sorted(self.settings))}")
        merged = dict(self.settings[self.setting])
        merged.update(self.spec.options or {})
        return merged

    # -- interface -------------------------------------------------------

    def argv(self, prompt, model, workspace=None, container=False):
        """Return the ``argv`` for one non-interactive run. Never a shell string.

        ``container=True`` means the argv will be executed *inside* a benchmark
        container rather than on this host: an adapter uses it to avoid baking
        host paths (an absolute interpreter path, a host-only binary) into the
        command, which is the difference between a command that runs and one that
        fails with "no such file".
        """
        raise NotImplementedError

    def env(self, model):
        """Environment variables the agent needs (names come from the config)."""
        return {}

    def container_mounts(self):
        """Host paths the command needs *inside* a benchmark container.

        ``(host, target)`` pairs, handed to the runtime as ``-v``/``--bind``.
        Apptainer does not mount the working directory the way docker does, so an
        adapter whose code lives in this repository has to say so explicitly,
        otherwise the in-container run dies with "No module named ...".
        """
        return ()

    def container_env(self, model):
        """Env for the in-container invocation (``env`` plus container paths)."""
        return dict(self.env(model))

    def harbor_python_path(self):
        """Host directory with the agent's package, for harbor's TB path.

        Harbor's agent interface has no bind mount, so an agent that is not
        installed in the task image has to be uploaded; returning a path here
        makes the suite pass it as ``--agent-kwarg python_path=...``.
        """
        return None

    def container_setup_files(self, model):
        """Files to write under the tools mount before the agent runs.

        ``{container_path: contents}``; the suite materialises them on the host
        directory that is bind-mounted at ``/opt/agent``. An adapter needs this
        when its CLI insists on a config file with the endpoint spelled out.
        """
        return {}

    def preflight(self, model, require_binary=True):
        """Return a list of problems that would stop this agent from running.

        ``require_binary=False`` is used by ``plan``: a dry run must be able to
        describe the command even on a machine where the agent is not installed
        (that is often the point of printing it).
        """
        problems = []
        if model.api != self.speaks:
            problems.append(
                f"agent {self.label!r} speaks the {self.speaks!r} API but model "
                f"{model.name!r} is {model.api!r}; run it through the proxy "
                f"described in benchmarks/models.py")
        if model.api_key_env and not os.environ.get(model.api_key_env):
            problems.append(
                f"agent {self.label!r} needs {model.api_key_env} in the "
                f"environment (model {model.name!r})")
        return problems

    def install_commands(self):
        """Shell commands that install the agent inside a container image."""
        return []

    def notes(self, model):
        """Informational remarks for ``plan`` output (never blocking)."""
        return []

    def harness_selection(self, model):
        """How a name-based harness should select this agent.

        Returns ``{"agent": <built-in name>}`` or
        ``{"import_path": <module:Class>, "kwargs": {...}}``. The default is the
        generic installed-agent class, which drives :meth:`argv` verbatim.
        """
        return {
            "import_path": "benchmarks.tb_agents:CommandAgent",
            # A *template* list, not a joined string: the installed-agent class
            # substitutes the instruction into its own element and quotes each
            # argument, so a prompt can never become shell syntax.
            "kwargs": {
                "argv_template": self.argv("{instruction}", model,
                                           container=True),
                "agent_name": self.label,
                "model": model.model,
                "base_url": model.base_url,
                "api_key_env": model.api_key_env or "",
                "extra_env": self.env(model),
                "install_commands": self.install_commands(),
            },
        }

    def harbor_selection(self, model):
        """How Harbor (Terminal-Bench 2.x) should select this agent.

        Same shape as :meth:`harness_selection`, but the bridge is
        ``benchmarks.harbor_agents:CommandAgent`` -- Harbor's installed-agent API
        is a different (async) one. Agents Harbor ships itself override this with
        ``{"agent": "<name>"}``.
        """
        return {
            "import_path": "benchmarks.harbor_agents:CommandAgent",
            "kwargs": {
                "argv_template": self.argv("{instruction}", model,
                                           container=True),
                "agent_name": self.label,
                "model": model.model,
                "base_url": model.base_url,
                "api_key_env": model.api_key_env or "",
                "extra_env": self.env(model),
                "install_commands": self.install_commands(),
            },
        }

    def metadata(self):
        """What goes into the manifest for this adapter/setting (no secrets)."""
        return {
            "agent": self.name,
            "setting": self.setting,
            "options": self.options(),
            "speaks": self.speaks,
        }


def which(program):
    """``shutil.which`` by another name: one place to stub out in tests."""
    import shutil

    return shutil.which(program)


def python_interpreter():
    """The interpreter used to launch a Python-based agent in this environment."""
    return sys.executable
