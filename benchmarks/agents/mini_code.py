"""mini-code adapter, including the "different settings" matrix.

mini-code is driven exactly as a user would drive it -- ``python -m
mini_code_cli`` with a ``--prompt`` -- so the benchmark measures the shipped CLI
rather than a benchmark-only code path.

The presets are the experiment. Each one turns on one *capability* of the agent
and nothing else, so a difference in success rate can be attributed to it:

======================  ====================================================
``default``             the bare agent: no shell, no plan, no sub-agents
``auto-shell``          ``--auto-mode --enable-shell --sudo``: unattended
                        shell access (safe only inside the disposable
                        benchmark container -- ``--sudo`` is what makes a
                        prompt-free run possible)
``plan``                ``--plan``: goal/plan tracking across turns
``subagents``           ``--enable-subagents``: delegation to the built-in
                        sub-agents
``skills``              ``--skills``: the reusable-skill directory
``long-context``        a high ``--max-context-tokens``: compact late, to
                        separate "ran out of context" from "could not solve"
``short-tool-timeout``  ``--tool-timeout 30``: fail fast on hung commands
``all``                 the kitchen sink (plan + subagents + shell + sudo)
======================  ====================================================

Every preset is also recorded in the run manifest, so a report can say which
setting produced which numbers without re-deriving it from flags.
"""
from .base import AgentAdapter, which, python_interpreter


class MiniCodeAgent(AgentAdapter):
    """The CLI under test."""

    name = "mini-code"
    speaks = "openai"
    install_hint = "pip install -e . (the package this repository ships)"

    settings = {
        "default": {},
        "auto-shell": {"flags": ["--auto-mode", "--enable-shell", "--sudo"]},
        "plan": {"flags": ["--auto-mode", "--plan"]},
        "subagents": {"flags": ["--auto-mode", "--enable-subagents"]},
        "skills": {"flags": ["--auto-mode", "--skills"]},
        "long-context": {"flags": ["--auto-mode"], "max_context_tokens": 131072},
        "short-tool-timeout": {"flags": ["--auto-mode"], "tool_timeout": 30},
        "all": {"flags": ["--auto-mode", "--enable-shell", "--sudo",
                          "--plan", "--enable-subagents"]},
    }

    def argv(self, prompt, model, workspace=None, container=False):
        """``python -m mini_code_cli --prompt ...``, mirroring a real invocation.

        ``--url`` is the *server root* (``http://host:port``): the CLI appends
        ``/v1/chat/completions`` itself, so a configured ``.../v1`` base URL is
        trimmed here rather than producing ``/v1/v1/...``.
        """
        options = self.options()
        argv = [
            _interpreter(self.options(), container), "-m", "mini_code_cli",
            "--prompt", prompt,
            "--url", _server_root(model.base_url),
            "--model", model.model,
        ]
        if workspace:
            argv += ["--allowed-dir", str(workspace)]
        if options.get("max_context_tokens"):
            argv += ["--max-context-tokens", str(options["max_context_tokens"])]
        if options.get("tool_timeout"):
            argv += ["--tool-timeout", str(options["tool_timeout"])]
        if options.get("llm_timeout"):
            argv += ["--llm-timeout", str(options["llm_timeout"])]
        argv += list(options.get("flags", []))
        if model.api_key_env:
            # Pass the key by *reference*: the runner expands {env:NAME} at exec
            # time, so the value never appears in the command line, in `ps`, or in
            # the recorded plan.
            argv += ["--api-key", _env_placeholder(model.api_key_env)]
        return argv

    def env(self, model):
        """The API key the CLI should read, without copying its value around."""
        if not model.api_key_env:
            return {}
        # A literal placeholder: mini_code_cli reads the flag value, so the
        # runner substitutes {env:NAME} at execution time (see runner.expand_env).
        return {model.api_key_env: _env_placeholder(model.api_key_env)}

    def preflight(self, model, require_binary=True):
        problems = super().preflight(model)
        if require_binary and not which(python_interpreter()):
            problems.append(f"agent {self.label!r}: no python interpreter found")
        return problems

    def metadata(self):
        payload = super().metadata()
        payload["invocation"] = "python -m mini_code_cli --prompt ..."
        return payload


def _interpreter(options, container):
    """Which ``python`` to name.

    On the host that is this interpreter (so an editable checkout is used); in a
    task container it must be the image's ``python3``, because the host's absolute
    path does not exist there. ``options.python`` overrides both.
    """
    if options.get("python"):
        return str(options["python"])
    return "python3" if container else python_interpreter()


def _server_root(base_url):
    """``http://host:port/v1`` -> ``http://host:port`` (the CLI adds the path)."""
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url[: -len("/v1")]
    return url


def _env_placeholder(name):
    """A cli value the runner expands from the environment at exec time."""
    return "{env:%s}" % name
