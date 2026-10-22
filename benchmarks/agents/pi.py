"""Pi adapter (``earendil-works/pi``, npm ``@earendil-works/pi-coding-agent``).

Pi is driven in its documented one-shot mode: ``pi --print`` writes the final
assistant text to stdout and exits, and ``--mode json`` writes the JSONL event
stream instead. Both are shell-free.

Unlike Claude Code, Pi can talk to a local vLLM/SGLang server directly: its
``models.json`` accepts a provider with ``"api": "openai-completions"`` and a
``baseUrl``, which is exactly what ``benchmarks/configs/pi_models.json`` contains.
The adapter therefore never downgrades to a hosted provider silently -- if the
configured model is not an OpenAI-compatible endpoint, preflight says so.
"""
from .base import AgentAdapter, which


class PiAgent(AgentAdapter):
    name = "pi"
    speaks = "openai"
    install_hint = ("curl -fsSL https://pi.dev/install.sh | sh, or "
                    "npm install -g --ignore-scripts @earendil-works/pi-coding-agent "
                    "(Node >= 22.19)")

    settings = {
        "default": {},
        "json-events": {"mode": "json"},
        "trusted-project": {"approve": True},
        "read-only-tools": {"tools": "read,grep,find,ls"},
        "all-tools": {"tools": "read,bash,edit,write,grep,find,ls"},
    }

    def argv(self, prompt, model, workspace=None, container=False):
        options = self.options()
        argv = ["pi"]
        if options.get("mode") == "json":
            argv += ["--mode", "json"]
        else:
            argv += ["--print"]
        # ``--provider``/``--model``: the provider name is the key the models.json
        # entry is registered under (default ``local`` for a vLLM endpoint).
        provider = options.get("provider")
        if provider:
            argv += ["--provider", str(provider), "--model", model.model]
        else:
            argv += ["--model", model.model]
        if options.get("tools"):
            argv += ["--tools", str(options["tools"])]
        if options.get("approve"):
            # Trust project-local .pi config for this process; needed when the
            # task image carries its own settings and nobody can answer a prompt.
            argv += ["--approve"]
        if options.get("no_session", True):
            # A benchmark instance must not inherit (or write) a user session:
            # resuming one would leak context between tasks.
            argv += ["--no-session"]
        argv += ["--", prompt] if str(prompt).lstrip().startswith("-") else [prompt]
        return argv

    def env(self, model):
        env = {}
        if model.api_key_env:
            env[model.api_key_env] = "{env:%s}" % model.api_key_env
        # Pi's models.json interpolates ``$NAME``/``${NAME}``, so pointing it at
        # the run's endpoint needs no file rewrite -- see configs/pi_models.json.
        # It is the server *root*: the config there appends ``/v1`` itself.
        env["PI_BENCH_BASE_URL"] = str(model.base_url).rstrip("/").removesuffix("/v1")
        env["PI_BENCH_MODEL"] = model.model
        return env

    def preflight(self, model, require_binary=True):
        problems = super().preflight(model)
        if require_binary and not which("pi"):
            problems.append(
                f"agent {self.label!r}: the 'pi' CLI is not on PATH "
                f"({self.install_hint})")
        return problems

    def install_commands(self):
        return [
            "apt-get update && apt-get install -y curl",
            "curl -fsSL https://pi.dev/install.sh | sh",
        ]

    def harbor_selection(self, model):
        # Harbor ships a Pi agent, and it already knows how to point Pi at a
        # custom OpenAI-compatible endpoint -- so no bridge is needed here.
        return {"agent": "pi", "kwargs": {}}

    def metadata(self):
        payload = super().metadata()
        payload["models_file"] = "configs/pi_models.json"
        return payload
