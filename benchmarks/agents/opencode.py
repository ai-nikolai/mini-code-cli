"""OpenCode adapter (``opencode`` CLI, npm ``opencode-ai``).

Harbor ships an OpenCode agent, but its installer calls
``ensure_system_dependencies("nodejs", "npm")``. That runs apt inside the task
container, which fails on the read-only SIF overlay for some images. It also
selects OpenCode's native OpenAI provider, which posts to ``/v1/responses``;
vLLM exposes ``/v1/chat/completions``.

This adapter therefore uses the generic CommandAgent bridge with the same
Node/OpenCode install approach as the Pi/Claude adapters: download Node into a
writable prefix, install ``opencode-ai`` there, and run a small wrapper that
writes an OpenAI-compatible provider config before invoking
``opencode run --format=json``. The wrapper also prepends the shared benchmark
preamble, so OpenCode receives the same instructions as the other agents.
"""
import json
import shlex

from . import container_setup
from .base import AgentAdapter

#: Writable directory for Node and OpenCode. The SIF's overlay is small, so
#: this is the same prefix style used by the Pi/Claude installers.
TOOLS = "/root/.agent-tools"
WRAPPER = f"{TOOLS}/opencode-run"


class OpenCodeAgent(AgentAdapter):
    name = "opencode"
    speaks = "openai"
    install_hint = "Node 22 + `npm install -g opencode-ai`, installed in the task image"

    settings = {
        "default": {},
        "no-web": {"disable_web_search": True},
    }

    def argv(self, prompt, model, workspace=None, container=False):
        """Run the wrapper; it writes the provider config and invokes OpenCode."""
        return [WRAPPER, str(prompt)]

    def env(self, model):
        env = {}
        if model.api_key_env:
            env[model.api_key_env] = "{env:%s}" % model.api_key_env
        if model.base_url:
            env["OPENAI_BASE_URL"] = (
                str(model.base_url).rstrip("/")
                if str(model.base_url).rstrip("/").endswith("/v1")
                else str(model.base_url).rstrip("/") + "/v1")
        return env

    def install_commands(self, model=None):
        """Install Node/OpenCode and write the wrapper used by :meth:`argv`."""
        if model is None:
            # Generic harnesses call this without a model; the endpoint and
            # model are only needed for the wrapper's config, so skip it then.
            return (container_setup.node_commands(prefix=TOOLS)
                    + container_setup.npm_global_commands(
                        "opencode-ai", "opencode", prefix=TOOLS))
        config = {
            "provider": {
                "openai": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {
                        "baseURL": f"{str(model.base_url).rstrip('/')}/v1",
                        "apiKey": "local-vllm",
                    },
                    "models": {str(model.model): {}},
                },
            },
        }
        if self.setting == "no-web":
            config["permission"] = {"websearch": "deny", "webfetch": "deny"}
        config_json = shlex.quote(json.dumps(config, sort_keys=True))
        body = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/.config/opencode
config={config_json}
printf '%s\\n' "$config" > /root/.config/opencode/opencode.json
instruction="$1"
if [ -f /opt/agent/agent.md ]; then
    instruction="$(cat /opt/agent/agent.md)"$'\\n\\n'"$instruction"
fi
exec {TOOLS}/bin/opencode \\
    --model=openai/{model.model} \\
    run --format=json --thinking --dangerously-skip-permissions \\
    -- "$instruction"
"""
        return (container_setup.node_commands(prefix=TOOLS)
                + container_setup.npm_global_commands(
                    "opencode-ai", "opencode", prefix=TOOLS)
                + [f"cat > {WRAPPER} <<'SH_OPENCODE'\n{body}SH_OPENCODE\n"
                   f"chmod +x {WRAPPER}"])

    def harbor_selection(self, model):
        # CommandAgent installs the wrapper, uploads agent.md, and passes the
        # endpoint/key env. Harbor's own OpenCode agent is bypassed because its
        # apt-based installer and /v1/responses default do not fit this setup.
        return {
            "import_path": "benchmarks.harbor_agents:CommandAgent",
            "kwargs": {
                "argv_template": self.argv("{instruction}", model, container=True),
                "agent_name": self.label,
                "model": model.model,
                "base_url": model.base_url,
                "api_key_env": model.api_key_env or "",
                "install_commands": self.install_commands(model),
                "agent_md_file": self.agent_md(container=False),
                "exec_timeout_sec": 3600,
            },
        }
