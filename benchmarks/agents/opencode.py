"""OpenCode adapter (``opencode`` CLI, npm ``opencode-ai``).

Harbor ships the OpenCode agent itself, including its Node install and the
``opencode run --format=json`` trajectory capture. This adapter only selects that
built-in agent and makes sure an OpenAI-compatible local endpoint is handed to
it: OpenCode asks for ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` for the
``openai`` provider.

The benchmark preamble is not injected here: Harbor's built-in OpenCode has no
``--append-system-prompt`` hook. It reads a project ``AGENTS.md`` instead, so a
future run that wants the shared preamble must upload one into the task
worktree.
"""
from .base import AgentAdapter


class OpenCodeAgent(AgentAdapter):
    name = "opencode"
    speaks = "openai"
    install_hint = "npm install -g opencode-ai (Node >= 18)"

    settings = {
        "default": {},
        "no-web": {"disable_web_search": True},
    }

    def argv(self, prompt, model, workspace=None, container=False):
        """One-shot ``opencode run`` command for a non-Harbor harness."""
        argv = [
            "opencode",
            "--model", f"openai/{model.model}",
            "run",
            "--format=json",
            "--thinking",
            "--dangerously-skip-permissions",
        ]
        return argv + (["--", str(prompt)] if str(prompt).lstrip().startswith("-")
                       else [str(prompt)])

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

    def harbor_selection(self, model):
        # Harbor's built-in OpenCode handles installation, provider config and
        # --dangerously-skip-permissions; no CommandAgent bridge is needed.
        selection = {"agent": "opencode"}
        if self.setting == "no-web":
            selection["kwargs"] = {"disable_web_search": True}
        return selection

    def install_commands(self):
        return ["npm install -g opencode-ai"]
