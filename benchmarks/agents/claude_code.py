"""Claude Code adapter.

Claude Code speaks the Anthropic Messages API and ships as a Node CLI. The
invocation mirrors what Terminal-Bench's own built-in ``claude-code`` agent does
(``claude --verbose --output-format stream-json -p <instruction>`` with an
explicit ``--allowedTools`` list), so numbers taken here are comparable with a
``tb run --agent claude-code`` baseline.

Two consequences worth knowing before reading any result:

* **Endpoint.** ``ANTHROPIC_BASE_URL`` is honoured, so a local model can be put
  behind Claude Code -- but only through a proxy that speaks ``/v1/messages``
  (LiteLLM in Anthropic mode; see ``benchmarks/models.py``). Pointing it straight
  at vLLM does not work, and this adapter refuses rather than silently falling
  back to the public API.
* **Permissions.** In a benchmark container there is no human to approve tools.
  ``--permission-mode bypassPermissions`` is therefore available, but it is *off*
  by default: the default uses the same explicit ``--allowedTools`` allowlist the
  upstream harness uses, which keeps the comparison fair and the blast radius
  inside the container.
"""
import os

from .base import AgentAdapter, which

# The tool allowlist Terminal-Bench's claude-code agent uses; kept identical so
# the two harnesses measure the same agent capabilities.
DEFAULT_ALLOWED_TOOLS = (
    "Bash", "Edit", "Read", "Write", "Glob", "Grep", "LS", "WebFetch",
    "NotebookEdit", "TodoWrite",
)


class ClaudeCodeAgent(AgentAdapter):
    name = "claude-code"
    speaks = "anthropic"
    install_hint = "npm install -g @anthropic-ai/claude-code (Node >= 18)"

    settings = {
        "default": {},
        "bypass-permissions": {"bypass_permissions": True},
        "no-web": {"allowed_tools": ("Bash", "Edit", "Read", "Write", "Glob",
                                     "Grep", "LS", "TodoWrite")},
    }

    def argv(self, prompt, model, workspace=None, container=False):
        options = self.options()
        argv = [
            "claude",
            "--print",
            "--verbose",
            "--output-format", "stream-json",
            "--model", model.model,
            "--allowedTools", " ".join(options.get("allowed_tools",
                                                   DEFAULT_ALLOWED_TOOLS)),
        ]
        if options.get("bypass_permissions"):
            argv += ["--permission-mode", "bypassPermissions"]
        argv += ["--", prompt] if _needs_separator(prompt) else [prompt]
        return argv

    def env(self, model):
        """``ANTHROPIC_*`` only -- the key is referenced, never inlined."""
        env = {}
        if model.api_key_env:
            env[model.api_key_env] = "{env:%s}" % model.api_key_env
        env["ANTHROPIC_MODEL"] = _anthropic_model_id(model.model)
        # A non-Anthropic base URL means a proxy; pass it through so the CLI
        # actually talks to it instead of api.anthropic.com.
        if model.base_url and "api.anthropic.com" not in model.base_url:
            env["ANTHROPIC_BASE_URL"] = model.base_url
        return env

    def preflight(self, model, require_binary=True):
        problems = super().preflight(model)
        if require_binary and not which("claude"):
            problems.append(
                f"agent {self.label!r}: the 'claude' CLI is not on PATH "
                f"({self.install_hint})")
        return problems

    def notes(self, model):
        """Non-blocking remarks surfaced by ``plan``.

        A non-Anthropic base URL *may* be a perfectly good proxy, so this cannot
        be an error -- but it must not pass silently either, because pointing
        Claude Code at a raw vLLM endpoint fails in a confusing way.
        """
        if model.api == "anthropic" and "api.anthropic.com" not in model.base_url:
            return [f"Claude Code will talk to {model.base_url}: it must expose an "
                    f"Anthropic-compatible /v1/messages (LiteLLM proxy), not a "
                    f"plain OpenAI-compatible path"]
        return []

    def install_commands(self):
        """Install Node + the CLI inside a Debian/Ubuntu-based task image."""
        return [
            "apt-get update && apt-get install -y curl",
            "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash",
            'source "$HOME/.nvm/nvm.sh" && nvm install 22 && '
            "npm install -g @anthropic-ai/claude-code",
        ]

    def harness_selection(self, model):
        # Terminal-Bench ships a claude-code agent; use it rather than our own
        # copy so the baseline is the one the leaderboard uses.
        return {"agent": "claude-code", "kwargs": {
            "model_name": _anthropic_model_id(model.model)}}

    def harbor_selection(self, model):
        # Harbor ships a Claude Code agent; the baseline should be the one the
        # leaderboard uses, not a re-implementation of it.
        return {"agent": "claude-code", "kwargs": {}}

    def metadata(self):
        payload = super().metadata()
        payload["api"] = "anthropic"
        return payload


def _anthropic_model_id(model_id):
    """``anthropic/claude-...`` -> ``claude-...`` (the CLI rejects the prefix)."""
    return str(model_id).removeprefix("anthropic/")


def _needs_separator(prompt):
    """Whether the prompt could be mistaken for a flag (``--help`` in a task)."""
    return str(prompt).lstrip().startswith("-")
