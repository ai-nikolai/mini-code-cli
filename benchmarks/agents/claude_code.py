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
import json
import os
import urllib.error
import urllib.request

from . import container_setup
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
        """``ANTHROPIC_*`` only -- the key is referenced, never inlined.

        Four variables matter when the endpoint is not Anthropic's:

        ``ANTHROPIC_BASE_URL``
            where to send ``/v1/messages``.
        ``ANTHROPIC_AUTH_TOKEN`` *and* ``ANTHROPIC_API_KEY``
            the CLI treats a key-less client as unauthenticated and fails with
            ``AgentAuthenticationError`` before it ever reaches a local server; a
            local server ignores the value, but the variable has to exist. Both
            spellings are set because different CLI versions read one or the other.
        ``ANTHROPIC_SMALL_FAST_MODEL``
            the CLI also calls a "small/fast" model (summaries, titles). Left
            unset it would ask the local server for a Claude model it does not
            serve, and the failure surfaces as an authentication error.
        """
        env = {}
        if model.api_key_env:
            env[model.api_key_env] = "{env:%s}" % model.api_key_env
        env["ANTHROPIC_MODEL"] = _anthropic_model_id(model.model)
        env["ANTHROPIC_SMALL_FAST_MODEL"] = _anthropic_model_id(model.model)
        if model.base_url and "api.anthropic.com" not in model.base_url:
            env["ANTHROPIC_BASE_URL"] = model.base_url
            # A local endpoint has no key to check, but the CLI requires one to
            # exist. This is a placeholder, not a secret.
            env.setdefault("ANTHROPIC_AUTH_TOKEN",
                           "{env:%s}" % (model.api_key_env or "ANTHROPIC_API_KEY"))
            # Never phone home from a compute node: those calls fail (no egress)
            # and are reported as authentication errors, which sends you looking
            # in the wrong place.
            env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
            env["DISABLE_TELEMETRY"] = "1"
            env["DISABLE_ERROR_REPORTING"] = "1"
        return env

    def preflight(self, model, require_binary=True):
        problems = super().preflight(model)
        if require_binary and not which("claude"):
            problems.append(
                f"agent {self.label!r}: the 'claude' CLI is not on PATH "
                f"({self.install_hint})")
        if require_binary:
            problems += self.endpoint_problems(model)
        return problems

    def endpoint_problems(self, model):
        """Probe ``/v1/messages`` and translate a failure into an instruction.

        ``AgentAuthenticationError`` from the CLI is the same opaque message
        whether the endpoint is unreachable, is not Anthropic-shaped, or genuinely
        rejects the key -- so the check is done here, where the reason is known.
        """
        if "api.anthropic.com" in (model.base_url or ""):
            return []
        problem = probe_messages_endpoint(model.base_url)
        if problem is None:
            return []
        return [f"agent {self.label!r}: {problem}"]

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
        """Install Node + the CLI inside a Debian/Ubuntu-based task image.

        No apt and no nvm: the images are read-only ``.sif`` files whose dpkg
        cannot move existing directories into the overlay (see
        :mod:`benchmarks.agents.container_setup`).
        """
        return container_setup.node_commands() + container_setup.npm_global_commands(
            "@anthropic-ai/claude-code", "claude")

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


def probe_messages_endpoint(base_url, timeout=5.0):
    """Return ``None`` when ``base_url`` serves ``/v1/messages``, else a message.

    A minimal, non-streaming request: enough to see whether the route exists and
    whether the body shape is accepted, without spending a model call on a real
    completion (the server rejects it on ``max_tokens``/``messages`` validation
    before generating anything when we send an empty message list).
    """
    if not base_url:
        return None
    url = str(base_url).rstrip("/") + "/v1/messages"
    body = json.dumps({"model": "probe", "max_tokens": 1, "messages": []}).encode()
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"content-type": "application/json",
                 "anthropic-version": "2023-06-01",
                 "x-api-key": "probe"})
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            return None
    except urllib.error.HTTPError as e:
        if e.code in (400, 422):
            # The route exists and parsed our body; it only disliked the empty
            # message list. That is a working Anthropic endpoint.
            return None
        if e.code in (401, 403):
            return (f"{url} rejects the credentials (HTTP {e.code}); the CLI will "
                    f"report this as AgentAuthenticationError. Set "
                    f"ANTHROPIC_API_KEY to any non-empty value -- a local server "
                    f"does not check it, but the CLI refuses to start without it")
        if e.code == 404:
            return (f"{url} does not exist (HTTP 404): {base_url} is not an "
                    f"Anthropic-compatible endpoint. Put a LiteLLM proxy in front "
                    f"of vLLM (benchmarks/models.py knows how) or upgrade to a "
                    f"vLLM that serves /v1/messages")
        return f"{url} answered HTTP {e.code}; Claude Code needs /v1/messages"
    except (urllib.error.URLError, OSError) as e:
        return (f"cannot reach {url} ({e}); start the endpoint before the run, and "
                f"remember the container must be able to route to it")


def _anthropic_model_id(model_id):
    """``anthropic/claude-...`` -> ``claude-...`` (the CLI rejects the prefix)."""
    return str(model_id).removeprefix("anthropic/")


def _needs_separator(prompt):
    """Whether the prompt could be mistaken for a flag (``--help`` in a task)."""
    return str(prompt).lstrip().startswith("-")
