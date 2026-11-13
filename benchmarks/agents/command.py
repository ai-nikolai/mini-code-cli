"""Generic command-template adapter: bring your own agent.

Many coding agents are "a CLI that takes a prompt", and adding a full adapter for
each one is busywork. This adapter turns a template into an argv: every element is
substituted and then passed to the process directly -- there is no shell, so a
prompt containing ``$(rm -rf /)`` is a string, not a command.

Placeholders: ``{prompt}``, ``{model}``, ``{base_url}``, ``{workspace}``,
``{setting}``. A template is a JSON list of strings in the config::

    {"name": "command", "setting": "aider", "options": {
        "template": ["aider", "--model", "{model}", "--message", "{prompt}"]}}
"""
from .base import AgentAdapter, AgentError

PLACEHOLDERS = ("prompt", "model", "base_url", "workspace", "setting")


class CommandAgent(AgentAdapter):
    name = "command"
    speaks = "openai"

    #: Settings are per-config here: the template *is* the setting. A default is
    #: required so validation has something to check.
    settings = {"default": {"template": ["true"]}}

    def options(self):
        # Any setting name is valid here: for a template-driven agent the
        # template *is* the setting, so there is no fixed preset list to check.
        merged = dict(self.spec.options or {})
        if not merged.get("template"):
            raise AgentError(
                "agent 'command' needs options.template: a list of argv elements "
                "using the placeholders " + ", ".join("{%s}" % p for p in PLACEHOLDERS))
        if not isinstance(merged["template"], (list, tuple)):
            raise AgentError("agent 'command': options.template must be a list")
        return merged

    def argv(self, prompt, model, workspace=None, container=False):
        values = {
            "prompt": prompt,
            "model": model.model,
            "base_url": model.base_url,
            "workspace": str(workspace or ""),
            "setting": self.setting,
        }
        argv = []
        for element in self.options()["template"]:
            try:
                argv.append(str(element).format(**values))
            except KeyError as e:
                raise AgentError(
                    f"agent 'command': unknown placeholder {e.args[0]!r} in "
                    f"template element {element!r}") from e
        return argv

    def env(self, model):
        if not model.api_key_env:
            return {}
        return {model.api_key_env: "{env:%s}" % model.api_key_env}

    def preflight(self, model, require_binary=True):
        problems = super().preflight(model)
        self.options()  # validates the template shape, raising AgentError
        return problems

    def harness_selection(self, model):
        return {
            "import_path": "benchmarks.tb_agents:CommandAgent",
            "kwargs": {
                "argv_template": self.argv("{instruction}", model),
                "agent_name": self.setting,
            },
        }
