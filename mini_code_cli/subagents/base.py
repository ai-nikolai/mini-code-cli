"""mini_code_cli.subagents.base -- the base class every sub-agent implements.

A *sub-agent* is a self-contained agent the main agent can delegate a task to.
Conceptually it is a *narrowed* :class:`~mini_code_cli.agent_loop.AgentSession`:
it has its own system prompt, its own (usually smaller) set of tools, runs
unattended in auto-mode, and hands a text result back to the caller.

Why sub-agents
--------------
Breaking a big job into focused sub-agents makes the main agent cheaper, safer
and easier to reason about. Each sub-agent:

  * runs in its own isolated session directory (a fresh folder under the
    parent's cache dir), so its history / plan / usage never collide with the
    parent's;
  * reuses the whole existing agent-loop machinery (LLM calls, tools, the
    shell-safety classifier, the ``--allowed-dir`` sandbox, history compaction
    and --plan mode) -- nothing is re-implemented here;
  * returns a concise text result the parent can feed back to the model.

Combining with --plan mode
--------------------------
In plan mode the parent keeps ``CURRENT_GOAL.md`` / ``CURRENT_PLAN.md``. A
sub-agent is typically handed *one* plan item to work on; it keeps its own
``CURRENT_PLAN.md`` / ``CURRENT_GOAL.md`` in its own session dir. Sub-agents may
also write a short ``report.md`` back into their own session dir, which the
parent reads and surfaces to the model.

Adding a new sub-agent
----------------------
1. Subclass :class:`SubAgent` and set ``name`` / ``description`` / the options
   you support (see ``__init__.py`` for examples).
2. Register it in :func:`mini_code_cli.subagents.get_registry`.
That is all -- the agent loop discovers every registered sub-agent and exposes
each one to the model as a tool automatically.
"""

import io
import os
import types

from .. import ui


class SubAgent:
    """Base class for all sub-agents.

    Subclasses set ``name`` / ``description`` and may override the hooks below.
    ``options`` is the dict of per-invocation settings (``task`` is passed
    separately, not stored in ``options``).
    """

    #: Tool name the model calls to invoke this sub-agent. Must be unique.
    name = "subagent"
    #: One-line description shown to the model in the tool schema.
    description = "Delegate a focused task to a specialised sub-agent."
    #: Which tools this sub-agent may use. ``None`` = inherit the parent's full
    #: tool set. A list of names restricts it (e.g. ``["read_file", "grep"]``).
    tools = None
    #: Whether this sub-agent may run shell commands.
    enable_shell = False
    #: Default cap on LLM calls the sub-agent may make (None = unbounded).
    max_steps = None

    def __init__(self, **options):
        # Options accepted by every sub-agent. Subclasses read extra keys from
        # self.options as they please.
        self.options = dict(options)
        self.max_steps = options.get("max_steps", self.max_steps)

    # -- identity ----------------------------------------------------------

    @property
    def display_name(self):
        return self.name

    def tool_schema(self):
        """Return the OpenAI-style tool schema the model uses to call us.

        Every sub-agent looks the same to the LLM: it takes a ``task`` plus a
        couple of common knobs (``max_steps``). Subclasses can extend
        ``_extra_tool_parameters`` to advertise more options.
        """
        properties = {
            "task": {
                "type": "string",
                "description": (
                    "A clear, self-contained description of the task the "
                    "sub-agent should complete."
                ),
            },
            "max_steps": {
                "type": "integer",
                "description": (
                    "Optional cap on how many model calls the sub-agent may "
                    "make before returning (leave out for the default)."
                ),
            },
        }
        properties.update(self._extra_tool_parameters())
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": ["task"],
                },
            },
        }

    def _extra_tool_parameters(self):
        """Hook for subclasses to add extra parameters to their tool schema."""
        return {}

    # -- prompt / tool selection ------------------------------------------

    def system_prompt(self, task, parent):
        """Return the system prompt for the nested agent.

        The default prepends a short "you are a sub-agent" preamble to the
        standard system prompt, and tells the nested agent where to write any
        report it wants handed back to the parent.
        """
        from .. import prompts

        allowed_tools = self._resolve_tools(parent)
        if allowed_tools is None:
            tools_dict = parent.tools_dict
        else:
            tools_dict = {n: parent.tools_dict[n] for n in allowed_tools if n in parent.tools_dict}

        base = prompts.system_prompt(
            tools_dict=tools_dict,
            allowed_dir=parent.allowed_dir,
            plan_mode=self.wants_plan_mode(parent),
        )
        report_path = self._report_path(parent)
        preamble = (
            "\n\n---\nYou are a SUB-AGENT working on behalf of the main agent.\n"
            f"Task: {task}\n"
            "Work autonomously to complete the task, then finish. When done, "
            "write a concise summary of what you did (and reference any files "
            f"the main agent needs) to: {report_path}\n"
            "Do not ask the user for anything; resolve the task yourself."
        )
        return base + preamble

    def _resolve_tools(self, parent):
        """The list of tool names this sub-agent may use (or None for all)."""
        return self.tools

    def wants_plan_mode(self, parent):
        """Whether the nested agent should run in plan mode.

        Defaults to matching the parent, so plan-mode sessions get plan-mode
        sub-agents. The ``plan`` sub-agent forces this on.
        """
        return getattr(parent, "plan_mode", False)

    # -- execution ---------------------------------------------------------

    def _child_cache_dir(self, parent):
        """Where the nested agent's own history lives (isolated under the parent)."""
        return os.path.join(parent.cache_dir, parent.unique_id, "subagents", self.name)

    def _report_path(self, parent):
        """Location the nested agent should write its report to."""
        return os.path.join(self._child_cache_dir(parent), "report.md")

    def run(self, task, parent, max_steps=None):
        """Run the nested agent for ``task`` inside ``parent`` and return text.

        Spins up a child :class:`AgentSession` (reusing all the normal loop
        machinery) with this sub-agent's prompt, tools, sandbox and shell
        settings, runs it unattended in auto-mode, then returns a summary built
        from the child's final answer and any ``report.md`` it wrote. Best
        effort: any failure degrades to an Error string rather than raising, so
        a broken sub-agent cannot take down the main session.
        """
        from .. import agent_loop  # local import to avoid an import cycle

        max_steps = self.max_steps if max_steps is None else max_steps
        child_args = self._build_child_args(parent, task)
        # Swallow the nested transcript so the parent's screen is not spammed;
        # only the parent's own notes/report get shown. The "Delegating" /
        # "finished" notes are emitted by the *parent loop* (AgentSession
        # ._run_subagent), not here -- that keeps a single announcement on the
        # parent console rather than a duplicate one from this nested run().
        child_console = ui.Console(stream=io.StringIO(), color=False, unicode=False, verbose=False)
        try:
            child = agent_loop.AgentSession(child_args, console=child_console)
            # The child writes its report into its own session dir. Read it
            # back from there: ``self._report_path(parent)`` is *exactly* the
            # path the system prompt (built at system_prompt() time) told the
            # child to write to, so the two always agree.
            report_path = self._report_path(parent)
            final_text = child._drive_auto(task, max_steps=max_steps)
            return self._build_result(child, final_text, report_path)
        except Exception as e:
            # Turn any failure into an Error string the parent loop feeds back
            # as the tool result (so the main agent sees why delegation failed
            # and can react), rather than letting it propagate and crash the
            # parent step.
            return f"Error: sub-agent '{self.display_name}' failed: {e}"

    def _build_child_args(self, parent, task):
        """Build the argparse-like namespace for the nested AgentSession.

        The child inherits the parent's server/cache/allow-dir/timeout config
        but is narrowed to this sub-agent's tools, shell setting and (optionally)
        its own model/url, and runs non-interactively in auto-mode.
        """
        child = types.SimpleNamespace()
        # LLM / server (allow an explicit override per sub-agent).
        child.url = self.options.get("url", parent.args.url)
        child.model = self.options.get("model", parent.args.model)
        child.api_key = parent.args.api_key
        child.max_tokens = parent.args.max_tokens
        child.temperature = parent.args.temperature
        child.llm_timeout = parent.args.llm_timeout
        child.tool_timeout = parent.args.tool_timeout
        # Behaviour.
        child.auto_mode = True          # sub-agents run unattended to completion
        child.ask_permission = False
        child.sudo = False
        child.agent_md = None
        child.system_prompt = self.system_prompt(task, parent)
        # Sandbox + shell (sub-agents never widen the parent's permissions).
        child.allowed_dir = parent.allowed_dir
        child.enable_shell = bool(self.enable_shell and parent.args.enable_shell)
        # Where the child's own history lives (isolated under the parent session).
        # This is also where its report.md goes: the system prompt tells the
        # child to write to self._report_path(parent), which is exactly
        # <this dir>/report.md, so the parent reads back what the child wrote.
        child.cache_dir = self._child_cache_dir(parent)
        # Plan mode for the child.
        child.plan = self.wants_plan_mode(parent)
        child.verbose = False
        child.restore_session = None
        child.load_last_session = False
        child.prompt = task
        # The report.md this sub-agent hands back to the parent. The system
        # prompt already tells the child to write to self._report_path(parent),
        # so we surface the same path to the child session: it uses this to
        # exempt its own report from the --allowed-dir sandbox (the report lives
        # under the parent's cache dir, which is usually outside --allowed-dir,
        # so without the exemption the child's write_file of the report would be
        # blocked). Only set when enabled so a non-delegating AgentSession stays
        # clean.
        child.report_path = self._report_path(parent)
        return child

    def _build_result(self, child, final_text, report_path):
        """Assemble the text handed back to the parent.

        Combines the child's final assistant answer with any ``report.md`` it
        wrote. Falls back to a short "no output" note if there is neither.
        """
        report = None
        if os.path.isfile(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report = f.read().strip()
            except OSError:
                report = None

        parts = []
        if final_text and final_text.strip():
            parts.append(f"Result from sub-agent '{self.display_name}':\n{final_text.strip()}")
        if report:
            parts.append(f"\nReport ({report_path}):\n{report}")
        if not parts:
            parts.append(f"Sub-agent '{self.display_name}' produced no output.")
        return "\n".join(parts)


def _oneline(text, limit):
    one = " ".join(line.strip() for line in str(text).splitlines())
    one = " ".join(one.split())
    return one[:limit] + "..." if len(one) > limit else one
