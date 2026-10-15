"""mini_code_cli.subagents.builtin -- two worked example sub-agents.

These are registered on import (see :mod:`mini_code_cli.subagents`). They double
as documentation for how to build your own: a sub-agent is just a small class
configuring a name, a description, and a few knobs.
"""
from .base import SubAgent
from . import register


class ResearchSubAgent(SubAgent):
    """Look something up on the web and report back.

    Uses only the read-only web tools (``web_search`` / ``read_website``), so it
    is safe to enable even in non-sandboxed sessions -- it cannot touch the
    filesystem or run shell commands. Ideal for "find out how X works" / "what's
    the latest on Y" style sub-tasks without derailing the main conversation.
    """

    name = "research"
    description = (
        "Research a topic on the web: given a topic, search the web and read a "
        "few sources, then return a concise, sourced summary. Safe: read-only."
    )
    tools = ["web_search", "read_website"]
    enable_shell = False
    max_steps = 8

    def _extra_tool_parameters(self):
        # Let the caller bias the research toward a domain, e.g. only fetch
        # wikipedia.org pages. Purely advisory (passed into the system prompt).
        return {
            "sources": {
                "type": "string",
                "description": "Optional hint about which sources to prefer (e.g. 'wikipedia.org only').",
            },
        }

    def system_prompt(self, task, parent):
        sources = self.options.get("sources")
        base = super().system_prompt(task, parent)
        if sources:
            base += f"\n\nPrefer sources related to: {sources}"
        return base


class PlanSubAgent(SubAgent):
    """Draft or refine a step-by-step plan for a goal.

    Always runs in plan mode and reports back by writing the plan it produced
    into ``report.md`` so the parent can surface it. Useful for delegating the
    "how do I even approach this?" step to a focused, throwaway agent.
    """

    name = "planner"
    description = (
        "Produce a clear, step-by-step plan for accomplishing a goal, and return "
        "it as the result. Use when a task needs to be broken down before acting."
    )
    tools = ["read_file", "write_file"]
    enable_shell = False
    max_steps = 5

    def wants_plan_mode(self, parent):
        # The planner always runs plan-mode so the nested agent keeps a tidy
        # CURRENT_PLAN.md / CURRENT_GOAL.md of its own.
        return True

    def system_prompt(self, task, parent):
        base = super().system_prompt(task, parent)
        return base + (
            "\n\nOutput the plan as a numbered markdown list in CURRENT_PLAN.md, "
            "then copy it verbatim into report.md so the main agent receives it."
        )


# Register the built-in examples.
register(ResearchSubAgent())
register(PlanSubAgent())
