"""mini_code_cli.subagents.builtin -- the built-in sub-agents.

These are registered on import (see :mod:`mini_code_cli.subagents`). They double
as documentation for how to build your own: a sub-agent is just a small class
configuring a name, a description, a set of tools, and a few knobs. Each one
targets a specific kind of work -- its ``system_prompt`` overrides add a short,
task-specific preamble on top of the shared base prompt so the nested agent
knows exactly what "done" looks like for this kind of task.
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

    Always runs in plan mode so the nested agent keeps its own CURRENT_PLAN.md /
    CURRENT_GOAL.md maintenance (as a self-contained example), but produces the
    plan as plain text and reports back by writing it into ``report.md`` so the
    parent can surface it. Useful for delegating the "how do I even approach
    this?" step to a focused, throwaway agent.
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
        # The planner always runs plan-mode so it demonstrates (and maintains)
        # the loop-owned CURRENT_PLAN.md / CURRENT_GOAL.md files as a
        # self-contained example, then reports the plan back via report.md.
        return True

    def system_prompt(self, task, parent):
        base = super().system_prompt(task, parent)
        return base + (
            "\n\nProduce a clear, step-by-step plan for this goal and return it "
            "as text; copy it verbatim into report.md so the main agent receives "
            "it. (The loop owns the actual CURRENT_PLAN.md / CURRENT_GOAL.md "
            "tracking files -- this agent produces the plan as output, not by "
            "writing those files.)"
        )


class SoftwareEngineeringSubAgent(SubAgent):
    """Implement, edit, refactor or test code in a project.

    This is the "do the actual engineering" sub-agent. It can read and write
    files and run shell commands, so it is meant for real coding work: writing a
    new module, refactoring, wiring up a fix, running the tests / a build to
    check it. It is given a task-specific system prompt that biases it toward
    producing working, tested, minimal changes rather than exploratory writing.

    Because it can run shell commands it should only be used in sessions where
    the caller is comfortable with that; the parent's sandbox / shell-safety
    rules still apply.
    """

    name = "software_engineer"
    description = (
        "Software-engineering sub-agent: write, edit, refactor, fix or test code "
        "in the project. It can read/write files and run shell commands (tests, "
        "builds, etc.) to verify the work. Hands back a summary of the changes."
    )
    tools = ["read_file", "write_file", "grep", "glob", "run_shell_command"]
    enable_shell = True
    max_steps = 20

    def system_prompt(self, task, parent):
        base = super().system_prompt(task, parent)
        return base + (
            "\n\n---\nYou are the SOFTWARE ENGINEERING sub-agent: you write and "
            "modify code to get a real task done. Follow these guidelines:\n"
            "* Understand the relevant code before editing: read the files and "
            "use grep/glob to find what you need. Prefer small, targeted edits over "
            "rewriting large files.\n"
            "* Keep changes minimal and focused on the task; do not touch unrelated "
            "code.\n"
            "* Make edits that are syntactically valid and consistent with the "
            "existing style (imports, formatting, naming).\n"
            "* When it makes sense, verify your work: run the project's tests, a "
            "build, a linter or the code itself with run_shell_command, and use "
            "the output to fix problems. Do not claim something works without "
            "checking when you are able to check.\n"
            "* If a task is ambiguous, make a sensible, minimal choice and state "
            "it. Note any trade-offs or follow-ups.\n"
            "When finished, write a concise summary of what you changed, why, and "
            "how you verified it (including any test/build commands you ran and "
            "their results) to report.md."
        )


class GenericSubAgent(SubAgent):
    """A general-purpose helper for tasks that do not fit a narrower agent.

    This is the catch-all sub-agent: it has the full set of read/write tools and
    may run shell commands, so it can handle "everything else" -- formatting a
    document, organising files, drafting text, running a quick script, or any
    one-off task the main agent wants to farm out. Its system prompt is a light,
    task-agnostic scaffold that works for a broad range of requests.
    """

    name = "generic"
    description = (
        "General-purpose sub-agent for tasks that do not fit a more specialised "
        "one. It can read/write files and run shell commands, so it suits most "
        "one-off jobs. Hands back a summary of what it did."
    )
    tools = ["read_file", "write_file", "grep", "glob", "run_shell_command"]
    enable_shell = True
    max_steps = 12

    def system_prompt(self, task, parent):
        base = super().system_prompt(task, parent)
        return base + (
            "\n\n---\nYou are a GENERAL-PURPOSE sub-agent: a capable, careful "
            "helper handed a single task. Follow these guidelines:\n"
            "* Start by making sure you understand the task. If anything is "
            "unclear, make a sensible, clearly-stated assumption and proceed -- "
            "do not stall.\n"
            "* Use the tools available to you: read before you write, and prefer "
            "small, correct steps over large, risky ones.\n"
            "* Prefer doing the task for real (editing files, running commands) "
            "over merely describing how it could be done.\n"
            "* If you run commands or make edits that could fail, check the "
            "result and adapt.\n"
            "When finished, write a concise summary of what you did and any "
            "decisions or follow-ups to report.md."
        )


# Register the built-in sub-agents.
register(ResearchSubAgent())
register(PlanSubAgent())
register(SoftwareEngineeringSubAgent())
register(GenericSubAgent())
