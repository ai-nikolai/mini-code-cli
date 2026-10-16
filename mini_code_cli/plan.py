"""mini_code_cli.plan -- support for ``--plan`` mode.

In plan mode the agent keeps a small, human-readable "what am I working on"
record in the session directory so both the model and the user can track the
active goal and the plan for reaching it:

  * ``CURRENT_GOAL.md``  -- the user's latest instruction / goal (verbatim).
  * ``CURRENT_PLAN.md``  -- the agent's current plan for that goal, followed by
    a short running log (e.g. an entry recording when a goal was set/changed,
    and when it was finished).

Two sandbox-scoped tools (:func:`read_plan` / :func:`write_plan`) let the agent
read and rewrite these files from *inside* the loop. Unlike :class:`PlanManager`
(which the loop uses directly), these go through the normal tool path and are
confined to the session directory via :func:`mini_code_cli.sandbox.path_within`,
so the prompt telling the agent to maintain the tracking files cannot be evaded.

The plan files always live in the session directory (``cache_dir/<unique_id>``)
-- i.e. the *cache* dir, independent of ``--allowed-dir``. Because that dir is
usually outside ``--allowed-dir``, the two tools confine themselves to it and
the prompt also points the agent at the concrete path. A read/write outside the
session directory, or of any file that is not one of the two plan files, is a
no-op that returns an error, so the tracking files cannot be moved or symlinked
out.
"""

import os
from datetime import datetime

from . import sandbox

# The two tracking files kept in each session directory.
PLAN_FILENAME = "CURRENT_PLAN.md"
GOAL_FILENAME = "CURRENT_GOAL.md"

# The only names an agent read/write of a session-dir file may touch. This keeps
# the plan tools from creating stray files in the session directory.
PLAN_ALLOWED_NAMES = (PLAN_FILENAME, GOAL_FILENAME)

# Tool names contributed when plan mode is enabled (exposed to the agent).
PLAN_TOOL_NAMES = ("read_plan", "write_plan")


class PlanManager:
    """Reads/writes ``CURRENT_PLAN.md`` and ``CURRENT_GOAL.md`` for a session.

    ``session_dir`` is the session's on-disk folder (the one holding
    ``history_{n}.json``). When ``enabled`` is False every method is a no-op, so
    callers can use the manager unconditionally and let the ``--plan`` flag
    decide whether anything actually happens.

    ``session_dir`` is the session directory (``cache_dir/<unique_id>``) -- i.e.
    the files always live in the *cache* dir, independent of ``--allowed-dir``.
    The manager writes with plain ``open`` (not through the file tools), so the
    records are created even when the allowed dir is a different location.
    """

    def __init__(self, session_dir, enabled=True, goal=None):
        self.session_dir = session_dir
        self.enabled = enabled
        # The last goal this manager was asked to record, so callers can tell a
        # new goal apart from the previous one. Kept in sync by write_goal().
        self.goal = goal
        # The last goal (stripped) that was logged into CURRENT_PLAN.md, so
        # write_goal can tell a genuine change apart from the re-stated goal and
        # avoid spamming the plan with duplicate "goal updated" entries.
        self._last_logged_goal = None

    # -- paths -------------------------------------------------------------
    @property
    def plan_path(self):
        return os.path.join(self.session_dir, PLAN_FILENAME)

    @property
    def goal_path(self):
        return os.path.join(self.session_dir, GOAL_FILENAME)

    # -- writing -----------------------------------------------------------
    def _append_plan_note(self, text):
        """Append one dated line to CURRENT_PLAN.md (creating it if needed).

        Shared by the goal-change log and ``mark_finished``. Best-effort and
        never raises: a plan-tracking hiccup must never break the agent loop.
        """
        if not self.enabled:
            return
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            if not os.path.exists(self.plan_path):
                with open(self.plan_path, "w", encoding="utf-8") as f:
                    f.write("# Current Plan\n\n")
            with open(self.plan_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass

    def write_goal(self, goal):
        """Persist the user's current goal / last instruction to CURRENT_GOAL.md.

        Overwrites any previous goal -- the file always reflects the *latest*
        instruction. Also makes sure ``CURRENT_PLAN.md`` exists in the cache dir
        (so both tracking files are present from the very first goal), and -- in
        plan mode -- logs the goal into CURRENT_PLAN.md the *first* time it is
        set and again whenever it changes, so the plan records goal pivots.
        Best-effort and never raises.
        """
        if not self.enabled:
            return
        stripped = (goal or "").strip()
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            # Ensure CURRENT_PLAN.md exists in the cache dir too, so the goal
            # always has a companion plan file sitting next to it.
            if not os.path.exists(self.plan_path):
                with open(self.plan_path, "w", encoding="utf-8") as f:
                    f.write("# Current Plan\n\n")
            with open(self.goal_path, "w", encoding="utf-8") as f:
                f.write("# Current Goal\n\n")
                f.write(stripped + "\n")
            # Log the goal change into the plan -- but only on a genuine change
            # (compared against the last goal actually logged), so re-stating the
            # same goal on later turns does not spam the plan log.
            if stripped and stripped != self._last_logged_goal:
                verb = "Goal set to" if self._last_logged_goal is None else "Goal updated to"
                stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._append_plan_note(f"- [{stamp}] {verb}: {stripped}")
                self._last_logged_goal = stripped
            self.goal = goal
        except OSError:
            pass

    def mark_finished(self, goal=None):
        """Append a timestamped entry to CURRENT_PLAN.md marking a goal done.

        Creates the plan document (with a header) if the agent never wrote one,
        so there is always a record that the goal was completed. Best-effort and
        never raises.
        """
        if not self.enabled:
            return
        # Prefer the goal the loop last recorded, then try reading the goal file
        # back off disk (so the plan still names the right goal even if the loop
        # never captured one itself, e.g. a one-shot --prompt preset), and only
        # then fall back to a generic description.
        goal = goal or self.goal or self._read(self.goal_path) or "current goal"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append_plan_note(f"- [{stamp}] Task finished: {goal}")

    # -- reading -----------------------------------------------------------
    def read_plan(self):
        """Return the contents of CURRENT_PLAN.md, or None if absent/unreadable.

        Never raises: a missing plan (e.g. the agent has not written one yet)
        simply yields None so the caller can skip it.
        """
        return self._read(self.plan_path)

    def read_goal(self):
        """Return the contents of CURRENT_GOAL.md, or None if absent/unreadable."""
        return self._read(self.goal_path)

    def _read(self, path):
        if not self.enabled:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None


# ---------------------------------------------------------------------------
# Tools: read_plan / write_plan
#
# These are the tools the *agent* uses to maintain the tracking files. They are
# confined to the session directory (the cache dir, usually outside
# --allowed-dir) via the shared sandbox primitive, and only the two known plan
# filenames may live there -- so the tracking files cannot be relocated and no
# stray files are created in the session directory.
# ---------------------------------------------------------------------------

def _plan_file_name(file_path, default, session_dir):
    """Return ``(resolved_path, name)`` for ``file_path`` inside ``session_dir``,
    or ``(None, None)`` when not allowed.

    Relative ``file_path`` values (e.g. ``CURRENT_PLAN.md``, which the prompt
    instructs the agent to use) are joined onto ``session_dir``; absolute values
    are honored verbatim. Either way the resolved path must live inside
    ``session_dir`` and resolve to one of the allowed plan file names, so the
    tracking files cannot be relocated outside the cache directory.
    """
    if file_path:
        resolved = os.path.realpath(os.path.join(session_dir, file_path)
                                    if not os.path.isabs(file_path) else file_path)
    else:
        resolved = os.path.realpath(os.path.join(session_dir, default or PLAN_FILENAME))
    if not sandbox.path_within(resolved, session_dir):
        return None, None
    name = os.path.basename(resolved)
    if name not in PLAN_ALLOWED_NAMES:
        return None, None
    return resolved, name


def _session_dir_ok(session_dir):
    return session_dir is not None


def read_plan(file_path=None, session_dir=None):
    """Read a plan tracking file (CURRENT_PLAN.md or CURRENT_GOAL.md).

    Parameters: file_path (optional). Only a file inside the session directory
    is honored; the target must be one of the two plan file names.
    """
    if not _session_dir_ok(session_dir):
        return "Error: plan mode is not available in this context."
    resolved, name = _plan_file_name(file_path, PLAN_FILENAME, session_dir)
    if resolved is None:
        return (f"Error: only {PLAN_FILENAME} / {GOAL_FILENAME} may be read from "
                f"the session directory.")
    if not os.path.isfile(resolved):
        return None
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Error: {e}"


def write_plan(file_path, content=None, session_dir=None):
    """Write to a plan tracking file (CURRENT_PLAN.md or CURRENT_GOAL.md).

    Parameters: file_path, content. Only a file inside the session directory is
    honored, and it must be one of the two plan file names, so the tracking
    files cannot be relocated outside the cache directory.
    """
    if not _session_dir_ok(session_dir):
        return "Error: plan mode is not available in this context."
    if not file_path:
        return "Error: a file_path is required (CURRENT_PLAN.md or CURRENT_GOAL.md)."
    resolved, name = _plan_file_name(file_path, None, session_dir)
    if resolved is None:
        return (f"Error: only {PLAN_FILENAME} / {GOAL_FILENAME} may be written to "
                f"the session directory.")
    try:
        os.makedirs(session_dir, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Wrote to {resolved}"
    except OSError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Wiring (tool schemas + bound functions)
# ---------------------------------------------------------------------------

def _plan_functions(session_dir):
    """{name: functools.partial} for the two plan tools, session_dir bound."""
    import functools
    return {
        "read_plan": functools.partial(read_plan, session_dir=session_dir),
        "write_plan": functools.partial(write_plan, session_dir=session_dir),
    }


def get_plan_tools(session_dir):
    """Return the {name: {function, description}} mapping for the plan tools.

    Mirrors ``skills.get_skills_tools_dict``: the four plan tools are built
    directly with ``session_dir`` already bound, so the agent (and the runner)
    receive a clean schema and bound callables without extra wiring.
    """
    from . import tools
    return tools.util_get_tools_dict(functions=_plan_functions(session_dir))


def get_plan_tool_schemas(session_dir):
    """Return the OpenAI tool schemas for the two plan tools."""
    from . import tools
    return tools.util_get_tools(functions=_plan_functions(session_dir))
