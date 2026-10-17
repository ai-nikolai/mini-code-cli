"""mini_code_cli.plan -- support for ``--plan`` mode.

In plan mode the agent keeps two small, human-readable tracking files in the
session directory (``cache_dir/<unique_id>`` -- i.e. the *cache* dir,
independent of ``--allowed-dir``):

  * ``CURRENT_GOAL.md``  -- the user's latest instruction / goal (verbatim).
  * ``CURRENT_PLAN.md``  -- the agent's current plan, with a finished entry
    appended when the goal is done.

:class:`PlanManager` is what the *loop* uses to write/read those files directly
(plain ``open`` -- not the file tools -- so the records land even when the
allowed dir is elsewhere).

The ``read_plan`` / ``write_plan`` *functions* below are the *agent*'s tools.
They are confined to the session directory via :func:`mini_code_cli.sandbox.path_within`,
and only the two known plan filenames may live there -- so the tracking files
cannot be relocated or symlinked out, and no stray files are created.
"""

import os
import functools
from datetime import datetime

from . import sandbox

PLAN_FILENAME = "CURRENT_PLAN.md"
GOAL_FILENAME = "CURRENT_GOAL.md"
# Only these names may live inside the session directory -- keeps the plan
# tools from creating stray files and from relocating the tracking files.
PLAN_ALLOWED_NAMES = (PLAN_FILENAME, GOAL_FILENAME)


class PlanManager:
    """Reads/writes ``CURRENT_PLAN.md`` and ``CURRENT_GOAL.md`` for a session.

    ``session_dir`` is the session's on-disk folder. When ``enabled`` is False
    every method is a no-op, so callers can use the manager unconditionally.

    Best-effort throughout: a tracking hiccup must never break the agent loop.
    """

    def __init__(self, session_dir, enabled=True, goal=None):
        self.session_dir = session_dir
        self.enabled = enabled
        self.goal = goal
        # Last goal logged into CURRENT_PLAN.md, so re-stating the same goal
        # doesn't spam the plan with duplicate "goal updated" entries.
        self._last_logged_goal = None

    @property
    def plan_path(self):
        return os.path.join(self.session_dir, PLAN_FILENAME)

    @property
    def goal_path(self):
        return os.path.join(self.session_dir, GOAL_FILENAME)

    def _append(self, text):
        """Append one dated line to CURRENT_PLAN.md (creating it if needed)."""
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

    def _write(self, path, content):
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass

    def write_goal(self, goal):
        """Persist the user's latest instruction to CURRENT_GOAL.md and, in
        plan mode, log the goal into CURRENT_PLAN.md only on a genuine change."""
        if not self.enabled:
            return
        stripped = (goal or "").strip()
        self._write(self.goal_path, f"# Current Goal\n\n{stripped}\n")
        if stripped and stripped != self._last_logged_goal:
            verb = "Goal set to" if self._last_logged_goal is None else "Goal updated to"
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._append(f"- [{stamp}] {verb}: {stripped}")
            self._last_logged_goal = stripped
        self.goal = goal

    def mark_finished(self, goal=None):
        """Append a timestamped "finished" entry to CURRENT_PLAN.md."""
        if not self.enabled:
            return
        goal = goal or self.goal or self._read(self.goal_path) or "current goal"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append(f"- [{stamp}] Task finished: {goal}")

    def read_plan(self):
        return self._read(self.plan_path)

    def read_goal(self):
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
# Tools: read_plan / write_plan (agent-facing, session-dir confined)
# ---------------------------------------------------------------------------

def _resolved(file_path, default, session_dir):
    """Resolve ``file_path`` inside ``session_dir`` if it is one of the allowed
    plan files, else None. Relative paths are joined onto ``session_dir``."""
    if not file_path:
        file_path = default
    resolved = os.path.realpath(
        file_path if os.path.isabs(file_path) else os.path.join(session_dir, file_path))
    if not sandbox.path_within(resolved, session_dir) \
            or os.path.basename(resolved) not in PLAN_ALLOWED_NAMES:
        return None
    return resolved


def read_plan(file_path=None, session_dir=None):
    """Read a plan tracking file (CURRENT_PLAN.md / CURRENT_GOAL.md)."""
    if not session_dir:
        return "Error: plan mode is not available in this context."
    resolved = _resolved(file_path, PLAN_FILENAME, session_dir)
    if resolved is None:
        return f"Error: only {PLAN_FILENAME} / {GOAL_FILENAME} may be read."
    if not os.path.isfile(resolved):
        return None
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Error: {e}"


def write_plan(file_path, content=None, session_dir=None):
    """Write a plan tracking file (CURRENT_PLAN.md / CURRENT_GOAL.md)."""
    if not session_dir:
        return "Error: plan mode is not available in this context."
    if not file_path:
        return f"Error: a file_path is required ({PLAN_FILENAME} / {GOAL_FILENAME})."
    resolved = _resolved(file_path, None, session_dir)
    if resolved is None:
        return f"Error: only {PLAN_FILENAME} / {GOAL_FILENAME} may be written."
    try:
        os.makedirs(session_dir, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as f:
            f.write(content or "")
        return f"Wrote to {resolved}"
    except OSError as e:
        return f"Error: {e}"


def _plan_functions(session_dir):
    """{name: functools.partial} for the two plan tools, session_dir bound."""
    return {
        "read_plan": functools.partial(read_plan, session_dir=session_dir),
        "write_plan": functools.partial(write_plan, session_dir=session_dir),
    }


def get_plan_tools(session_dir):
    """Return the {name: {function, description}} mapping for the plan tools."""
    from . import tools
    return tools.util_get_tools_dict(functions=_plan_functions(session_dir))


def get_plan_tool_schemas(session_dir):
    """Return the OpenAI tool schemas for the two plan tools."""
    from . import tools
    return tools.util_get_tools(functions=_plan_functions(session_dir))
