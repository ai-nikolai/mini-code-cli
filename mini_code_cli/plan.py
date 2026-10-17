"""mini_code_cli.plan -- support for ``--plan`` mode.

In plan mode the *loop* keeps two small, human-readable tracking files in the
session directory (``cache_dir/<unique_id>`` -- i.e. the *cache* dir,
independent of ``--allowed-dir``):

  * ``CURRENT_GOAL.md``  -- the user's latest instruction / goal (verbatim).
  * ``CURRENT_PLAN.md``  -- a running log of the requests made, and a
    "Task finished: <goal>" marker once the goal is complete.

:class:`PlanManager` is what the *loop* uses to read/write those files directly
(plain ``open`` -- not through any agent tool -- so the records land even when
the allowed dir is elsewhere). The agent itself does not author these files:
the loop writes the goal after each user request, logs the request into the
plan, re-injects them into the conversation after a history compaction, and
marks the goal finished on shutdown.
"""

import os
from datetime import datetime

PLAN_FILENAME = "CURRENT_PLAN.md"
GOAL_FILENAME = "CURRENT_GOAL.md"


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

    @property
    def plan_path(self):
        return os.path.join(self.session_dir, PLAN_FILENAME)

    @property
    def goal_path(self):
        return os.path.join(self.session_dir, GOAL_FILENAME)

    def _append(self, text):
        """Append one dated line to CURRENT_PLAN.md (creating it if needed).

        Shared by ``append_plan_line`` and ``mark_finished``. Best-effort and
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

    def _write(self, path, content):
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass

    def write_goal(self, goal):
        """Persist the user's latest instruction to CURRENT_GOAL.md (overwriting
        any previous goal) and remember it in ``self.goal`` so
        :meth:`mark_finished` can name the goal later.

        Best-effort: a tracking hiccup must never break the agent loop.
        """
        if not self.enabled:
            return
        self.goal = goal
        self._write(self.goal_path, f"# Current Goal\n\n{(goal or '').strip()}\n")

    def append_plan_line(self, text):
        """Log one user request (or other note) into CURRENT_PLAN.md.

        The loop calls this after every real user request so the running plan
        reflects the work as it proceeds. Best-effort and never raises.
        """
        if not self.enabled:
            return
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append(f"- [{stamp}] {text}")

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
