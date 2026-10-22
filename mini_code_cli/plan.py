"""mini_code_cli.plan -- the tracking files behind ``--plan`` mode.

In plan mode the *loop* keeps two small, human-readable files in the session
directory (``cache_dir/<unique_id>``, independent of ``--allowed-dir``):

  * ``CURRENT_GOAL.md`` -- the user's latest instruction.
  * ``CURRENT_PLAN.md`` -- the generated step-by-step plan, plus the last
    plan-completion review.

:class:`PlanManager` is what the loop uses to read/write them directly (plain
``open``, not through an agent tool, so the records land even when the allowed
dir is elsewhere). The agent never authors these files: the loop writes the
goal, asks the model to author the plan, re-injects both after a compaction,
records the completion review, and marks the goal finished on shutdown.
"""

import os
from datetime import datetime

PLAN_FILENAME = "CURRENT_PLAN.md"
GOAL_FILENAME = "CURRENT_GOAL.md"

_PLAN_HEADER = "# Current Plan"
_GOAL_HEADER = "# Current Goal"
_SUMMARY_HEADER = "# Plan completion summary"


def _stamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PlanManager:
    """Reads/writes ``CURRENT_PLAN.md`` and ``CURRENT_GOAL.md`` for one session.

    ``session_dir`` is the session's on-disk folder. When ``enabled`` is False
    every method is a no-op, so callers can use the manager unconditionally.
    Every write is best-effort: a tracking hiccup must never break the loop.
    """

    def __init__(self, session_dir, enabled=True, goal=None):
        self.session_dir = session_dir
        self.enabled = enabled
        self.goal = goal
        # Last plan / last review generated for this goal, kept in memory so a
        # compaction re-injection and the files share one source.
        self.plan = None
        self.review_summary = None

    @property
    def plan_path(self):
        return os.path.join(self.session_dir, PLAN_FILENAME)

    @property
    def goal_path(self):
        return os.path.join(self.session_dir, GOAL_FILENAME)

    # -- writing ------------------------------------------------------------

    def write_goal(self, goal):
        """Persist the user's latest instruction (overwriting the previous goal)."""
        if not self.enabled:
            return
        self.goal = goal
        self._write(self.goal_path, f"{_GOAL_HEADER}\n\n{(goal or '').strip()}\n")

    def write_plan(self, plan):
        """Store the generated step-by-step plan for the current goal."""
        if not self.enabled:
            return
        self.plan = plan or ""
        body = self.plan.strip()
        body = f"{body}\n" if body else ""
        self._write(self.plan_path,
                    f"{_PLAN_HEADER}\n\n{body}\n_Updated: {_stamp()}_\n")

    def write_plan_summary(self, summary):
        """Record the plan-completion review under a fixed section header.

        The review replaces any previous one while the plan above it is
        preserved, so the file always shows the current plan + latest verdict.
        """
        if not self.enabled:
            return
        summary = summary or ""
        self.review_summary = summary
        current = self._read(self.plan_path) or ""
        head = current.split(_SUMMARY_HEADER, 1)[0]
        if _PLAN_HEADER not in head:
            head = f"{_PLAN_HEADER}\n"
        review = (
            f"{_SUMMARY_HEADER}\n"
            f"_Updated: {_stamp()}_\n"
            f"{summary.strip() or '(no summary generated)'}\n"
        )
        self._write(self.plan_path, head.rstrip("\n") + "\n\n" + review)

    def mark_finished(self, goal=None):
        """Append a timestamped "finished" entry to ``CURRENT_PLAN.md``."""
        if not self.enabled:
            return
        goal = goal or self.goal or (self._read(self.goal_path) or "").strip() or "current goal"
        head = (self._read(self.plan_path) or f"{_PLAN_HEADER}\n").rstrip("\n")
        self._write(self.plan_path, f"{head}\n\n- [{_stamp()}] Task finished: {goal}\n")

    # -- reading ------------------------------------------------------------

    def read_goal(self):
        return self._read(self.goal_path)

    def read_plan(self):
        """Return the plan text from disk, falling back to the last generated one."""
        if not self.enabled:
            return None
        text = self._read(self.plan_path)
        return text if text and text.strip() else self.plan

    def read_current_status(self):
        """Return goal + plan + last review as one flat block (for ``/plan``).

        The tracking files live outside ``--allowed-dir``, so the model has no
        tool to display them; this gives the interactive ``/plan`` command a
        single, ready-to-render view. Missing sections fall back to in-memory
        values or a "(none)" note rather than raising.
        """
        if not self.enabled:
            return None
        sections = [
            (_GOAL_HEADER, (self.goal or (self._read(self.goal_path) or "").strip())
             or "(no goal set)"),
            (_PLAN_HEADER, (self.plan or (self._read(self.plan_path) or "").strip())
             or "(no plan generated yet)"),
            (_SUMMARY_HEADER, (self.review_summary or "").strip()
             or "(no completion review yet)"),
        ]
        return "\n\n".join(f"{header}\n\n{body}" for header, body in sections)

    # -- disk helpers -------------------------------------------------------

    def _write(self, path, content):
        if not self.enabled:
            return
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass

    def _read(self, path):
        if not self.enabled:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None
