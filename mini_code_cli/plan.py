"""mini_code_cli.plan -- support for ``--plan`` mode.

In plan mode the *loop* keeps two small, human-readable tracking files in the
session directory (``cache_dir/<unique_id>`` -- i.e. the *cache* dir,
independent of ``--allowed-dir``):

  * ``CURRENT_GOAL.md``  -- the user's latest instruction / goal.
  * ``CURRENT_PLAN.md``  -- a running log of the requests made, plus the
    current, generated step-by-step plan for the active goal and a review of
    whether it is complete (marked finished once the goal is complete).

:class:`PlanManager` is what the *loop* uses to read/write those files directly
(plain ``open`` -- not through any agent tool -- so the records land even when
the allowed dir is elsewhere). The agent itself does not author these files:
the loop writes the goal after each user request, asks the model (with no
tools) to author the plan from the whole conversation, saves it, re-sends it
into the conversation, re-injects both after a history compaction, reviews
completion once the work looks done, and marks the goal finished on shutdown.
"""

import os
from datetime import datetime

PLAN_FILENAME = "CURRENT_PLAN.md"
GOAL_FILENAME = "CURRENT_GOAL.md"

# A marker section that holds the latest plan-completion review.
_SUMMARY_HEADER = "# Plan completion summary"


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
        # The last plan the loop generated for the current goal. Kept in memory
        # so a compaction re-injection (and the plan file) share one source.
        self.plan = None
        # The last plan-completion review the loop generated (see
        # write_plan_summary), so a compaction re-injection can restate it.
        self.review_summary = None

    @property
    def plan_path(self):
        return os.path.join(self.session_dir, PLAN_FILENAME)

    @property
    def goal_path(self):
        return os.path.join(self.session_dir, GOAL_FILENAME)

    def _append(self, text):
        """Append one dated line to CURRENT_PLAN.md (creating it if needed).

        Used by ``mark_finished`` (and ``write_plan_summary`` for its header
        note). Best-effort and never raises: a plan-tracking hiccup must never
        break the agent loop.
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

    def write_plan(self, plan):
        """Store the generated step-by-step plan for the current goal.

        The loop asks the model (with no tools) to author the plan from the
        whole conversation, then calls this to keep ``self.plan`` and the
        ``CURRENT_PLAN.md`` file in sync with it. Best-effort and never raises.
        """
        if not self.enabled:
            return
        plan = plan or ""
        self.plan = plan
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(self.plan_path, "w", encoding="utf-8") as f:
                f.write("# Current Plan\n\n")
                if plan.strip():
                    f.write(plan.strip() + "\n")
                f.write(f"\n_Updated: {stamp}_\n")
        except OSError:
            pass

    def write_plan_summary(self, summary):
        """Record a plan-completion review into CURRENT_PLAN.md and memory.

        After the model reviews the plan (marks tasks done / outstanding and
        emits one summary line, e.g. "ALL TASKS DONE. Go back to user." or
        "TASKS still in progress." + which ones remain), the loop calls this to
        keep ``self.review_summary`` and the ``CURRENT_PLAN.md`` file in sync.

        The summary is written under a fixed "# Plan completion summary"
        section, overwriting any previous review so the file always reflects the
        latest conclusion. Best-effort and never raises.
        """
        if not self.enabled:
            return
        summary = summary or ""
        self.review_summary = summary
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = summary.strip() or "(no summary generated)"
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            # Read whatever is currently in the file so we can preserve the plan
            # above it and only swap the review section in place.
            try:
                with open(self.plan_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except OSError:
                content = ""
            head = content.split(_SUMMARY_HEADER, 1)[0] if _SUMMARY_HEADER in content else content
            # Make sure the plan body exists above the summary section.
            if "# Current Plan" not in head:
                head = "# Current Plan\n\n"
            review = (
                "\n" + _SUMMARY_HEADER + "\n"
                f"_Updated: {stamp}_"
                f"\n{body}\n"
            )
            with open(self.plan_path, "w", encoding="utf-8") as f:
                f.write(head.rstrip("\n") + "\n" + review)
        except OSError:
            pass

    def read_plan(self):
        """Return the current plan text from disk (or ``self.plan`` if absent)."""
        if not self.enabled:
            return None
        text = self._read(self.plan_path)
        return text if text is not None and text.strip() else self.plan

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

    def mark_finished(self, goal=None):
        """Append a timestamped "finished" entry to CURRENT_PLAN.md."""
        if not self.enabled:
            return
        goal = goal or self.goal or self._read(self.goal_path) or "current goal"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._append(f"- [{stamp}] Task finished: {goal}")
