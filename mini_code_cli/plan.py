"""mini_code_cli.plan — support for ``--plan`` mode.

In plan mode the agent keeps a small, human-readable "what am I working on"
record in the session directory so both the model and the user can track the
active goal and the plan for reaching it:

  * ``CURRENT_GOAL.md``  -- the user's latest instruction / goal (verbatim).
  * ``CURRENT_PLAN.md``  -- the agent's current plan for that goal, followed by
    a short running log (e.g. an entry recording when a goal was set/changed,
    and when it was finished).

The plan document itself is authored by the agent (the system prompt tells it
to, and it uses its ``write_file`` tool). This manager owns the parts the loop
is responsible for -- seeding/reading the goal, reading the plan back so history
compaction can re-inject it, and appending the "goal finished" / goal-change log
entries. It is deliberately free of any network or terminal dependency, and
every method is a no-op when disabled, so it is trivial to unit-test in
isolation.
"""

import os
from datetime import datetime

# The two tracking files kept in each session directory.
PLAN_FILENAME = "CURRENT_PLAN.md"
GOAL_FILENAME = "CURRENT_GOAL.md"


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
        # write_goal can tell a genuine change apart from a re-stated goal and
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
