"""mini_code_cli.plan — support for ``--plan`` mode.

In plan mode the agent keeps a small, human-readable "what am I working on"
record in the session directory so both the model and the user can track the
active goal and the plan for reaching it:

  * ``CURRENT_GOAL.md``  -- the user's latest instruction / goal (verbatim).
  * ``CURRENT_PLAN.md``  -- the agent's current plan for that goal, followed by
    a short running log (e.g. an entry recording when a goal was finished).

The plan document itself is authored by the agent (the system prompt tells it
to, and it uses its ``write_file`` tool). This manager owns the parts the loop
is responsible for -- seeding/reading the goal, reading the plan back so history
compaction can re-inject it, and appending the "goal finished" log entry. It is
deliberately free of any network or terminal dependency, and every method is a
no-op when disabled, so it is trivial to unit-test in isolation.
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
    """

    def __init__(self, session_dir, enabled=True, goal=None):
        self.session_dir = session_dir
        self.enabled = enabled
        # The last goal this manager was asked to record, so callers can tell a
        # new goal apart from the previous one. Kept in sync by write_goal().
        self.goal = goal

    # -- paths -------------------------------------------------------------
    @property
    def plan_path(self):
        return os.path.join(self.session_dir, PLAN_FILENAME)

    @property
    def goal_path(self):
        return os.path.join(self.session_dir, GOAL_FILENAME)

    # -- writing -----------------------------------------------------------
    def write_goal(self, goal):
        """Persist the user's current goal / last instruction to CURRENT_GOAL.md.

        Overwrites any previous goal -- the file always reflects the *latest*
        instruction. Best-effort and never raises: a plan-tracking hiccup must
        never break the agent loop.
        """
        if not self.enabled:
            return
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(self.goal_path, "w", encoding="utf-8") as f:
                f.write("# Current Goal\n\n")
                f.write((goal or "").strip() + "\n")
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
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            if not os.path.exists(self.plan_path):
                with open(self.plan_path, "w", encoding="utf-8") as f:
                    f.write("# Current Plan\n\n")
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.plan_path, "a", encoding="utf-8") as f:
                f.write(f"\n- [{stamp}] Task finished: {goal}\n")
        except OSError:
            pass

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
