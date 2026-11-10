"""Agent-owned files kept in ``--allowed-dir/.mini``.

Two opt-in features put a file the *agent* may read and write there, rather
than in the session directory the loop owns:

  * the **scratchpad** (``--scratchpad``): a free-form note file for thoughts,
    plans and learnings, plus as many additional scratch files as the agent
    wants to create;
  * the **open plan** (``--plan-strategy open-plan``): a plan document the
    agent maintains itself with the normal file tools.

Both live under the same ``.mini`` directory as the skills folder and the
latest-session pointer, so they are inside the ``--allowed-dir`` sandbox and
the normal ``read_file`` / ``write_file`` / ``edit`` tools can reach them. The
directory name is read from :mod:`mini_code_cli.persistence` at call time, so
the test suite can redirect it globally (exactly like the session pointer).
"""

import os

from . import persistence

SCRATCHPAD_FILENAME = "SCRATCHPAD.md"
OPEN_PLAN_FILENAME = "PLAN.md"

_SCRATCHPAD_HEADER = """# Scratchpad

This file belongs to the agent. It may read and write it freely and may create
additional scratch files in the same directory. Use it for thoughts, plans,
learnings and notes that should survive a compaction.
"""

_OPEN_PLAN_HEADER = """# Agent Plan

This plan document belongs to the agent. Read it before continuing, keep it up
to date as work progresses, and mark steps done or outstanding.
"""


def mini_dir(allowed_dir):
    """The ``.mini`` directory inside ``allowed_dir`` (not created)."""
    return os.path.join(os.path.abspath(allowed_dir), persistence.SESSION_DIRNAME)


def scratchpad_path(allowed_dir):
    """Path to the main scratch-pad file."""
    return os.path.join(mini_dir(allowed_dir), SCRATCHPAD_FILENAME)


def open_plan_path(allowed_dir):
    """Path to the agent-authored plan document."""
    return os.path.join(mini_dir(allowed_dir), OPEN_PLAN_FILENAME)


def ensure_scratchpad(allowed_dir):
    """Create the main scratch-pad file (and its directory) if missing."""
    return _ensure_file(scratchpad_path(allowed_dir), _SCRATCHPAD_HEADER)


def ensure_open_plan(allowed_dir):
    """Create the agent plan document (and its directory) if missing."""
    return _ensure_file(open_plan_path(allowed_dir), _OPEN_PLAN_HEADER)


def read_text(path, limit=None):
    """Read ``path`` as text, returning ``""`` when it cannot be read.

    ``limit`` caps the returned characters (the open-plan re-injection after a
    compaction should not be able to re-fill the context with the document).
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return ""
    if limit is not None and len(text) > limit:
        return text[:limit] + "\n[... truncated ...]\n"
    return text


def _ensure_file(path, header):
    """Best-effort create-with-header; never raises, always returns ``path``."""
    try:
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(header)
    except OSError:
        pass
    return path
