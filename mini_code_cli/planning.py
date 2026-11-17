"""Planning strategies for ``--plan`` mode.

The default is ``plan-verify2``, the machine-readable successor to the
``plan-verify`` strategy that won the clean benchmark comparison:

  * the loop asks the model to author a plan;
  * after the work settles it asks for a short verdict ending in
    ``STATUS: DONE`` / ``STATUS: INCOMPLETE`` / ``STATUS: IN PROGRESS``;
  * a DONE verdict ends the turn immediately. An unfinished verdict injects
    the outstanding work and lets the agent act; another verdict follows the
    next time the work settles. If the agent makes no tool call after an
    unfinished verdict, the turn ends instead of verifying forever.

Alternative strategies:

  * ``plan-verify`` -- author a plan, review it once the work settles, then
    keep the post-review step only when it calls a tool. A no-tool answer is
    shown to the user and ends the turn without being stored.
  * ``plan-verify-step`` -- the original behaviour: same plan + review, then
    always run one extra step and keep whatever that step says.
  * ``open-plan`` -- the agent owns a plan document in ``--allowed-dir/.mini``
    and reads/writes it with the normal file tools; the loop neither authors
    nor reviews a plan.

Keeping the names here (rather than inline in ``agent_loop`` or ``argparse``)
means a future strategy only has to be added in one place plus the loop branch
that implements it.
"""
import re

PLAN_VERIFY_STEP = "plan-verify-step"
PLAN_VERIFY = "plan-verify"
PLAN_VERIFY2 = "plan-verify2"
OPEN_PLAN = "open-plan"

PLAN_STRATEGIES = (PLAN_VERIFY_STEP, PLAN_VERIFY, PLAN_VERIFY2, OPEN_PLAN)
DEFAULT_PLAN_STRATEGY = PLAN_VERIFY2

#: Outcomes returned by :func:`parse_verify_status`.
VERIFY_DONE = "done"
VERIFY_INCOMPLETE = "incomplete"

_STATUS_LINE_RE = re.compile(
    r"(?im)^\s*(?:status|verdict|result)\s*[:\-]\s*(?P<state>[^\n]+?)\s*$")

_INCOMPLETE_RE = re.compile(
    r"(?i)\b(?:incomplete|unfinished|in[\s_-]*progress|not[\s_-]*done|"
    r"not[\s_-]*complete|todo|pending|blocked)\b")

_DONE_RE = re.compile(
    r"(?i)\b(?:done|complete|completed|finished|resolved)\b")


def parse_verify_status(text):
    """Extract the machine-readable completion verdict from a verify reply.

    The loop asks the verifier to end with exactly one line such as
    ``STATUS: DONE`` or ``STATUS: INCOMPLETE``. This turns that line into
    :data:`VERIFY_DONE`, :data:`VERIFY_INCOMPLETE`, or ``None`` when the reply
    did not state a usable verdict.

    A ``STATUS:`` line wins over prose, and only the last one counts: the
    verifier may discuss steps before stating its verdict. If there is no such
    line, only the final non-empty line is considered, so an analysis sentence
    ("the build is not complete") cannot flip the verdict of a later explicit
    marker. Anything ambiguous returns ``None``; callers treat that as
    unfinished but stop if the agent takes no action.
    """
    text = str(text or "")
    matches = list(_STATUS_LINE_RE.finditer(text))
    if matches:
        return _status_from_state(matches[-1].group("state"))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return _status_from_state(lines[-1])
    return None


def _status_from_state(state):
    """Normalise one status phrase; ``None`` when it is not a known verdict."""
    state = str(state or "").strip().strip(".:!*_`()[]")
    if not state:
        return None
    # Check the unfinished spellings first: "INCOMPLETE" contains "COMPLETE".
    if _INCOMPLETE_RE.search(state):
        return VERIFY_INCOMPLETE
    if _DONE_RE.search(state):
        return VERIFY_DONE
    return None
