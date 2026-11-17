"""Runtime recovery strategies: what to do when tool use keeps failing.

The agent loop can stay entirely reactive (``normal``) or add a heuristic
checkpoint when the same tool or the same call fails repeatedly:

  * ``normal`` -- no extra behaviour; the model sees tool errors and decides.
  * ``prompt-change`` -- after three consecutive failures of the same call (or
    of the same tool with different arguments), inject a user message telling
    the model to stop repeating the approach and try something else.
  * ``prompt-change-plan`` -- the same prompt, followed by a fresh planning
    request for the current goal, so the model re-plans instead of retrying.

The helpers below are deliberately pure and conservative: only clear failures
count, and the approach key groups calls that are really the same attempt
(same shell command, same file, same search, ...).
"""
import hashlib
import re

NORMAL = "normal"
PROMPT_CHANGE = "prompt-change"
PROMPT_CHANGE_PLAN = "prompt-change-plan"

RUNTIME_STRATEGIES = (NORMAL, PROMPT_CHANGE, PROMPT_CHANGE_PLAN)
DEFAULT_RUNTIME_STRATEGY = NORMAL

#: Consecutive failures of one approach before the loop speaks up.
FAILURE_THRESHOLD = 3

#: Safety cap so a model that ignores the hint cannot be buried in prompts.
MAX_PROMPTS_PER_TURN = 3

_EXIT_CODE_RE = re.compile(r"(?im)^\s*EXIT CODE:\s*(-?\d+)\s*$")
_NO_OUTPUT_EXIT_RE = re.compile(
    r"(?i)^\s*command produced no output \(exit code (-?\d+)\)\.?\s*$")

_FAILURE_PREFIXES = (
    "error:",
    "error ",
    "refused:",
    "command refused",
    "user denied",
    "unknown tool",
    "command timed out",
)


def tool_result_failed(result):
    """Whether a tool result string reports a clear failure.

    Shell failures carry ``EXIT CODE: N`` (or the no-output variant); every
    other tool reports ``Error: ...``-style text. Success output that merely
    mentions an error word is not treated as a failure, which keeps the
    heuristic from firing on search results or file contents.
    """
    text = str(result or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith(_FAILURE_PREFIXES):
        return True
    if "timed out" in lowered and lowered.startswith("error:"):
        return True
    for regex in (_EXIT_CODE_RE, _NO_OUTPUT_EXIT_RE):
        match = regex.search(text)
        if match:
            try:
                return int(match.group(1)) != 0
            except ValueError:
                return True
    return False


def approach_key(name, params):
    """A stable key for "the same attempt" across calls.

    Shell commands use the command text; file tools use the path; searches use
    the pattern/query. Anything else falls back to the tool name. Long values
    are hashed so two different large edits to the same file still share the
    key (they are the same approach) without building huge strings.
    """
    name = str(name or "")
    params = params if isinstance(params, dict) else {}

    if name == "run_shell_command":
        command = " ".join(str(params.get("command", "")).split())
        return f"{name}:{_digest(command)}"

    for field in ("file_path", "path", "target", "pattern", "query", "url",
                  "task", "name"):
        value = params.get(field)
        if value is not None and str(value).strip():
            return f"{name}:{field}={_digest(str(value).strip())}"
    return name


def _digest(value):
    if len(value) <= 180:
        return value
    return "sha1:" + hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()[:16]


def recovery_prompt(tool, attempts, same_call=False, result="",
                    can_search=False):
    """The message injected after repeated failures.

    ``same_call`` distinguishes "you repeated one command" from "the tool keeps
    failing with different inputs"; the advice is the same but the diagnosis
    is more useful when the call was literally repeated. ``can_search`` adds
    the docs/web hint only when the session actually exposes a search tool --
    asking for a tool the agent does not have would just add noise.
    """
    tool = tool or "the same tool"
    attempts = max(int(attempts or FAILURE_THRESHOLD), FAILURE_THRESHOLD)
    if same_call:
        headline = (f"You have tried {tool} {attempts} times with the same "
                    f"call and it failed every time.")
    else:
        headline = (f"You have tried {tool} {attempts} times and it failed "
                    f"every time.")
    excerpt = _failure_excerpt(result)
    lines = [
        headline,
        "Rethink your approach and try something else instead of repeating it.",
        "Choose a different command, tool, or method; if the current approach "
        "cannot work, establish what is actually available first.",
    ]
    if can_search:
        lines.append(
            "Search the documentation or the web (if your tools allow it) for "
            "the correct approach before trying again.")
    if excerpt:
        lines += ["", f"Last failure:\n{excerpt}"]
    return "\n".join(lines)


def _failure_excerpt(result, limit=400):
    text = " ".join(str(result or "").split())
    if not text:
        return ""
    return text[:limit] + ("…" if len(text) > limit else "")


def describe(strategy):
    """One-line human description used by settings/help text."""
    return {
        NORMAL: "no automatic recovery prompts",
        PROMPT_CHANGE: ("prompt for a different approach after 3 consecutive "
                        "failures of the same call or tool"),
        PROMPT_CHANGE_PLAN: ("prompt for a different approach and start a fresh "
                             "plan after 3 consecutive failures"),
    }.get(strategy, "unknown")


def normalise(strategy):
    """Return a valid strategy name, falling back to ``normal``."""
    return strategy if strategy in RUNTIME_STRATEGIES else DEFAULT_RUNTIME_STRATEGY
