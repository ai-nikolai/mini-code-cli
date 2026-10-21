"""mini_code_cli.persistence — everything about session histories on disk.

A session owns a directory under ``cache_dir/{session_name}/`` and keeps:

  * ``history_{n}.json`` — one file per conversation *segment*. A brand-new
    segment is started (``n -> n+1``) every time the history is compacted; a
    segment is overwritten every step with the latest messages. Because messages
    only ever *append* within a segment, any earlier message index still marks a
    faithful prefix -- so a recorded turn's ``index`` lets you rebuild the exact
    state at the end of that turn.

  * ``summary_history.json`` — a flat, high-level *recorder*: one record per
    turn (the running turn number, *what* happened -- an assistant reply, a
    tool call, or a compaction summary -- and which ``history_{n}.json`` +
    message index the turn lives in). It lets the user navigate and rewind the
    conversation without reading the full detailed histories.

  * ``usage_{n}.json`` — the token usage spent while segment *n* was live, so
    usage survives a compaction (which starts a new segment).

  * ``total_usage.json`` — the whole-session token usage, accumulated across
    every still-available ``usage_{n}.json``.

Every loader here is *best-effort*: a missing/garbled file must never break a
restore, so each function returns a safe default (or ``None``) instead of
raising.
"""
import json
import os
import re
import time

from . import ui
from .config import DEFAULT_CACHE_LOCATION


# ---------------------------------------------------------------------------
# directory / path helpers
# ---------------------------------------------------------------------------

def history_dir(cache_dir, session_name=""):
    """Return (creating it if needed) this session's on-disk history folder."""
    cache_dir = cache_dir or DEFAULT_CACHE_LOCATION
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, session_name), exist_ok=True)
    return os.path.join(cache_dir, session_name)


def resolve_session_dir(restore_session, cache_dir=None):
    """Resolve a ``--restore-session`` value to the session *directory* it names.

    This is the one place that decides what a user-supplied session reference
    means (it used to be two near-duplicate helpers, ``_resolve_history_path``
    and ``_restore_dir``, which disagreed on edge cases). It accepts:

      * ``None`` / ``""``              -> ``None`` (nothing to restore);
      * a path to a history file       -> the directory containing it, so both
                                          ``history.json`` and ``history_{n}.json``
                                          work;
      * a path to a session directory  -> that directory;
      * a bare session id             -> ``cache_dir/<id>`` when ``cache_dir`` is
                                          given, so ``--restore-session 1792-abc``
                                          works without spelling out the full path;
      * ``~``                          -> expanded.

    The result is *not* required to exist: callers decide whether a missing
    directory is fatal (a requested restore) or merely means "nothing to
    restore". Never raises.
    """
    if not restore_session:
        return None
    candidate = os.path.expanduser(str(restore_session))
    if os.path.isfile(candidate):
        return os.path.dirname(os.path.abspath(candidate))
    if not os.path.isdir(candidate) and cache_dir:
        # A bare session id: look for it inside the cache dir. Only accept it
        # when it resolves to something real, so a typo'd path stays a typo.
        under_cache = os.path.join(cache_dir, candidate)
        if os.path.isdir(under_cache):
            return under_cache
    return candidate


def generate_unique_id():
    """Generate a unique id based on the current time and a random component."""
    import time
    import uuid
    timestamp = int(time.time() * 1_000_000)  # microseconds
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# latest-session pointer (.mini)
# ---------------------------------------------------------------------------

# The .mini directory (inside the user's --allowed-dir) holds a tiny pointer to
# the most recent session. A previous session is restored *only* from this
# pointer -- there is deliberately NO timestamp/mtime fallback over the cache dir
# (find_last_session_dir below is pointer-only).
#
# NOTE: .mini is also the skills feature's directory. The pointer is a single
# dedicated file inside it (LAST_SESSION_FILENAME), so the two never collide and
# the skills code (which only reads <skill-name>/ + SKILLS.md) is unaffected.
SKILLS_DIRNAME = ".mini"
LAST_SESSION_FILENAME = "last_session.json"


def _last_session_dir_store(allowed_dir):
    """Return the .mini directory for ``allowed_dir`` (creating it if needed).

    Best-effort: never raises. Returns None if the directory cannot be ensured,
    in which case the pointer is simply not consulted/updated.
    """
    if not allowed_dir:
        return None
    store = os.path.join(os.path.abspath(allowed_dir), SKILLS_DIRNAME)
    try:
        os.makedirs(store, exist_ok=True)
        return store
    except OSError as e:
        ui.console.note(f"Could not use {store} for the latest-session pointer: {e}", "warn")
        return None


def save_last_session_pointer(allowed_dir, session_dir):
    """Record ``session_dir`` as the latest session in ``allowed_dir/.mini``.

    Writes a tiny JSON file (the absolute path plus a timestamp) so a later
    ``--load-last-session`` run can resume the most recent session without
    scanning the cache dir. This is the *write* half of the pointer that
    :func:`find_last_session_dir` reads; without it the reader always saw an
    empty store and ``--load-last-session`` could never fire.

    Written atomically (temp file + ``os.replace``) so a reader never observes a
    half-written pointer. Best-effort: never raises, returns the written path or
    None.
    """
    if not session_dir:
        return None
    store = _last_session_dir_store(allowed_dir)
    if store is None:
        return None
    path = os.path.join(store, LAST_SESSION_FILENAME)
    payload = {
        "session_dir": os.path.abspath(session_dir),
        "recorded_at": time.time(),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
        return path
    except OSError as e:
        ui.console.note(f"Could not record latest-session pointer: {e}", "warn")
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None


def load_last_session_pointer(allowed_dir):
    """Read the latest-session pointer recorded under ``allowed_dir/.mini``.

    Returns the recorded session directory (a path into the cache folder), or
    None if no pointer exists or it cannot be read. Best-effort: a missing or
    malformed pointer simply yields None so the caller falls back to the
    timestamp heuristic.
    """
    if not allowed_dir:
        return None
    store = os.path.join(os.path.abspath(allowed_dir), SKILLS_DIRNAME)
    path = os.path.join(store, LAST_SESSION_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    session_dir = payload.get("session_dir")
    if not session_dir or not os.path.isdir(session_dir):
        return None
    return session_dir


def find_last_session_dir(allowed_dir):
    """Return the most recent session dir recorded in the ``.mini`` pointer.

    A previous session is restored *only* when the ``.mini`` pointer (written by
    a prior run under ``--allowed-dir``) names one. There is deliberately NO
    timestamp/mtime fallback over the cache dir. Never raises: a missing/unreadable
    pointer simply yields ``None``.
    """
    return load_last_session_pointer(allowed_dir)


# history segments (history_{n}.json)
# ---------------------------------------------------------------------------

def save_history(messages, cache_dir, session_name="", history_index=0):
    """Persist a detailed conversation-history *segment* to disk."""
    d = history_dir(cache_dir, session_name)
    history_path = os.path.join(d, f"history_{history_index}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
    return history_path


def load_history_from_dir(directory):
    """Load the most complete history for a session directory.

    Prefers the newest ``history_{n}.json`` segment and falls back to the
    classic ``history.json``. Returns the messages list, or ``None`` if nothing
    could be loaded (never raises: a failed restore falls back to a fresh
    session).
    """
    if not directory or not os.path.isdir(directory):
        return None
    newest, highest = None, -1
    for fn in os.listdir(directory):
        m = re.fullmatch(r"history_(\d+)\.json", fn)
        if m and int(m.group(1)) > highest:
            highest, newest = int(m.group(1)), fn
    path = (os.path.join(directory, newest) if newest is not None
            else os.path.join(directory, "history.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        return messages if isinstance(messages, list) else None
    except Exception as e:
        ui.console.note(f"Failed to load history from {directory}: {e}", "error")
        return None


def load_history(restore_session):
    """Load a previous session's message history from disk (single-file path).

    Returns the messages list, or ``None`` if nothing could be loaded (missing
    file, wrong shape, or unreadable JSON). Never raises.
    """
    if not restore_session:
        return None
    directory = resolve_session_dir(restore_session)
    if os.path.isfile(os.path.expanduser(str(restore_session))):
        history_path = os.path.expanduser(str(restore_session))
    else:
        history_path = os.path.join(directory, "history.json")
    if not os.path.isfile(history_path):
        ui.console.note(f"No history found to restore from: {restore_session}", "warn")
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError("history file did not contain a message list")
    except Exception as e:
        ui.console.note(f"Failed to load history from {restore_session}: {e}", "error")
        return None
    return messages


# ---------------------------------------------------------------------------
# high-level recorder (summary_history.json)
# ---------------------------------------------------------------------------

def save_summary(summary, cache_dir, session_name=""):
    """Persist the high-level recorder to ``summary_history.json``."""
    d = history_dir(cache_dir, session_name)
    summary_path = os.path.join(d, "summary_history.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def load_summary_from_dir(directory):
    """Load a session's high-level recorder (``summary_history.json``).

    Returns the list of records, or an empty list if there is none or it cannot
    be read. Best-effort only.
    """
    if not directory or not os.path.isdir(directory):
        return []
    path = os.path.join(directory, "summary_history.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return saved if isinstance(saved, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# token usage (usage_{n}.json per segment, total_usage.json per session)
# ---------------------------------------------------------------------------
#
# Usage is recorded per history *segment*, mirroring ``history_{n}.json``:
# ``usage_{n}.json`` holds the tokens spent while segment *n* was the live one.
# That matters because history compaction rolls the conversation over to a fresh
# segment: a single ``usage.json`` could not say what the pre-compaction half of
# the session cost, so per-segment files let usage survive a compaction.
#
# ``total_usage.json`` is the whole-session accumulator. It is written from the
# running session total (which never decreases -- a rewind does not un-spend
# tokens), and :func:`load_total_usage_from_dir` reconstructs it by summing
# every still-available ``usage_{n}.json`` when the file is missing.

USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls", "tool_calls")


def empty_usage():
    """Return a fresh, all-zero usage dict (one key per :data:`USAGE_KEYS`)."""
    return {key: 0 for key in USAGE_KEYS}


def _coerce_usage(saved):
    """Normalise a loaded JSON payload into a complete, int-valued usage dict."""
    stats = empty_usage()
    if isinstance(saved, dict):
        for key in USAGE_KEYS:
            try:
                stats[key] = int(saved.get(key, 0) or 0)
            except (TypeError, ValueError):
                stats[key] = 0
    return stats


def accumulate_usage(token_stats, usage):
    """Add a single LLM call's token usage into the running totals."""
    if not usage:
        return
    token_stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    token_stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    token_stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    token_stats["llm_calls"] += 1


def _segment_usage_path(directory, history_index):
    return os.path.join(directory, f"usage_{int(history_index)}.json")


def save_usage_segment(token_stats, cache_dir, session_name="", history_index=0):
    """Persist one history segment's token usage to ``usage_{n}.json``."""
    d = history_dir(cache_dir, session_name)
    path = _segment_usage_path(d, history_index)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(token_stats, f, indent=2)
    return path


def load_usage_segment(directory, history_index=0):
    """Load one segment's usage (``usage_{n}.json``); zeros when absent.

    Best-effort: a missing or malformed file yields an all-zero dict so a restore
    never breaks on it.
    """
    if not directory:
        return empty_usage()
    try:
        with open(_segment_usage_path(directory, history_index), "r", encoding="utf-8") as f:
            return _coerce_usage(json.load(f))
    except Exception:
        return empty_usage()


def save_total_usage(total_stats, cache_dir, session_name=""):
    """Persist the whole-session token usage to ``total_usage.json``."""
    d = history_dir(cache_dir, session_name)
    path = os.path.join(d, "total_usage.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(total_stats, f, indent=2)
    return path


def load_usage_from_dir(directory):
    """Sum every still-available usage file in a session directory.

    That is all ``usage_{n}.json`` segments plus a legacy single ``usage.json``
    (written by older versions), so restoring an old session still recovers its
    totals. Best-effort: unreadable files are skipped, never raised.
    """
    stats = empty_usage()
    if not directory or not os.path.isdir(directory):
        return stats
    for fn in sorted(os.listdir(directory)):
        if not (re.fullmatch(r"usage_\d+\.json", fn) or fn == "usage.json"):
            continue
        try:
            with open(os.path.join(directory, fn), "r", encoding="utf-8") as f:
                part = _coerce_usage(json.load(f))
        except Exception:
            continue
        for key in USAGE_KEYS:
            stats[key] += part[key]
    return stats


def load_total_usage_from_dir(directory):
    """Load the whole-session totals (``total_usage.json``).

    Falls back to summing the segment files when it is missing/unreadable, and
    never reports less than the segments on disk record (a total may only grow).
    """
    if not directory or not os.path.isdir(directory):
        return empty_usage()
    segments = load_usage_from_dir(directory)
    path = os.path.join(directory, "total_usage.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            stats = _coerce_usage(json.load(f))
    except Exception:
        return segments
    for key in USAGE_KEYS:
        stats[key] = max(stats[key], segments[key])
    return stats
