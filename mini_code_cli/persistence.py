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

The on-disk shape above is the *only* one this module reads. A restore is
strict: the recorder must be a readable ``summary_history.json`` and the segment
its last record names must be a readable ``history_{n}.json``. Session files
written by older versions (a single ``history.json`` / ``usage.json``, or a
folder whose recorder was never written) are not accepted, and a folder that
does not match is refused so the caller can start a fresh session instead.

The loaders themselves never raise: a missing or garbled file yields a safe
default (``None`` for the transcript and the recorder, zeros for usage) plus an
error note, so a damaged folder degrades to "cannot restore" rather than a
traceback.
"""
import json
import os
import re
import time

from . import ui
from .config import DEFAULT_CACHE_LOCATION, SESSION_DIRNAME


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
      * a path to a session directory  -> that directory;
      * a bare session id             -> ``cache_dir/<id>`` when ``cache_dir`` is
                                          given, so ``--restore-session 1792-abc``
                                          works without spelling out the full path;
      * ``~``                          -> expanded.

    Only *directories* are resolved. A path to a file used to resolve to its
    parent directory, so ``--restore-session history_0.json`` quietly restored
    whichever session happened to live above it -- the user named a file and got
    a conversation. A value that names no existing directory is now returned
    unchanged (with ``~`` expanded) instead of guessed at: the restore then
    fails with a message naming the value, rather than adopting a neighbour.

    The result is *not* required to exist: callers decide whether a missing
    directory is fatal (a requested restore) or merely means "nothing to
    restore". Never raises.
    """
    if not restore_session:
        return None
    candidate = os.path.expanduser(str(restore_session))
    if os.path.isdir(candidate):
        return candidate
    if cache_dir:
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
# latest-session pointer (``<allowed_dir>/.mini/last_session.json``)
# ---------------------------------------------------------------------------
#
# The session directory (inside the user's --allowed-dir) holds a tiny pointer to
# the most recent session. A previous session is restored *only* from this
# pointer -- there is deliberately NO timestamp/mtime fallback over the cache dir
# (find_last_session_dir below is pointer-only).
#
# NOTE: by default this is the same ``.mini`` directory the skills feature uses
# (see :data:`mini_code_cli.config.SESSION_DIRNAME`). The pointer is a single
# dedicated file inside it (LAST_SESSION_FILENAME), so the two never collide and
# the skills code (which only reads <skill-name>/ + SKILLS.md) is unaffected.
#
# ``SESSION_DIRNAME`` is a module-level *variable* (not just an import) so a test
# suite can point it somewhere of its own -- tests use ``.mini_tests`` so a test
# run can never write into a real ``.mini`` (the user's skills + resume pointer).
LAST_SESSION_FILENAME = "last_session.json"


def pointer_dir(allowed_dir):
    """Return the absolute pointer/session-metadata dir inside ``allowed_dir``.

    No side effects (nothing is created); ``None`` when ``allowed_dir`` is empty.
    """
    if not allowed_dir:
        return None
    return os.path.join(os.path.abspath(allowed_dir), SESSION_DIRNAME)


def _last_session_dir_store(allowed_dir):
    """Return the pointer directory for ``allowed_dir`` (creating it if needed).

    Best-effort: never raises. Returns None if the directory cannot be ensured,
    in which case the pointer is simply not consulted/updated.
    """
    store = pointer_dir(allowed_dir)
    if store is None:
        return None
    try:
        os.makedirs(store, exist_ok=True)
        return store
    except OSError as e:
        ui.console.note(f"Could not use {store} for the latest-session pointer: {e}", "warn")
        return None


def save_last_session_pointer(allowed_dir, session_dir):
    """Record ``session_dir`` as the latest session in ``<allowed_dir>/.mini``.

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
    """Read the latest-session pointer recorded under ``<allowed_dir>/.mini``.

    Returns the recorded session directory (a path into the cache folder), or
    None if no pointer exists or it cannot be read. Best-effort: a missing or
    malformed pointer simply yields None.
    """
    store = pointer_dir(allowed_dir)
    if store is None:
        return None
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


# atomic JSON writes
# ---------------------------------------------------------------------------
#
# Every history-shaped file is rewritten whole, many times per session. A plain
# ``open(path, "w")`` truncates the file first, so an interrupt (Ctrl-C), a
# crash or a full disk in the middle of ``json.dump`` leaves a *truncated* file
# behind -- and that session can then never be restored. Writing to a temporary
# file in the same directory and ``os.replace``-ing it into place makes the swap
# atomic: a reader (or a later restore) sees either the old file or the new one,
# never half of each.

def _atomic_write_json(path, payload, indent=2):
    """Write ``payload`` as JSON to ``path`` atomically, returning ``path``.

    The temporary file lives in the same directory (so ``os.replace`` stays a
    same-filesystem rename, which is atomic). On any failure the temp file is
    removed and the exception propagates -- callers decide whether that is fatal
    (a normal save) or something to roll back (``/reset``).
    """
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


# history segments (history_{n}.json)
# ---------------------------------------------------------------------------

_HISTORY_SEGMENT_RE = re.compile(r"history_(\d+)\.json")
_USAGE_SEGMENT_RE = re.compile(r"usage_(\d+)\.json")


def history_segment_indices(directory):
    """Return the sorted segment indices present as ``history_{n}.json``.

    Empty list when the directory is missing or holds no segments. Never raises.
    """
    return _segment_indices(directory, _HISTORY_SEGMENT_RE)


def usage_segment_indices(directory):
    """Return the sorted segment indices present as ``usage_{n}.json``."""
    return _segment_indices(directory, _USAGE_SEGMENT_RE)


def _segment_indices(directory, pattern):
    """Sorted integers extracted from filenames in ``directory`` matching ``pattern``."""
    indices = set()
    if not directory or not os.path.isdir(directory):
        return []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    for fn in names:
        m = pattern.fullmatch(fn)
        if m:
            indices.add(int(m.group(1)))
    return sorted(indices)


def delete_segments_above(directory, keep_index):
    """Delete every history/usage segment whose index is above ``keep_index``.

    A ``/reset`` that lands on an *earlier* segment must not leave the segments
    above it behind: a later restore prefers the newest segment, so the orphaned
    ``history_{m}.json`` would resurrect exactly the turns the user rewound past.
    Removing them (and their ``usage_{m}.json``) makes the rewind stick.

    Each file is removed independently and a failure is reported as a warning
    rather than raised -- the rewind itself has already been persisted, so a
    stray undeletable file must not undo it. Returns the list of removed paths.
    """
    removed = []
    if not directory or not os.path.isdir(directory):
        return removed
    try:
        keep = int(keep_index)
    except (TypeError, ValueError):
        return removed
    indices = sorted(set(history_segment_indices(directory))
                     | set(usage_segment_indices(directory)))
    for idx in indices:
        if idx <= keep:
            continue
        for name in (f"history_{idx}.json", f"usage_{idx}.json"):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            try:
                os.unlink(path)
                removed.append(path)
            except OSError as e:
                ui.console.note(f"Could not remove {name}: {e}", "warn")
    return removed


def save_history(messages, cache_dir, session_name="", history_index=0):
    """Persist a detailed conversation-history *segment* to disk (atomically)."""
    d = history_dir(cache_dir, session_name)
    history_path = os.path.join(d, f"history_{history_index}.json")
    return _atomic_write_json(history_path, messages)


def _load_message_list(path):
    """Load one history file as a message list, or ``None`` if it is unusable.

    "Unusable" covers a missing file, unreadable/invalid JSON, a payload that is
    not a list, and an empty list (a segment always starts with at least the
    system prompt, so an empty one cannot be a real conversation). Never raises.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)
    except Exception:
        return None
    return messages if isinstance(messages, list) and messages else None


def load_history_with_index(directory, preferred_index):
    """Load the ``history_{n}.json`` segment a session's recorder points at.

    The segment index is *mandatory* and always comes from the recorder's last
    record (see ``AgentSession.restore_previous_session``), so the messages come
    from the segment the user was really looking at. There is deliberately no
    "newest readable segment wins" scan and no single-file ``history.json``
    fallback any more: the scan is what used to undo a ``/reset`` that had
    rewound onto an older segment (the orphaned higher segment won the restore),
    and both guessed at layouts that are no longer written.

    Returns ``(messages, index)``, where ``index`` is the segment number the
    messages came from. Returns ``(None, None)`` when the index is not a
    non-negative integer or that segment is missing, empty or garbled; the
    *caller* reports it (a restore knows which note to print). Never raises.
    """
    if not directory or not os.path.isdir(directory):
        return None, None

    try:
        index = int(preferred_index)
    except (TypeError, ValueError):
        index = -1
    if index < 0:
        return None, None

    messages = _load_message_list(os.path.join(directory, f"history_{index}.json"))
    if messages is None:
        return None, None
    return messages, index


def load_history_from_dir(directory, preferred_index):
    """Load the message list of the segment ``preferred_index`` names.

    Thin wrapper over :func:`load_history_with_index` for callers that only need
    the messages. There is no default index on purpose: a caller has to say
    *which* segment it wants, so "load whatever happens to be on disk" cannot
    creep back in. Returns the messages list, or ``None`` (never raises).
    """
    messages, _ = load_history_with_index(directory, preferred_index)
    return messages


# ---------------------------------------------------------------------------
# high-level recorder (summary_history.json)
# ---------------------------------------------------------------------------

def save_summary(summary, cache_dir, session_name=""):
    """Persist the high-level recorder to ``summary_history.json`` (atomically)."""
    d = history_dir(cache_dir, session_name)
    summary_path = os.path.join(d, "summary_history.json")
    return _atomic_write_json(summary_path, summary)


def load_summary_from_dir(directory):
    """Load a session's high-level recorder (``summary_history.json``).

    Returns the list of records, which may legitimately be empty -- that is a
    real, if turn-less, session (started, then quit before the first turn
    finished). ``None`` means there is no usable recorder at all: the file is
    missing, unreadable, or does not hold a list. The two cases are deliberately
    distinct, because only the second one makes a folder unrestorable.
    """
    if not directory or not os.path.isdir(directory):
        return None
    path = os.path.join(directory, "summary_history.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        return None
    return saved if isinstance(saved, list) else None


# ---------------------------------------------------------------------------
# token usage (usage_{n}.json per segment, total_usage.json per session)
# ---------------------------------------------------------------------------
#
# Usage is recorded per history *segment*, mirroring ``history_{n}.json``:
# ``usage_{n}.json`` holds the tokens spent while segment *n* was the live one.
# That matters because history compaction rolls the conversation over to a fresh
# segment: a single whole-session usage file could not say what the
# pre-compaction half of the session cost, so per-segment files let usage
# survive a compaction.
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
    """Persist one history segment's token usage to ``usage_{n}.json`` (atomically)."""
    d = history_dir(cache_dir, session_name)
    path = _segment_usage_path(d, history_index)
    return _atomic_write_json(path, token_stats)


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
    """Persist the whole-session token usage to ``total_usage.json`` (atomically)."""
    d = history_dir(cache_dir, session_name)
    path = os.path.join(d, "total_usage.json")
    return _atomic_write_json(path, total_stats)


def load_usage_from_dir(directory):
    """Sum the ``usage_{n}.json`` segments of a session directory.

    Only per-segment files are read. Best-effort: an unreadable segment is
    skipped and a session with no usage files simply reads as zeros (usage is
    optional -- a restore never depends on it).
    """
    stats = empty_usage()
    if not directory or not os.path.isdir(directory):
        return stats
    for index in usage_segment_indices(directory):
        try:
            with open(_segment_usage_path(directory, index), "r", encoding="utf-8") as f:
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
