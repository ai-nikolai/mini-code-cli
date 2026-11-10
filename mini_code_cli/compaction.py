"""History-compaction strategies.

Every strategy starts from the same compact summary of the older transcript.
They differ in what else is carried into the new segment after that summary:

``summarise``
    Only the summary (the original behaviour).

``predict-files``
    Ask the model, once the summary exists, which files it needs to re-read to
    continue. The read-only tool calls it returns are executed and the
    assistant tool-call turn plus the tool results are inserted after the
    summary. The plan and the in-flight turn follow, as usual.

``predict-messages``
    Ask the model which original messages to keep verbatim, as integer
    indexes. Those messages are inserted after the summary (in their original
    order). The plan and the in-flight turn follow, as usual.

Adding a strategy is one :func:`register_strategy` call: the builder receives a
:class:`CompactionContext` and returns the messages to insert after the
summary. ``agent_loop.maybe_compact`` handles the surrounding summary, plan and
in-flight-tail wiring for every strategy.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

SUMMARISE = "summarise"
PREDICT_FILES = "predict-files"
PREDICT_MESSAGES = "predict-messages"

DEFAULT_COMPACTION_STRATEGY = SUMMARISE

#: Tools the ``predict-files`` strategy may execute. They only read, so a
#: compaction can never modify the workspace as a side effect.
READ_ONLY_TOOLS = ("read_file", "grep", "glob")

#: Safety caps: a prediction must not be able to refill the context.
MAX_PREDICTED_FILE_CALLS = 6
MAX_KEPT_MESSAGES = 20
MAX_STRATEGY_CHARS = 40_000


@dataclass
class CompactionContext:
    """Everything a strategy builder may need.

    ``predict_files`` and ``predict_messages`` are callables supplied by the
    owning session (they do the LLM call and tool execution); a strategy that
    does not use one simply ignores it.
    """

    summary: str
    messages: Sequence[dict]
    tail: Sequence[dict] = ()
    predict_files: Optional[Callable] = None
    predict_messages: Optional[Callable] = None
    extra: dict = field(default_factory=dict)


_BUILDERS = {}


def register_strategy(name):
    """Decorator: register a strategy builder under ``name``."""
    def decorate(builder):
        _BUILDERS[name] = builder
        return builder
    return decorate


def strategy_choices():
    """Registered strategy names, in registration order (built-ins first)."""
    return tuple(_BUILDERS)


def is_strategy(name):
    return name in _BUILDERS


def build_extra_messages(strategy, context):
    """Messages a strategy inserts after the summary (may be empty)."""
    builder = _BUILDERS.get(strategy)
    if builder is None:
        raise ValueError(f"unknown compaction strategy {strategy!r}")
    return list(builder(context) or ())


@register_strategy(SUMMARISE)
def _summarise(context):
    return []


@register_strategy(PREDICT_FILES)
def _predict_files(context):
    if context.predict_files is None:
        return []
    return context.predict_files(context)


@register_strategy(PREDICT_MESSAGES)
def _predict_messages(context):
    if context.predict_messages is None:
        return []
    return context.predict_messages(context)


def parse_keep_indexes(text):
    """Parse a model reply into a list of message indexes.

    Accepts a JSON array (the requested form), a JSON object with a ``keep``
    key, or a bare comma-separated list. Invalid pieces are ignored rather than
    failing the compaction.
    """
    if not text:
        return []
    match = re.search(r"\[[^\]]*\]", str(text), re.DOTALL)
    candidate = match.group(0) if match else str(text)
    try:
        parsed = json.loads(candidate)
    except ValueError:
        if match:
            return []
        parsed = [piece for piece in re.split(r"[,\s]+", str(text).strip())
                  if piece]
    if isinstance(parsed, dict):
        parsed = parsed.get("keep") or parsed.get("indexes") or []
    if not isinstance(parsed, list):
        return []
    indexes = []
    for value in parsed:
        try:
            indexes.append(int(value))
        except (TypeError, ValueError):
            continue
    return indexes


def cap_messages(messages, limit=MAX_STRATEGY_CHARS):
    """Truncate strategy-carried message content to ``limit`` characters total.

    A summary frees context; a strategy that re-inserts a huge tool result (or
    twenty large messages) would immediately re-trigger compaction. The cap is
    deliberately generous but finite, and marks each truncation so the model
    knows to read the file/tool again if it needs the rest.
    """
    remaining = max(int(limit), 0)
    capped = []
    for message in messages:
        message = dict(message) if isinstance(message, dict) else message
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and len(content) > remaining:
                message["content"] = (content[:remaining]
                                      + "\n[... truncated for compaction ...]")
            remaining = max(0, remaining - len(message.get("content") or ""))
        capped.append(message)
    return capped


def _coherent_indexes(messages, indexes):
    """Drop index selections that would break an assistant tool-call exchange.

    A kept assistant message with ``tool_calls`` needs the matching ``tool``
    messages immediately after it; keeping one without the other leaves a
    transcript the chat API may reject. We therefore keep a tool-call assistant
    only together with all of its results, and drop orphan tool messages.
    """
    selected = set(indexes)
    coherent = set()
    index = 0
    while index < len(messages):
        if index not in selected:
            index += 1
            continue
        message = messages[index]
        if not isinstance(message, dict):
            index += 1
            continue
        if message.get("role") == "assistant" and message.get("tool_calls"):
            call_ids = [call.get("id") for call in message.get("tool_calls") or ()
                        if isinstance(call, dict)]
            found = {}
            cursor = index + 1
            while cursor < len(messages) and isinstance(messages[cursor], dict) \
                    and messages[cursor].get("role") == "tool":
                tool_id = messages[cursor].get("tool_call_id")
                if tool_id in call_ids:
                    found[tool_id] = cursor
                cursor += 1
            if call_ids and all(call_id in found for call_id in call_ids):
                coherent.add(index)
                coherent.update(found.values())
                index = cursor
                continue
            index += 1
            continue
        if message.get("role") == "tool":
            # An orphan tool result (its assistant call was not kept).
            index += 1
            continue
        coherent.add(index)
        index += 1
    return coherent


def select_messages(messages, indexes, tail=(), exclude=(), limit=MAX_KEPT_MESSAGES):
    """Return the original messages named by ``indexes`` in original order.

    Indexes are **0-based**, matching the list the prompt shows the model and
    Python's own indexing. The system prompt, messages in ``exclude`` and the
    in-flight tail (which ``maybe_compact`` appends itself) are never selected;
    duplicates and out-of-range values are ignored. At most ``limit`` messages
    are returned, newest first when there are more, so a greedy prediction
    cannot refill the context.
    """
    tail_start = len(messages) - len(tail) if tail else len(messages)
    excluded = {int(index) for index in (exclude or ())}
    allowed = []
    for raw in indexes or ():
        try:
            index = int(raw)
        except (TypeError, ValueError):
            continue
        if index <= 0 or index >= len(messages) or index >= tail_start:
            continue
        if index in excluded:
            continue
        message = messages[index]
        if not isinstance(message, dict) or message.get("role") == "system":
            continue
        allowed.append(index)
    allowed = _coherent_indexes(messages, allowed)
    if len(allowed) > limit:
        # Keep the most recent messages, then repair tool pairs again: the cut
        # could otherwise leave a tool result without its assistant call.
        allowed = _coherent_indexes(messages, set(sorted(allowed)[-limit:]))
    selected = [(index, dict(messages[index])) for index in allowed]
    # The model may list indexes in any order; the kept messages always go back
    # in their original transcript order.
    selected.sort(key=lambda pair: pair[0])
    return [message for _index, message in selected]
