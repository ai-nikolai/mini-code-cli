"""mini_code_cli.tokens — cheap token estimation for the compaction decision.

Compaction has to decide *before* the next request whether the conversation will
still fit in the model's context window. Getting that number exactly right needs
the model's tokenizer, which we do not have; a local vLLM/SGLang server reports
``prompt_tokens`` only *after* a call, i.e. one round-trip too late to prevent an
overflow.

So two things are combined:

  * :func:`estimate_messages_tokens` -- a dependency-free heuristic (roughly four
    characters per token, plus a small per-message overhead for the role markers
    and chat-format boilerplate).
  * :func:`project_input_tokens` -- the server's own reported ``prompt_tokens``
    for the transcript it has already seen, plus the heuristic for whatever has
    been appended since (a tool result, an assistant reply, the injected plan).

The heuristic deliberately errs on the *high* side: compacting a little early
costs one summary call, whereas compacting too late costs a failed request in
the middle of a turn. Note the one known under-estimate -- tool schemas sent with
every request are not counted -- which is why the budget resolver keeps a
percentage of the window as slack (see :mod:`mini_code_cli.config`).
"""
import json
import math

# Roughly how many characters of text one token covers. English prose and code
# both sit near 4; JSON tool output is a little denser, so this is a slight
# over-estimate for those, which is the safe direction.
CHARS_PER_TOKEN = 4

# Per-message boilerplate: the role marker, the separators, and the tokens the
# chat template spends turning one message into the next.
MESSAGE_OVERHEAD_TOKENS = 4

# The "assistant has begun replying" suffix a chat template appends to prime the
# next completion. Counted once per request, not per message.
REPLY_PRIMING_TOKENS = 3


def estimate_text_tokens(text):
    """Estimate the tokens in a plain string (0 for empty/None)."""
    if not text:
        return 0
    return math.ceil(len(str(text)) / CHARS_PER_TOKEN)


def estimate_message_tokens(message):
    """Estimate the tokens one chat message contributes to a request.

    Counts the content, the JSON of any ``tool_calls`` (a tool call's arguments
    can be far larger than its content), any thinking-mode ``reasoning_content``
    (which the server concatenates back into the context on requests that carry
    ``tools``, so it really is prompt weight) and the name / tool-call-id
    fields, plus the fixed per-message overhead. Any non-dict input counts as 0.
    """
    if not isinstance(message, dict):
        return 0

    total = MESSAGE_OVERHEAD_TOKENS
    total += estimate_text_tokens(message.get("content"))
    # A thinking-mode turn keeps its chain of thought on the assistant message
    # and re-sends it (see utils.call_openai_server), so it occupies prompt
    # space in every later request that carries tools.
    total += estimate_text_tokens(message.get("reasoning_content"))

    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            total += estimate_text_tokens(json.dumps(tool_calls, default=str))
        except (TypeError, ValueError):
            # Something pathological in the payload: count it by repr instead of
            # raising, because a bad estimate must never break the loop.
            total += estimate_text_tokens(repr(tool_calls))

    for key in ("name", "tool_call_id"):
        value = message.get(key)
        if value:
            total += estimate_text_tokens(value)

    return total


def estimate_messages_tokens(messages):
    """Estimate the prompt tokens for a whole message list.

    Returns 0 for an empty/None transcript; otherwise the sum of the per-message
    estimates plus the one-off reply priming of the request.
    """
    if not messages:
        return 0
    return sum(estimate_message_tokens(m) for m in messages) + REPLY_PRIMING_TOKENS


def project_input_tokens(messages, reported_prompt_tokens=0, reported_message_count=0):
    """Estimate the input size of the *next* request for ``messages``.

    ``reported_prompt_tokens`` is what the server said the last request cost, and
    ``reported_message_count`` is how many messages that request actually
    contained (both come from the previous LLM call, see
    ``AgentSession._note_prompt_size``). Because the server's count is exact for
    the prefix it covered, it is used as-is and only the messages appended since
    are estimated on top -- which is what keeps the projection accurate as a turn
    grows, instead of only right after a reply.

    Falls back to a plain :func:`estimate_messages_tokens` when there is no
    usable report (no call yet, a server that reports nothing, or a transcript
    that has *shrunk* since -- the last case being a compaction or a ``/reset``,
    where the old report describes a conversation that no longer exists).
    """
    messages = messages or []
    reported = max(int(reported_prompt_tokens or 0), 0)
    seen = max(int(reported_message_count or 0), 0)

    if reported <= 0 or seen <= 0 or seen > len(messages):
        return estimate_messages_tokens(messages)

    return reported + estimate_messages_tokens(messages[seen:])
