import requests
import json


def oneline(text, limit=None):
    """Collapse ``text`` onto a single line, optionally truncated to ``limit``.

    Multi-line tool output, file contents and model replies are pointless (and
    disruptive) in a one-line status message, so they are flattened: every line
    is stripped, the lines are joined with single spaces, and runs of whitespace
    collapse. When ``limit`` is given the result is truncated with a trailing
    ellipsis if it does not fit.

    This is the single implementation used by the console renderers, the agent
    loop and the sub-agent runner (it previously existed three times, so the
    three call sites could drift apart).
    """
    one = " ".join(line.strip() for line in str(text).splitlines())
    one = " ".join(one.split())
    if limit is not None and len(one) > limit:
        return one[:limit] + "..."
    return one


def plural(count, noun, suffix="s"):
    """``"1 message"`` / ``"4 messages"`` -- for text a human reads.

    Lives here, next to :func:`oneline`, because both the agent loop (compaction
    notes, the good-bye box) and the console (usage lines) build user-facing
    counts; three inline ``'' if n == 1 else 's'`` expressions had already
    drifted before this was factored out.
    """
    return f"{count} {noun}" + ("" if count == 1 else suffix)


def _coerce_usage_int(value):
    """Best-effort: coerce a raw usage field to a non-negative int.

    Servers sometimes report ``null``/``None`` or oddly-shaped numbers for token
    usage. This normalises those to 0 so a downstream
    ``accumulate_usage``/``usage["tool_calls"]`` assignment can never raise.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _usage_or_zero(value):
    """Return ``int(value)`` when that is safe, else 0 (for summing fields)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def determine_vllm_model(server_url, api_key=None, timeout=120):
    """In case of a vLLm local server, read the first served model id from
    ``{server_url}/v1/models`` so the caller does not have to name the model.

    Returns the model id, or ``None`` if the server could not be reached or
    did not return a usable response (never raises).
    """
    url = f"{server_url}/v1/models"

    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()

        result = response.json().get('data', [{}])[0]
        return result.get("id", "")

    except requests.exceptions.RequestException:
        # Silent by design: this is a best-effort probe, and the caller already
        # warns the user (see AgentSession._resolve_model). Printing here used to
        # scribble over the session banner.
        return None

    except (json.JSONDecodeError, KeyError, IndexError):
        return None


def unpack_llm_result(result):
    """Split a :func:`call_openai_server` result into its four fields.

    Returns ``(content, tool_calls, usage, reasoning_content)``. The reasoning
    text needs its own slot (rather than being folded into ``content``) because
    thinking-mode models require it to be passed back verbatim, yet it must
    never be shown to the user as the assistant's reply.

    The shape is fixed: the 4-tuple above, or ``None`` when the call failed
    (which yields four ``None``s). Anything else is a caller error and raises --
    a short 3-tuple used to be accepted silently, which quietly discarded a
    reply's reasoning text instead of surfacing the mismatch.
    """
    if result is None:
        return None, None, None, None
    content, tool_calls, usage, reasoning_content = result
    return content, tool_calls, usage, reasoning_content


def call_openai_server(messages, max_tokens=2048, temperature=0.0, model="default",
                       server_url=None, api_key=None, tools=None, timeout=300):
    """Query an OpenAI-compatible server (local or remote) using ``requests``.

    Supports authentication via API key if provided.

    ``timeout`` is the number of seconds before the HTTP request is aborted.
    Reasoning models can take a while to produce a response, so the default is
    deliberately high.

    Returns ``(content, tool_calls, usage, reasoning_content)``. On any failure
    it prints a diagnostic and returns ``(None, None, None, None)`` so the
    caller can degrade gracefully instead of crashing; use
    :func:`unpack_llm_result` to unpack that 4-tuple.

    ``reasoning_content`` is the chain-of-thought text that thinking-mode
    servers (DeepSeek V3.2+ ``deepseek-reasoner``, ...) return alongside
    ``content``. It is returned separately -- never merged into ``content`` --
    because it must be preserved verbatim and sent back on the *assistant*
    message: for requests that carry ``tools``, the API rejects the follow-up
    request with HTTP 400 ("The `reasoning_content` in the thinking mode must
    be passed back to the API") if any earlier assistant turn dropped it. It is
    also not user-visible output, so the console never renders it.
    """
    url = f"{server_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Note: the request timeout is a *transport* concern and is passed to
    # ``requests.post(timeout=...)`` below, not put in the JSON body -- it is
    # not a field of the chat-completions API, and strict gateways reject
    # unexpected keys.
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    # Bound up front so that a failure *during* the request itself (a timeout,
    # a refused connection, DNS failure, ...) does not leave ``response``
    # unassigned -- otherwise the error handler below crashes with
    # "local variable 'response' where it is not associated with a value".
    response = None
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()

        # Parse the body once: a second ``response.json()`` call re-decodes it
        # and can raise on a server that sends a slightly different body.
        body = response.json()
        result = body.get("choices", [{}])[0].get("message", {})
        content = result.get("content", "")
        tool_calls = result.get("tool_calls", [])
        # Thinking-mode servers return the chain of thought next to ``content``.
        # Keep it verbatim (including an empty string, which is what such a
        # model returns when it produced no reasoning) and keep it *distinct*
        # from the reply. ``None`` means "this server/model did not return the
        # field" -- in that case the caller omits it rather than inventing a
        # value, since a fabricated reasoning_content would be sent back as if
        # the model had produced it.
        reasoning_content = result.get("reasoning_content")

        # Token usage, when the server reports it (many local servers do not).
        usage = body.get("usage", {}) or {}
        prompt_tokens = _coerce_usage_int(usage.get("prompt_tokens", 0))
        completion_tokens = _coerce_usage_int(usage.get("completion_tokens", 0))
        total_tokens = _coerce_usage_int(
            usage.get("total_tokens",
                      _usage_or_zero(usage.get("prompt_tokens"))
                      + _usage_or_zero(usage.get("completion_tokens"))))

        return content, tool_calls, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }, reasoning_content

    except requests.exceptions.RequestException as e:
        _report_request_failure(e, response)
        return None, None, None, None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        _report(f"Error processing the model's response: {e}")
        return None, None, None, None
    except Exception as e:  # noqa: BLE001 -- a network/parse hiccup must not crash the loop
        _report(f"Unexpected error calling the model server: {e}")
        return None, None, None, None


def _report(message, level="error"):
    """Report a request failure through the console, never straight to stderr.

    ``ui`` imports :func:`oneline` from this module, so the console is imported
    lazily here to keep the import graph acyclic. A single note -- rather than a
    multi-line ``print`` -- keeps the transient region intact: this can fire
    while the "… thinking" spinner owns the bottom line, and writing multi-line
    text underneath it left the screen mangled.
    """
    from . import ui  # local: ui imports oneline from here
    ui.console.note(oneline(message, 300), level)


def _report_request_failure(error, response):
    """Explain an HTTP failure, folding the server's own words into one line.

    Only the response is inspected more closely when there *is* one: a
    request-level failure (timeout, refused connection, DNS) leaves ``response``
    as None, and there is nothing to inspect then. The body is the useful part
    (servers put the real reason there), so it is kept -- collapsed and
    truncated -- instead of dumping the whole payload into the terminal.
    """
    if response is None:
        _report(f"HTTP error talking to the model server: {error}")
        return
    status = getattr(response, "status_code", "?")
    detail = ""
    try:
        body = response.text or ""
        try:
            body = json.dumps(response.json())
        except json.JSONDecodeError:
            pass  # not JSON: the raw body is the best we have
        if body.strip():
            detail = f": {oneline(body, 200)}"
    except Exception as e:  # a broken response must not break the report
        detail = f" (and the error body was unreadable: {e})"
    _report(f"HTTP {status} from the model server: {error}{detail}")
