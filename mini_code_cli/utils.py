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

    except requests.exceptions.RequestException as e:
        print(f"Error calling OpenAI Server...: {e}")
        return None

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None


def unpack_llm_result(result):
    """Split a :func:`call_openai_server` result into its four fields.

    Returns ``(content, tool_calls, usage, reasoning_content)``. The reasoning
    text needs its own slot (rather than being folded into ``content``) because
    thinking-mode models require it to be passed back verbatim, yet it must
    never be shown to the user as the assistant's reply.

    Legacy callers and test stubs that still return the old 3-tuple
    ``(content, tool_calls, usage)`` keep working: a short tuple simply yields
    ``reasoning_content=None``, and a non-tuple (e.g. ``None``) yields four
    ``None``s. This lives here so every caller unpacks the same way.
    """
    if result is None:
        return None, None, None, None
    try:
        parts = tuple(result)
    except TypeError:
        return result, None, None, None
    content = parts[0] if len(parts) > 0 else None
    tool_calls = parts[1] if len(parts) > 1 else None
    usage = parts[2] if len(parts) > 2 else None
    reasoning_content = parts[3] if len(parts) > 3 else None
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
    :func:`unpack_llm_result` to unpack either shape.

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

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout,
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

        result = response.json().get("choices", [{}])[0].get("message", {})
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
        usage = response.json().get("usage", {}) or {}
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
        print(f"HTTP Error: {e}")
        # Only probe the response further if we actually got one back. A
        # request-level failure (timeout, connection refused, DNS) leaves
        # ``response`` as None, and there is nothing to inspect then.
        if response is not None:
            try:
                # Print the status code / raw body to see the detailed error.
                print(f"Status Code: {response.status_code}")
                print(f"Response Body: {response.text}")
                # If the response is JSON, parse and pretty-print it.
                try:
                    error_details = response.json()
                    print("Parsed Error Details:")
                    print(json.dumps(error_details, indent=2))
                except json.JSONDecodeError:
                    print("Response body is not valid JSON.")
            except Exception as e:
                print(f"Problem with (error) response: {e}")
        return None, None, None, None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None, None, None, None
    except Exception as e:  # noqa: BLE001 -- a network/parse hiccup must not crash the loop
        print(f"Unexpected error calling OpenAI server: {e}")
        return None, None, None, None
