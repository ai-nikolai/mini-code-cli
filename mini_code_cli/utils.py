import requests
import json


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


def call_openai_server(messages, max_tokens=2048, temperature=0.0, model="default",
                       server_url=None, api_key=None, tools=None, timeout=300):
    """Query an OpenAI-compatible server (local or remote) using ``requests``.

    Supports authentication via API key if provided.

    ``timeout`` is the number of seconds before the HTTP request is aborted.
    Reasoning models can take a while to produce a response, so the default is
    deliberately high.

    Returns ``(content, tool_calls, usage)``. On any failure it prints a
    diagnostic and returns ``(None, None, None)`` so the caller can degrade
    gracefully instead of crashing.
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

        # Token usage, when the server reports it (many local servers do not).
        usage = response.json().get("usage", {}) or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return content, tool_calls, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    except requests.exceptions.RequestException as e:
        print(f"HTTP Error: {e}")
        try:
            print(f"Status Code: {response.status_code}")
            # Print the raw response body to see the detailed validation error.
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
        return None, None, None
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error processing response: {e}")
        return None, None, None
