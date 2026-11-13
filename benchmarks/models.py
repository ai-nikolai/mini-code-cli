"""Local model serving (H200) and the Anthropic-compatible proxy path.

Two very different endpoints are needed, and confusing them is the most common
way to lose an afternoon:

``vLLM`` (OpenAI-compatible, ``/v1/chat/completions``)
    Serves the local checkpoint. Consumed directly by mini-code and Pi, and by
    Claude Code only through the proxy below.

``LiteLLM in Anthropic mode`` (``/v1/messages``)
    Claude Code speaks the Anthropic Messages API and nothing else, so a local
    model has to be translated. ``litellm --config ...`` exposes both surfaces;
    ``ANTHROPIC_BASE_URL`` then points Claude Code at it.

Neither command is executed by this module -- they are built as argv so ``plan``
can print them and a human can run them on the GPU host. Starting a model server
is an explicit, separate step, because a 35B checkpoint takes minutes to load and
must outlive several benchmark runs.
"""
import json
import urllib.error
import urllib.request

DEFAULT_PORT = 30_000
DEFAULT_PROXY_PORT = 40_000


def vllm_argv(model, port=DEFAULT_PORT, tensor_parallel=1, max_model_len=None,
              tool_call_parser="qwen3_xml", reasoning_parser="qwen3",
              extra=()):
    """The ``vllm serve`` command line for one endpoint.

    ``--enable-auto-tool-choice`` plus a tool-call parser is what makes a local
    model usable as an agent at all (without it the model cannot emit tool calls
    and every benchmark measures the harness's error handling).
    """
    argv = [
        "vllm", "serve", model,
        "--host", "0.0.0.0",              # containers must be able to reach it
        "--port", str(int(port)),
        "--tensor-parallel-size", str(int(tensor_parallel)),
        "--enable-auto-tool-choice",
        "--tool-call-parser", str(tool_call_parser),
        "--trust-remote-code",
    ]
    if reasoning_parser:
        argv += ["--reasoning-parser", str(reasoning_parser)]
    if max_model_len:
        argv += ["--max-model-len", str(int(max_model_len))]
    argv += [str(item) for item in extra]
    return argv


def litellm_config(model, base_url, proxy_port=DEFAULT_PROXY_PORT):
    """A LiteLLM config that fronts ``model`` at ``base_url`` in both protocols.

    ``model_list`` gives it the OpenAI-compatible upstream; ``litellm_settings``
    keeps request/response logging off by default, because benchmark prompts and
    patches routinely contain repository secrets.
    """
    return {
        "model_list": [{
            "model_name": model,
            "litellm_params": {"model": f"openai/{model}", "api_base": base_url,
                               "api_key": "sk-no-key-required"},
        }],
        "litellm_settings": {"drop_params": True},
        "general_settings": {"master_key": None},
        "router_settings": {"routing_strategy": "simple-shuffle"},
        "server_settings": {"port": int(proxy_port)},
    }


def litellm_argv(config_path, proxy_port=DEFAULT_PROXY_PORT):
    """``litellm --config ... --port ...``: exposes ``/v1/messages`` for Claude Code."""
    return ["litellm", "--config", str(config_path), "--port", str(int(proxy_port))]


def write_litellm_config(model, base_url, path, proxy_port=DEFAULT_PROXY_PORT):
    """Write the proxy config and return ``path`` (JSON is valid YAML)."""
    payload = litellm_config(model, base_url, proxy_port)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return path


def health_url(base_url):
    """The endpoint a vLLM/SGLang server answers ``/health`` on."""
    return base_url.rstrip("/") + "/health"


def wait_for_health(base_url, timeout=1_200, interval=5.0, opener=None):
    """Poll ``/health`` until the server answers 200 or the budget runs out.

    Returns ``(healthy, waited_seconds)``; never raises, because "the model did
    not come up" is a normal benchmark precondition failure, not a crash.
    """
    import time

    opener = opener or urllib.request.urlopen
    started = time.monotonic()
    while True:
        waited = time.monotonic() - started
        try:
            with opener(health_url(base_url), timeout=5) as response:
                if getattr(response, "status", 200) == 200:
                    return True, waited
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if waited >= timeout:
            return False, waited
        time.sleep(interval)


def endpoint_kwargs(model_spec):
    """The ``(base_url, model, api)`` triplet an adapter needs, as a dict."""
    return {"base_url": model_spec.base_url, "model": model_spec.model,
            "api": model_spec.api}
