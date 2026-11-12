"""Collect token usage from a vLLM server's Prometheus endpoint.

Agents do not agree on how (or whether) they report usage: harbor's Pi and
Claude Code agents expose ``agent_result.n_input_tokens``, while the mini-code
bridge reports none. vLLM, however, counts every prompt and generated token
behind all of them. Scraping ``/metrics`` before and after a trial therefore
gives a comparable, agent-neutral total.

The counters are cumulative for the life of the server, so callers take a
snapshot before the trial and subtract it from the post-trial snapshot. A
missing endpoint (a remote API, a server that does not expose metrics) simply
returns ``None`` and the benchmark falls back to whatever the agent reported.
"""
import re
import urllib.request

# vLLM has used both the plain and ``_total`` spellings across releases. Match
# the counter name exactly: histogram series (``_sum``/``_count``/``_bucket``)
# would otherwise be counted twice.
_COUNTERS = {
    "prompt_tokens": re.compile(r"^vllm:prompt_tokens(?:_total)?$"),
    "generation_tokens": re.compile(r"^vllm:generation_tokens(?:_total)?$"),
    "requests": re.compile(r"^vllm:request_success(?:_total)?$"),
}


def metrics_url(base_url):
    """The Prometheus endpoint for a vLLM server root (or ``/v1`` root)."""
    root = str(base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root + "/metrics"


def scrape(base_url, timeout=5):
    """Return cumulative token/request counters, or ``None`` if unavailable."""
    if not base_url:
        return None
    try:
        with urllib.request.urlopen(metrics_url(base_url), timeout=timeout) as handle:
            text = handle.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - metrics are best effort
        return None

    totals = {key: 0.0 for key in _COUNTERS}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        for key, pattern in _COUNTERS.items():
            if not pattern.match(name):
                continue
            try:
                value = float(line.rsplit(" ", 1)[1])
            except (IndexError, ValueError):
                continue
            totals[key] += value
            break
    return totals


def delta(before, after):
    """Per-trial counters from two cumulative snapshots.

    Returns ``None`` unless both snapshots exist. Negative values (the server
    was restarted mid-trial) are clamped to zero rather than reported as a
    nonsensical negative usage.
    """
    if not before or not after:
        return None
    return {
        f"{key}_delta": max(0, int(round(after.get(key, 0) - before.get(key, 0))))
        for key in _COUNTERS
    }
