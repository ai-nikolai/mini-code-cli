"""mini_code_cli.config — process-wide default settings.

Kept in one small module so every other module (the agent loop, persistence,
the tool runner, sub-agents, ...) can share the same defaults without importing
each other and creating import cycles.
"""
import os
from typing import NamedTuple

# Where session histories live on disk by default.
DEFAULT_CACHE_LOCATION = os.path.expanduser("~/.cache/mini_code/sessions/")

# The LLM server the CLI talks to by default (a local vLLM/SGLang endpoint).
DEFAULT_MODEL_URL = "http://0.0.0.0:30000"

# Sampling + operational defaults.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOOL_TIMEOUT = 120     # seconds before a single tool call is aborted
DEFAULT_HTTP_TIMEOUT = 500     # seconds before an LLM HTTP request is aborted

# History compaction is triggered by how full the *context window* is -- and by
# nothing else. The number of messages is deliberately NOT a trigger: it is a
# poor proxy for what makes a request fail (a handful of large tool outputs can
# overflow a window that a hundred short turns would not), and a message-count
# cap compacts healthy conversations early for no reason.


# ---------------------------------------------------------------------------
# context windows / compaction budget
# ---------------------------------------------------------------------------
#
# Approximate *context window* sizes, in tokens, for the model families this CLI
# is likely to talk to (a local vLLM/SGLang server, or a hosted endpoint).
#
# These are deliberately round, published figures rather than exact tokenizer
# limits: the value is only used to decide when to compact, and the resolution
# below subtracts the reserved output tokens *and* a safety margin before using
# it, so a small inaccuracy costs an early (cheap) summary rather than a failed
# request. A model that is not listed falls back to ``DEFAULT_CONTEXT_TOKENS``,
# which is deliberately conservative: a quantized local checkpoint with an
# unknown window is far more likely to be 32k than 200k.
MODEL_CONTEXT_WINDOWS = {
    # -- OpenAI -----------------------------------------------------------
    "gpt-4.1": 1_047_576,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o4-mini": 200_000,
    "o3": 200_000,
    "o1": 200_000,
    # -- Anthropic --------------------------------------------------------
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "claude-3-7": 200_000,
    "claude-3-5": 200_000,
    "claude-3": 200_000,
    "claude-4": 200_000,
    "claude-2": 100_000,
    # -- Qwen (the common local workhorse) --------------------------------
    "qwen3": 131_072,
    "qwen2.5": 131_072,
    "qwen2": 32_768,
    "qwen1.5": 32_768,
    # -- Meta Llama -------------------------------------------------------
    "llama-3.3": 131_072,
    "llama-3.2": 131_072,
    "llama-3.1": 131_072,
    "llama-3": 8_192,
    "llama-2": 4_096,
    "codellama": 16_384,
    # -- DeepSeek ---------------------------------------------------------
    "deepseek-v3": 65_536,
    "deepseek-r1": 65_536,
    "deepseek-coder": 65_536,
    "deepseek": 32_768,
    "deepseek-flash": 250_000,
    "deepseek-pro": 250_000,
    "deepseek-v4": 250_000,
    # -- Mistral ----------------------------------------------------------
    "mixtral": 32_768,
    "mistral": 32_768,
    # -- Google -----------------------------------------------------------
    "gemini-2": 1_048_576,
    "gemini-1.5": 1_048_576,
    "gemini": 32_768,
    # -- other local families ---------------------------------------------
    "command-r": 131_072,
    "internlm": 32_768,
    "phi-4": 16_384,
    "phi-3": 131_072,
    "gemma-2": 8_192,
    "gemma": 8_192,
    # -- other local families ---------------------------------------------
    "ornith-ai/Ornith-1.5-397B-FP8" : 250_000,
    "ornith-ai/Ornith-1.5-397B" : 250_000,
    "ornith-ai/Ornith-1.5-35B-A3B-FP8" : 250_000,
    "ornith-ai/Ornith-1.5-35B-A3B" : 250_000,
    "ornith-ai/Ornith-1.5-122B-FP8" : 250_000,
    "ornith-ai/Ornith-1.5-122B" : 250_000,
}

# Used when the model is unknown / unnamed. Conservative on purpose.
DEFAULT_CONTEXT_TOKENS = 125_000

# How much of the window to keep free for the *reply*. A reply that gets
# truncated mid-tool-call is useless, so this is reserved up front: a request is
# "too big" once the input plus this reserve would not fit.
DEFAULT_RESERVED_OUTPUT_TOKENS = 8_192

# Fraction of the window kept as slack on top of the output reserve, covering
# the gap between our estimate and the model's real tokenizer (tool schemas,
# chat-format boilerplate, ...).
CONTEXT_SAFETY_FRACTION = 0.10

# ... but never let that margin eat a huge window: 10% of 1M would be 100k of
# slack nobody needs.
MAX_SAFETY_MARGIN_TOKENS = 20_000

# A budget below this is unusable (it would compact after every single turn),
# so it is the floor the resolver will never go under.
MIN_COMPACTION_TOKEN_BUDGET = 2_048


class CompactionBudget(NamedTuple):
    """The resolved input-token budget for one session.

    ``token_limit`` is the input size at which compaction triggers,
    ``reserved_output_tokens`` is the headroom kept for the reply (so the
    trigger test is ``input + reserved > token_limit``), ``context_window`` is
    the raw window the limit was derived from, and ``source`` records *why* --
    ``"explicit"`` (``--max-context-tokens``), ``"model"`` (looked up from
    :data:`MODEL_CONTEXT_WINDOWS`) or ``"default"`` (the fallback).
    """

    token_limit: int
    reserved_output_tokens: int
    context_window: int
    source: str


# Longest key first, so ``gpt-4o`` is tested before ``gpt-4`` and
# ``qwen2.5`` before ``qwen2``: a substring match on the *most specific* family
# name is what makes ``Qwen/Qwen2.5-Coder-32B-Instruct`` resolve to 131k rather
# than 32k.
_CONTEXT_KEYS_BY_SPECIFICITY = tuple(
    sorted(MODEL_CONTEXT_WINDOWS, key=len, reverse=True)
)


def context_window_for(model, default=None):
    """Approximate context window (in tokens) for ``model``, else ``default``.

    Matching is case-insensitive and tolerant of the decorations real model ids
    carry: an exact id wins (``"gpt-4"``), otherwise the most specific family
    name found anywhere in the id is used, so ``"Qwen/Qwen2.5-Coder-32B-Instruct"``
    and ``"claude-3-5-sonnet-20241022"`` both resolve. ``None``/empty input and
    unknown models return ``default`` (never raises).
    """
    if not model:
        return default
    name = str(model).strip().lower()
    if not name:
        return default
    if name in MODEL_CONTEXT_WINDOWS:
        return MODEL_CONTEXT_WINDOWS[name]
    for key in _CONTEXT_KEYS_BY_SPECIFICITY:
        if key in name:
            return MODEL_CONTEXT_WINDOWS[key]
    return default


def compaction_budget(model=None, explicit_limit=None, max_output_tokens=None):
    """Resolve the compaction budget for ``model``.

    Precedence, highest first:

    1. ``explicit_limit`` (``--max-context-tokens``) is used verbatim -- the
       user knows their deployment better than any table does;
    2. otherwise the model's context window, minus the reserved output tokens
       and a safety margin;
    3. for an unknown model, :data:`DEFAULT_CONTEXT_TOKENS` stands in for the
       window in (2), so an unlisted local checkpoint still gets a sane limit
       instead of an unbounded one.

    The reserve is ``max_output_tokens`` (``--max-tokens``) when the user capped
    replies, else :data:`DEFAULT_RESERVED_OUTPUT_TOKENS`. It is clamped to half
    the window, so a silly ``--max-tokens`` cannot produce a negative budget,
    and the result never drops below :data:`MIN_COMPACTION_TOKEN_BUDGET`
    (nor below the window, if the window itself is tinier than that floor).
    """
    reserve = int(max_output_tokens) if max_output_tokens else 0

    if explicit_limit and int(explicit_limit) > 0:
        limit = int(explicit_limit)
        window = context_window_for(model) or DEFAULT_CONTEXT_TOKENS
        if reserve <= 0:
            reserve = _reserve_for(window)
        return CompactionBudget(limit, reserve, window, "explicit")

    window = context_window_for(model)
    source = "model"
    if window is None:
        window, source = DEFAULT_CONTEXT_TOKENS, "default"

    reserve = _reserve_for(window, reserve)
    margin = min(int(window * CONTEXT_SAFETY_FRACTION), MAX_SAFETY_MARGIN_TOKENS)
    floor = min(MIN_COMPACTION_TOKEN_BUDGET, window)
    limit = max(window - reserve - margin, floor)
    return CompactionBudget(limit, reserve, window, source)


def _reserve_for(window, requested=0):
    """Clamp the output reserve to something sane for ``window``."""
    reserve = int(requested) if requested and int(requested) > 0 else DEFAULT_RESERVED_OUTPUT_TOKENS
    if window <= 0:
        return reserve
    # Never reserve more than half the window: the input needs the other half.
    return max(0, min(reserve, window // 2))


def resolve_compaction_token_limit(model=None, explicit_limit=None, max_output_tokens=None):
    """Just the input-token limit from :func:`compaction_budget`."""
    return compaction_budget(model, explicit_limit, max_output_tokens).token_limit
