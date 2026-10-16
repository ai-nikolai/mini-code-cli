"""mini_code_cli.config — process-wide default settings.

Kept in one small module so every other module (the agent loop, persistence,
the tool runner, sub-agents, ...) can share the same defaults without importing
each other and creating import cycles.
"""
import os

# Where session histories live on disk by default.
DEFAULT_CACHE_LOCATION = os.path.expanduser("~/.cache/mini_code/sessions/")

# The LLM server the CLI talks to by default (a local vLLM/SGLang endpoint).
DEFAULT_MODEL_URL = "http://0.0.0.0:30000"

# Sampling + operational defaults.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOOL_TIMEOUT = 120     # seconds before a single tool call is aborted
DEFAULT_HTTP_TIMEOUT = 500     # seconds before an LLM HTTP request is aborted

# History compaction is triggered by total accumulated prompt + completion
# tokens (not a message count), so it reacts to how full the context window is
# rather than how long the transcript has grown.
DEFAULT_MAX_TOTAL_TOKENS = 125_000

# Kept for backwards compatibility: the old message-count threshold.
DEFAULT_MAX_MESSAGES = 150     # deprecated, retained for external callers/tests
