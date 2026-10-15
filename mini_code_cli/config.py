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
DEFAULT_MAX_MESSAGES = 150     # messages in history before it is compacted
