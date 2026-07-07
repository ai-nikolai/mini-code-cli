# A minimal coding agent cli

> No bloat, no dependencies, no tracking, no telemetry...

Purpose: A no bloat, minimal coding cli, to enjoy experimenting and researching...

## Getting started:

### Installation (from pypi)
```bash
pip3 install mini-code-cli
```

### Run agent:

1. Basic usage: (get local LLM running (e.g. via SGLang or Vllm))
```bash
mini-code #run the agent using `localhost:30000/v1` model
```
2. Custom API based model:
```bash
export MINI_CODE_API_KEY="sk-YOUR_API_KEY"
mini-code --url "https://api.deepseek.com" --model "deepseek-v4-flash"
```

Auto-mode, Agent.md, Shell Enabled mode and a specific allowed dir:
```bash
mini-code --enable-shell --auto-mode --agent-md "./skill.md" --allowed-dir "./" --ask-permission --system-prompt "Your awesome new system prompt..."
```
- auto-mode: the agent will continue querying the LLM and executing tools, until no more tool calls are possible
- agent.md: this will read an agent.md file with the specified name (e.g. ./skill.md)
- shell-enabled: **Be careful**: This will allow the agent to execute shell commands. Specifically, there is no permission checks.
- allowed-dir: this specifies which dirs the agent can read and write from, that does not stop the shell comands!
- ask-permission: asks user permission before every tool call.
- system-prompt: you can override the default prompt by using a string. (Not Agent.md is always appended after the system prompt.)

3. Try a prompt:
```txt
Write an agent.md file for me that helps an LLM write efficient triton kernels, given a reference implementation.
```

4. Get help:
```bash
mini-code --help
```
```
Agent loop for interacting with a language model

options:
  -h, --help            show this help message and exit
  --url URL             API host
  --max-tokens MAX_TOKENS
                        Maximum tokens for LLM response
  --temperature TEMPERATURE
                        Temperature for LLM
  --model MODEL         Model name
  --api-key API_KEY     API key for authentication. If not set, it tries to find it in env variable: MINI_CODE_API_KEY.
  --cache-dir CACHE_DIR
                        The cache dir for the session histories.
  --system-prompt SYSTEM_PROMPT
                        Replace system prompt, with a custom system prompt.
  --auto-mode           Whether to run the agent in `auto-mode'. Or default: `manual-mode'.
  --agent-md AGENT_MD   If the agent should use an agent-md file... (it will added after system message.)
  --prompt PROMPT       An initial prompt from the user...
  --ask-permission      Ask for permission before any tool call.
  --allowed-dir ALLOWED_DIR
                        Allowed directory for file operations
  --enable-shell        Allow shell execution. Default: False
  ```


## (C) Nikolai Rozanov, 2026 - Present