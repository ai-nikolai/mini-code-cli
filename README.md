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
3. Auto-mode and shell enabled: (this will run localhost:30000)
```bash
mini-code --enable-shell --auto-mode
```

4. Advanced usage:
Auto-mode, Agent.md, Shell Enabled mode and a specific allowed dir:
```bash
mini-code --enable-shell --auto-mode --agent-md "./skill.md" --allowed-dir "./" --ask-permission --system-prompt "Your awesome new system prompt..."
```
- auto-mode: the agent will continue querying the LLM and executing tools, until no more tool calls are possible
- agent.md: this will read an agent.md file with the specified name (e.g. ./skill.md)
- shell-enabled: **Be careful**: This will allow the agent to execute shell commands. **Every command is first checked by a shell-safety classifier** that sorts it into three tiers:
    - **DENY** — obviously destructive commands (e.g. `rm -rf /`, `dd of=/dev/...`, `mkfs.*`, fork bombs) are **refused outright** and never run.
    - **ASK** — risky commands (e.g. `rm file`, `sudo ...`, `chmod`, `curl ... | sh`) require your confirmation before running.
    - **ALLOW** — clearly safe commands (e.g. `ls`, `cat`, `git status`) run without prompting.
  This protection is **always enforced** for the shell tool, even in auto-mode / without `--ask-permission`, so the agent can never run dangerous commands unattended. (The classifier is a heuristic, not a sandbox — use with care.)
- allowed-dir: this specifies which dirs the agent can read and write from, that does not stop the shell comands! (though dangerous shell commands are still blocked by the classifier above).
- ask-permission: asks user permission before every tool call. For shell commands, commands in the ALLOW tier skip the prompt, DENY is blocked, and ASK always prompts.
- system-prompt: you can override the default prompt by using a string. (Note: Agent.md is always appended after the system prompt.)

### Available tools

The agent can call a small set of tools, all of which live in `mini_code_cli/tools.py`:

- **`read_file` / `write_file`** — read a file or write/overwrite one.
- **`grep`** — search a file for a pattern.
- **`glob`** — list files matching a glob pattern.
- **`web_search`** — best-effort web lookup: returns popular related queries (DuckDuckGo autocomplete) plus a short extract from the top Wikipedia match. Never raises; always returns a string.
- **`read_website`** — fetch the raw text/HTML of a URL (bounded by `--tool-timeout`).
- **`run_shell_command`** — run a shell command (only when `--enable-shell` is set). Every command is passed through the shell-safety classifier described above.

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
  --url URL             API host, default=http://0.0.0.0:30000
  --max-tokens MAX_TOKENS
                        Maximum tokens for LLM response, default=None
  --temperature TEMPERATURE
                        Temperature for LLM, default=1.0
  --model MODEL         Model name, default=None
  --api-key API_KEY     API key for authentication. If not set, it tries to find it in env variable: MINI_CODE_API_KEY.
  --cache-dir CACHE_DIR
                        The cache dir for the session histories.
  --restore-session RESTORE_SESSION
                        The location of a previous session to continue.
                        (e.g. ~/.cache/mini_code/sessions/1792028090823027-a9d77653/)
  --system-prompt SYSTEM_PROMPT
                        Replace system prompt, with a custom system prompt.
  -a, --auto-mode       Whether to run the agent in `auto-mode'. Or default: `manual-mode'.
  --agent-md AGENT_MD   If the agent should use an agent-md file... (it will added after system message.)
  --prompt PROMPT       An initial prompt from the user...
  --ask-permission      Ask for permission before any tool call. (Shell ALLOW-tier commands skip the prompt; DENY is blocked.)
  --allowed-dir ALLOWED_DIR
                        Allowed directory for file operations
  -s, --enable-shell    Allow shell execution. Default: False
  --llm-timeout LLM_TIMEOUT
                        HTTP timeout in seconds for LLM requests (reasoning models can be slow). default=300
  --tool-timeout TOOL_TIMEOUT
                        Timeout in seconds for a single tool call. default=120
  ```

## Testing

The project ships with a small standard-library test suite (no extra dependencies to install):

```bash
# Run the whole suite from the repo root:
python3 -m unittest discover -s tests -p "test_*.py"

# ...or run a single test file:
python3 tests/test_tools.py
python3 tests/test_sandboxing.py
python3 tests/test_agent_loop.py
python3 tests/test_shell_safety.py
```

The tests focus on the parts of the agent that are easy to get wrong and worth guarding:

- **`test_sandboxing.py`** — the `--allowed-dir` sandbox. Verifies that child paths are allowed, while `..` traversal and sibling paths outside the allowed directory are rejected.
- **`test_tools.py`** — the tool functions. Confirms they *return* their results/errors (instead of printing), that `run_shell_command` captures stderr and honors a timeout, and that the generated tool schema exposes the expected parameters.
- **`test_agent_loop.py`** — agent-loop behaviour. Confirms that enabling the shell forces per-command permission prompts, that memory compaction is correctly wired (accepts token accounting), and that token-usage formatting handles servers that report no usage.
- **`test_shell_safety.py`** — the shell command classifier. Confirms commands are sorted into the three tiers correctly: destructive commands (`rm -rf /`, `dd of=/dev/...`, `mkfs.*`, fork bombs) are **DENY**, risky ones (`rm file`, `sudo`, `curl | sh`, redirects to `/etc/...`) are **ASK**, and safe ones (`ls`, `cat`, `git status`) are **ALLOW**.

This keeps the tool honest as it grows, and makes it safe to experiment with riskier features (like `--enable-shell`) without regressions.

## Other useful commands:

### Serving a vllm model locally (with correct tools). vllm v0.30 (tested):
```bash
vllm serve ornith-ai/Ornith-1.5-397B-FP8 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

## (C) Nikolai Rozanov, 2026 - Present