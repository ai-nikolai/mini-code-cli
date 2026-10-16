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
mini-code -s -a --sudo
mini-code -s -a -p -g -l --sudo #plan, previous session, subagents (#or short-hand the next commands)
mini-code --all
mini-code --all-last
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
- load-last-session: instead of starting a fresh conversation, automatically continue your most recently active session (the one whose history was last modified) from the default cache dir. It resumes under a brand-new session id, so the original session is never overwritten. An explicit `--restore-session` (a specific path) takes precedence if both are given.
- plan: `--plan` enables **plan mode**. The agent now keeps two tracking files in the session directory: `CURRENT_GOAL.md` (the user's latest instruction) and `CURRENT_PLAN.md` (its own step-by-step plan, which it must update as it works and mark finished once the goal is done). The system message gets an extra instruction telling the agent to do this. Plan mode also changes how history compaction works: after the history is compacted to a summary, the current `CURRENT_PLAN.md` and the user's last instruction (`CURRENT_GOAL.md`) are re-injected into the messages so the agent keeps working against the same goal. When the session ends, the active goal is marked finished in `CURRENT_PLAN.md`.

### Available tools

The agent can call a small set of tools, all of which live in `mini_code_cli/tools.py`:

- **`read_file` / `write_file`** — read a file or write/overwrite one.
- **`grep`** — search a file for a pattern.
- **`glob`** — list files matching a glob pattern.
- **`web_search`** — best-effort web lookup: returns popular related queries (DuckDuckGo autocomplete) plus a short extract from the top Wikipedia match. Never raises; always returns a string.
- **`read_website`** — fetch the raw text/HTML of a URL (bounded by `--tool-timeout`).
- **`run_shell_command`** — run a shell command (only when `--enable-shell` is set). Every command is passed through the shell-safety classifier described above.

### Sub-agents

Passing `--enable-subagents` lets the main agent **delegate a focused task to a sub-agent**, which is exposed to the model as an extra tool. A sub-agent is a self-contained, narrowed agent: it runs on its own in auto-mode (unattended), in its own isolated session folder, and hands a short text result (plus an optional `report.md`) back to the main agent. It reuses all the existing machinery -- tools, the shell-safety classifier, the `--allowed-dir` sandbox, history compaction and `--plan` mode -- and cannot recursively spawn further sub-agents.

Enable it and the available sub-agents appear in the help (`/help`):

```bash
mini-code --enable-subagents
```

Two worked examples ship out of the box:

- **`research`** -- read-only web research. Given a topic it searches the web, reads a few sources and returns a concise, sourced summary. It can only use `web_search` / `read_website` -- never the filesystem or shell -- so it is safe even without a sandbox. Accepts an optional `sources` hint (e.g. `"wikipedia.org only"`).
- **`planner`** -- drafts/refines a clear step-by-step plan for a goal and returns it. Always runs in plan mode.

The `tools`, `enable_shell` and `max_steps` knobs are always intersected with the parent's own permissions, so a sub-agent can never do anything the main agent could not. Sub-agents also compose with `--plan` mode: each sub-agent keeps its own `CURRENT_PLAN.md` / `CURRENT_GOAL.md` in its own session folder.

#### Adding your own sub-agent

A sub-agent is just a small class. Subclass `SubAgent` and register it:

```python
# my_subagents.py
from mini_code_cli.subagents import SubAgent, register

class SummarizeSubAgent(SubAgent):
    name = "summarize"
    description = "Summarize a piece of text the main agent provides."
    tools = ["read_file", "write_file"]   # restrict the nested agent's tools
    enable_shell = False
    max_steps = 5

register(SummarizeSubAgent())
```

Then make sure it is imported (e.g. add `import my_subagents`), and it will be discovered and exposed to the model automatically when `--enable-subagents` is set. See `mini_code_cli/subagents/builtin.py` for the full list of knobs and `ARCHITECTURE.md` (sections 2-3) for the design and every available option.

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
                        The cache dir for the session histories. default=~/.cache/mini_code/sessions/
  --restore-session RESTORE_SESSION
                        The location of a previous session to continue. (e.g. ~/.cache/mini_code/sessions/1792028090823027-a9d77653/)
  --load-last-session   Load the most recent previous session from the cache dir and continue it (under a new session id), instead of starting fresh. --restore-session takes precedence if also given.
  --system-prompt SYSTEM_PROMPT
                        Replace system prompt, with a custom system prompt.
  -a, --auto-mode       Whether to run the agent in `auto-mode'. Or default: `manual-mode'.
  --agent-md AGENT_MD   If the agent should use an agent-md file... (it will added after system message.)
  --prompt PROMPT       An initial prompt from the user...
  --ask-permission      Ask for permission before any tool call. (Shell ALLOW-tier commands skip the prompt; DENY is blocked.)
  --sudo                DANGEROUS: never ask for permission. Every non-destructive tool call and shell command runs automatically without confirmation. Destructive (DENY-tier) commands are still refused.
  --allowed-dir ALLOWED_DIR
                        Allowed directory for file operations, default='./'
  -s, --enable-shell    Allow shell execution. default= False
  --enable-subagents    Allow the agent to delegate work to sub-agents (exposed as extra tools). default= False
  --llm-timeout LLM_TIMEOUT
                        HTTP timeout in seconds for LLM requests (reasoning models can be slow). default=500
  --tool-timeout TOOL_TIMEOUT
                        Timeout in seconds for a single tool call. default=120
  --plan                Enable plan mode: the agent keeps a CURRENT_PLAN.md
                        (its step-by-step plan, marked finished when the goal
                        is done) and CURRENT_GOAL.md (the user's latest
                        instruction) in the session directory. Also changes
                        how history compaction works.
  --verbose             Print the full, detailed output. Without it (default),
                        only user/assistant turns, token usage and tool calls
                        are shown.
  ```

### In-session commands

```
/history            list the conversation (turn number, what happened, where)
/history <n>        show the full messages recorded for turn n
/reset [<n>]        rewind to the end of turn n (drops everything after it)
/clear              clear the screen (keeps the conversation history)
/help               show this help
/quit, /exit        save the session and exit
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