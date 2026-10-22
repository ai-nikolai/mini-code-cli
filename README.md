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
- load-last-session: instead of starting a fresh conversation, automatically continue your most recently active session from the default cache dir. It resumes under a brand-new session id, so the original session is never overwritten. An explicit `--restore-session` (a specific path) takes precedence if both are given. The "most recent" session is resolved from a single source: at the end of every interactive session, its path is recorded in a tiny `.mini/last_session.json` pointer inside the `--allowed-dir`, and `--load-last-session` continues whatever that pointer names (there is deliberately no timestamp fallback over the cache dir — with no pointer, a fresh session simply starts). The `.mini` pointer is the same directory the `--skills` feature uses, but it is a single dedicated file, so the two never clash.
- **Interactive session offer:** if a previous session exists in the cache dir but you did **not** explicitly ask to resume one (i.e. neither `--load-last-session` nor `--restore-session` was passed), the prompt asks what you'd like to do before the conversation starts: 1) **Resume** the previous session (splice its history in under a brand-new session id and continue it), 2) **Start a new** session (empty history), or 3) **Exit** (start nothing). Ctrl-C / EOF at the offer is treated as a graceful exit. This is distinct from `--load-last-session`, which silently resumes the most recent session without asking.
- plan: `--plan` enables **plan mode**. The loop keeps two human-readable tracking files in the session directory: `CURRENT_GOAL.md` (the user's latest instruction) and `CURRENT_PLAN.md` (a step-by-step plan the loop generates for the active goal, marked finished once the goal is done). These files are authored by the *loop*, not the agent (they live outside `--allowed-dir`), and you can inspect them anytime with `/plan` (only meaningful in plan mode). After the work looks done, plan mode reviews the plan's completion and re-queues any outstanding tasks before returning to you. Plan mode also changes how history compaction works: after the history is compacted to a summary, the current `CURRENT_PLAN.md` and the user's last instruction (`CURRENT_GOAL.md`) are re-injected into the messages so the agent keeps working against the same goal. When the session ends, the active goal is marked finished in `CURRENT_PLAN.md`.
- max-context-tokens: the input-token budget that triggers **history compaction**. Compaction is decided by **tokens only** — never by the number of messages — because it is the tokens that make a request overflow the model's context window (a few large tool outputs can fill a window that hundreds of short turns would not). The size of the next request is projected from the server-reported `prompt_tokens` plus an estimate for whatever was appended since, plus the output reserve (`--max-tokens`). Unset (default) derives the limit from the model's context window minus the output reserve and a safety margin; lower it to compact sooner, raise it to compact later. There is no message-count cap to configure.

### Available tools

The agent can call a small set of tools, all of which live in `mini_code_cli/tools.py`:

Each tool exposes only the parameters a model actually needs; every optional parameter
has a default, so leaving it out keeps the previous behaviour.

- **`read_file`** — read a file. `offset` (1-based first line) and `limit` (max lines) page
  through large files instead of loading them whole.
- **`write_file`** — write/overwrite a file. `append` adds to the end instead of
  overwriting; `create_dirs` (default true) creates missing parent directories.
- **`edit`** — replace an **exact string** in a file (`file_path`, `old_string`,
  `new_string`, optional `replace_all`). It refuses an edit when `old_string` is missing
  or matches more than once (unless `replace_all=true`), so a wrong edit is never applied
  silently. Prefer this over rewriting a whole file.
- **`grep`** — search a file or a directory with a regular expression. `recursive`,
  `include` (e.g. `"*.py"`), `ignore_case`, `line_numbers` (default true) and
  `max_results`. Noise directories (`.git`, `__pycache__`, `node_modules`, ...) are skipped.
- **`glob`** — list files matching a glob pattern (one per line, `**` supported), optionally
  resolved against a `root` directory.
- **`web_search`** — best-effort web lookup: returns popular related queries (DuckDuckGo autocomplete) plus a short extract from the top Wikipedia match. Never raises; always returns a string.
- **`read_website`** — fetch the raw text/HTML of a URL (bounded by `--tool-timeout`); `max_chars` caps how much of a huge page is returned.
- **`run_shell_command`** — run a shell command (only when `--enable-shell` is set). `cwd`
  sets the working directory and `timeout` the wall clock; stdout, stderr and a non-zero
  exit code are all reported. Every command is passed through the shell-safety classifier
  described above.

### Sub-agents

Passing `--enable-subagents` lets the main agent **delegate a focused task to a sub-agent**, which is exposed to the model as an extra tool. A sub-agent is a self-contained, narrowed agent: it runs on its own in auto-mode (unattended), in its own isolated session folder, and hands a short text result (plus an optional `report.md`) back to the main agent. It reuses all the existing machinery -- tools, the shell-safety classifier, the `--allowed-dir` sandbox, history compaction and `--plan` mode -- and cannot recursively spawn further sub-agents.

Enable it and the available sub-agents appear in the help (`/help`):

```bash
mini-code --enable-subagents
```

Four worked examples ship out of the box:

- **`research`** -- read-only web research. Given a topic it searches the web, reads a few sources and returns a concise, sourced summary. It can only use `web_search` / `read_website` -- never the filesystem or shell -- so it is safe even without a sandbox. Accepts an optional `sources` hint (e.g. `"wikipedia.org only"`).
- **`planner`** -- drafts/refines a clear step-by-step plan for a goal and returns it. Always runs in plan mode.
- **`software_engineer`** -- write, edit, refactor, fix or test code: it can read/write/edit files and run shell commands (tests, builds) to verify the work.
- **`generic`** -- a general-purpose helper for tasks that do not fit a narrower agent.

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
usage: agent_loop.py [-h] [--url URL] [--max-tokens MAX_TOKENS]
                     [--temperature TEMPERATURE] [--model MODEL]
                     [--api-key API_KEY] [--cache-dir CACHE_DIR]
                     [--restore-session RESTORE_SESSION] [-l]
                     [--system-prompt SYSTEM_PROMPT] [-a]
                     [--agent-md AGENT_MD] [--prompt PROMPT]
                     [--ask-permission] [--sudo] [--allowed-dir ALLOWED_DIR]
                     [-s] [-g] [--skills] [--llm-timeout LLM_TIMEOUT]
                     [--tool-timeout TOOL_TIMEOUT]
                     [--max-context-tokens MAX_CONTEXT_TOKENS] [-p] [--all]
                     [--all-last] [--verbose]

Agent loop for interacting with a language model

options:
  -h, --help            show this help message and exit
  --url URL             API host, default=http://0.0.0.0:30000
  --max-tokens MAX_TOKENS
                        Maximum tokens for a single LLM response (the reply
                        budget), default=None. Also used as the output
                        headroom reserved when deriving the compaction limit
                        from the model's context window.
  --temperature TEMPERATURE
                        Temperature for LLM, default=1.0
  --model MODEL         Model name, default=None
  --api-key API_KEY     API key for authentication. If not set, it tries to
                        find it in env variable: MINI_CODE_API_KEY.
  --cache-dir CACHE_DIR
                        The cache dir for the session histories.
                        default=~/.cache/mini/sessions/
  --restore-session RESTORE_SESSION
                        The location of a previous session to continue. (e.g. 
                        ~/.cache/mini/sessions/1792028090823027-
                        a9d77653/)
  -l, --load-last-session
                        Continue the most recent previous session recorded in
                        the --allowed-dir .mini pointer, in place and without
                        the resume prompt, instead of starting fresh.
                        --restore-session takes precedence if also given.
  --system-prompt SYSTEM_PROMPT
                        Replace system prompt, with a custom system prompt.
  -a, --auto-mode       Whether to run the agent in `auto-mode'. Or default:
                        `manual-mode'.
  --agent-md AGENT_MD   If the agent should use an agent-md file... (it will
                        added after system message.)
  --prompt PROMPT       An initial prompt from the user...
  --ask-permission      Ask for permission before any tool call.
  --sudo                DANGEROUS: never ask for permission. Every non-
                        destructive tool call and shell command runs
                        automatically without confirmation. Destructive (DENY-
                        tier) commands are still refused.
  --allowed-dir ALLOWED_DIR
                        Allowed directory for file operations, default='./'
  -s, --enable-shell    Allow shell execution. default= False
  -g, --enable-subagents
                        Allow the agent to delegate work to sub-agents
                        (exposed as extra tools). default= False
  --skills              Enable the skills feature: a local .mini directory
                        (inside --allowed-dir) of skill.md files, skill sub-
                        dirs, a high-level SKILLS.md summary, and executable
                        Python scripts, exposed to the agent as
                        list_skills/read_skill/create_skill/run_skill tools.
                        default= False
  --llm-timeout LLM_TIMEOUT
                        HTTP timeout in seconds for LLM requests (reasoning
                        models can be slow). default=500
  --tool-timeout TOOL_TIMEOUT
                        Timeout in seconds for a single tool call. default=120
  --max-context-tokens MAX_CONTEXT_TOKENS
                        Input-token budget before the conversation is
                        compacted into a summary. This is the primary
                        compaction trigger: it is compared against the
                        projected size of the next request (the server-
                        reported prompt tokens, plus an estimate for whatever
                        was appended since) plus the output reserve (--max-
                        tokens, or a default). Unset (default): derived from
                        the model's context window minus the output reserve
                        and a safety margin, falling back to a conservative
                        window for unknown models. Lower it to compact sooner.
  -p, --plan            Enable plan mode: the agent keeps a CURRENT_PLAN.md
                        (its step-by-step plan, marked finished when the goal
                        is done) and CURRENT_GOAL.md (the user's latest
                        instruction) in the session directory. Also changes
                        how history compaction works.
  --all                 Enable: --plan, --subagents, --shell, --automode,
                        --sudo
  --all-last            Enable: --plan, --subagents, --shell, --automode,
                        --sudo, --load-last-session
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
/plan               show the current goal + plan (plan mode only)
/help               show this help
/quit, /exit        save the session and exit
```

## Testing

The project ships with a small standard-library test suite (no extra dependencies to install):

```bash
# Run the whole suite from the repo root:
python3 -m unittest discover -s tests -p "test_*.py"

# ...or run a single test file:
python3 tests/test_tokens.py
python3 tests/test_e2e_agent_loop.py
python3 tests/test_interactive.py
python3 tests/test_subagents.py
```

The tests focus on the parts of the agent that are easy to get wrong and worth guarding:

The suite is hermetic: every test builds its own temporary `--allowed-dir` and cache dir, so
nothing is written into your real `~/.cache/mini/sessions/` and — importantly — the session
pointers the tests create go into a `.mini_tests` directory inside the test's temporary allowed
dir, never into a real `.mini` (that one is yours: your skills folder and `last_session.json`).

- **`test_e2e_agent_loop.py`** — the whole loop, driven the way a user drives it
  (`AgentSession(args).run(...)`), with only the LLM stubbed: real tools run on disk, the
  `--allowed-dir` sandbox is exercised end to end, plan mode's tracking files are written,
  sub-agent delegation runs a nested loop, history compaction rolls the segment over
  (keeping per-segment usage), sessions restore in place, and the `edit` tool / richer
  tool schemas are covered.
- **`test_tokens.py`** — the compaction *policy*: the model → context-window table, the
  budget resolution (explicit / model / conservative fallback), the token estimator, and
  the trigger itself — which fires on projected tokens (+ the reply reserve) and **never**
  on the message count.
- **`test_interactive.py`** — the shell-safety classifier (DENY / ASK / ALLOW tiers) and
  compaction edge cases (an empty summary is retried, then abandoned without losing the
  transcript; a failing summariser call never corrupts the segment).
- **`test_subagents.py`** — sub-agent delegation, isolation of the child session, the
  `report.md` channel and the intersection of child permissions with the parent's.

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

## Citation

If you use `mini-code-cli` in your research or writing, please cite it using the BibTeX entry below:

```bibtex
@misc{mini-code-cli,
  author       = {Rozanov, Nikolai},
  title        = {mini-code-cli: A minimal, dependency-light coding agent},
  year         = {2026},
  version      = {0.1.0.0},
  howpublished = {\url{https://github.com/ai-nikolai/mini-code-cli}},
  note         = {A no-bloat, dependency-free agent CLI for experimenting.}
}
```

## (C) Nikolai Rozanov, 2026 - Present