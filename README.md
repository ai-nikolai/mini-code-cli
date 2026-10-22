# mini-code-cli

### A minimal coding agent CLI

> No bloat, no dependencies, no tracking, no telemetry...

[![CI](https://img.shields.io/github/actions/workflow/status/ai-nikolai/mini-code-cli/ci.yml?branch=main&label=CI&logo=github&style=flat-square)](https://github.com/ai-nikolai/mini-code-cli/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/mini-code-cli?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/mini-code-cli/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/mini-code-cli/)
[![Stars](https://img.shields.io/github/stars/ai-nikolai/mini-code-cli?style=flat-square&logo=github)](https://github.com/ai-nikolai/mini-code-cli/stargazers)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue?style=flat-square)](LICENSE)
[![Cite](https://img.shields.io/badge/cite-BibTeX-blueviolet?style=flat-square&logo=latex&logoColor=white)](#citation)

**Purpose:** a no-bloat, minimal coding CLI to enjoy experimenting and researching!

**Install** · [Quick start](#quick-start) · [Modes &amp; safety](#modes--safety) · [Options](#command-line-options) · [Tools](#available-tools) · [Sub-agents](#sub-agents) · [In-session commands](#in-session-commands) · [Development](#development) · [Docs](#documentation) · [Citation](#citation)

> [!TIP]
> **New here?** `pip install mini-code-cli`, point it at a model (a local server on
> `localhost:30000`, or `MINI_CODE_API_KEY` + `--url`), and run `mini-code`. Everything else
> is optional.

## Install

```bash
pip3 install mini-code-cli
```

- **PyPI:** <https://pypi.org/project/mini-code-cli/>
- **Python:** `>= 3.10` &nbsp;·&nbsp; **runtime dependency:** `requests` only
- **From source / contributing:** see [docs/README_DEV.md](docs/README_DEV.md)

## Quick start

**1 — Local model** (e.g. vLLM or SGLang on port 30000; see [serving a model locally](#serving-a-model-locally)):

```bash
mini-code                          # talks to http://0.0.0.0:30000/v1
```

**2 — Hosted / custom API:**

```bash
export MINI_CODE_API_KEY="sk-YOUR_API_KEY"
mini-code --url "https://api.deepseek.com" --model "deepseek-v4-flash"
```

**3 — The shortcuts** (plan mode, sub-agents, shell, auto-mode, no permission prompts):

```bash
mini-code --all                    # = --plan --enable-subagents --enable-shell --auto-mode --sudo
mini-code --all-last               # ...and continue your last session
```

**4 — A first prompt:**

```txt
Write an agent.md file for me that helps an LLM write efficient triton kernels,
given a reference implementation.
```

> [!NOTE]
> **Every flag** is listed once, in [Command-line options](#command-line-options) — or run
> `mini-code --help`. Modes and their caveats are described in [Modes & safety](#modes--safety).

## Command-line options

| Option | Default | What it does |
| --- | --- | --- |
| `--url URL` | `http://0.0.0.0:30000` | API host |
| `--model MODEL` | — | model name |
| `--api-key KEY` | `$MINI_CODE_API_KEY` | API key for authentication |
| `--max-tokens N` | — | reply budget; also the output headroom used when deriving the compaction limit |
| `--temperature T` | `1.0` | sampling temperature |
| `--system-prompt TEXT` | built-in | replace the system prompt (an `--agent-md` file is appended *after* it) |
| `--prompt TEXT` | — | initial user prompt |
| `-a`, `--auto-mode` | off | keep querying the model and running tools until no tool calls remain |
| `--agent-md FILE` | — | append an `agent.md` file after the system prompt (e.g. `./skill.md`) |
| `--ask-permission` | off | ask before every tool call (ALLOW-tier shell commands still skip the prompt) |
| `--sudo` | off | **dangerous:** never ask. DENY-tier commands are still refused |
| `--allowed-dir DIR` | `./` | sandbox for **file** tools (does not restrict the shell) |
| `-s`, `--enable-shell` | off | allow shell execution (classifier always applies) |
| `-g`, `--enable-subagents` | off | expose sub-agents as tools |
| `--skills` | off | enable the `.mini` skills directory (`skill.md`s, `SKILLS.md`, runnable scripts) |
| `-p`, `--plan` | off | plan mode (`CURRENT_GOAL.md` / `CURRENT_PLAN.md`, `/plan`) |
| `--llm-timeout S` | `500` | HTTP timeout for LLM requests |
| `--tool-timeout S` | `120` | timeout for a single tool call |
| `--max-context-tokens N` | derived | input-token budget before the history is compacted |
| `--cache-dir DIR` | `~/.cache/mini/sessions/` | where session histories are stored |
| `--restore-session PATH` | — | continue a specific previous session |
| `-l`, `--load-last-session` | off | continue the session recorded in the `.mini` pointer |
| `--all` | off | `--plan --enable-subagents --enable-shell --auto-mode --sudo` |
| `--all-last` | off | `--all` + `--load-last-session` |
| `--verbose` | off | print full detail (default: user/assistant turns, token usage, tool calls only) |
| `-h`, `--help` | — | show the built-in help |

## Available tools

The agent can call a small set of tools, all defined in `mini_code_cli/tools.py`. Every
parameter a model does not need is defaulted, so omitting it keeps the previous behaviour.

| Tool | What it does |
| --- | --- |
| `read_file` | read a file; `offset` (1-based line) and `limit` page through large files |
| `write_file` | write/overwrite; `append` adds to the end, `create_dirs` creates parents |
| `edit` | replace an **exact** string (`old_string` → `new_string`); refuses a missing or ambiguous match unless `replace_all` |
| `grep` | regex search a file or directory; `recursive`, `include`, `ignore_case`, `line_numbers`, `max_results` (noise dirs skipped) |
| `glob` | list files matching a glob (one per line, `**` supported), optionally against a `root` |
| `web_search` | best-effort lookup: related queries + a short Wikipedia extract. Never raises |
| `read_website` | fetch raw text/HTML of a URL (bounded by `--tool-timeout`), `max_chars` caps size |
| `run_shell_command` | run a shell command (needs `--enable-shell`); `cwd`, `timeout`, and stdout/stderr/exit code are all reported |

## Sub-agents

`--enable-subagents` lets the main agent **delegate a focused task** to a sub-agent, exposed to
the model as an extra tool. A sub-agent is a self-contained, narrowed agent: it runs unattended
in auto-mode, in its own isolated session folder, and hands back a short text result (plus an
optional `report.md`). It reuses all the existing machinery — tools, the shell-safety
classifier, the `--allowed-dir` sandbox, compaction and `--plan` — and cannot spawn further
sub-agents.

```bash
mini-code --enable-subagents       # the available sub-agents then show up in /help
```

| Sub-agent | What it is for |
| --- | --- |
| `research` | read-only web research: searches, reads a few sources, returns a sourced summary. Can only use `web_search` / `read_website` — never the filesystem or shell. Optional `sources` hint (e.g. `"wikipedia.org only"`) |
| `planner` | drafts or refines a step-by-step plan for a goal; always runs in plan mode |
| `software_engineer` | writes, edits, refactors, fixes and tests code; can run shell commands (tests, builds) to verify |
| `generic` | general-purpose helper for anything that fits no narrower agent |

> [!NOTE]
> A sub-agent's `tools`, `enable_shell` and `max_steps` are always **intersected with the
> parent's** permissions, so it can never do anything the main agent could not. Sub-agents
> compose with `--plan`: each keeps its own `CURRENT_PLAN.md` / `CURRENT_GOAL.md`.

<details>
<summary><b>Adding your own sub-agent</b></summary>

A sub-agent is just a small class — subclass `SubAgent` and register it:

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

Make sure the module is imported (e.g. `import my_subagents`) and it is discovered and exposed
to the model automatically when `--enable-subagents` is set. See
`mini_code_cli/subagents/builtin.py` for every knob and
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (sections 2–3) for the design.

</details>

## In-session commands

| Command | What it does |
| --- | --- |
| `/history` | list the conversation (turn number, what happened, where) |
| `/history <n>` | show the full messages recorded for turn `n` |
| `/history last` | ...for the most recent turn |
| `/reset [<n>]` | rewind to the end of turn `n` (`last` works too); drops everything after it |
| `/clear` | clear the screen (keeps the conversation history) |
| `/plan` | show the current goal + plan (plan mode only) |
| `/help` | show this help |
| `/quit`, `/exit` | save the session and exit |

> [!NOTE]
> `/history` and `/reset` are driven by `summary_history.json`, a flat recorder that maps every
> turn to the `history_{n}.json` segment and message index it lives in — so no full history is
> read just to list the conversation.
>
> * **`<n>` is validated**: `last`, or a positive 1-based turn number. `0`, negatives and
>   non-numbers are rejected with `No such turn` instead of silently showing a different turn.
> * **A bare `/reset` never destroys anything**: it reports `Already at turn N` when the
>   transcript already ends at the last recorded turn, and *warns instead of truncating* if
>   there are messages that were never recorded. Only `/reset <n>` / `/reset last` rewind.
> * **A rewind really sticks**: the target `history_{n}.json` is truncated in place (no backup)
>   and every higher `history_{m}.json` / `usage_{m}.json` segment is deleted, so a restart
>   cannot resurrect the dropped turns. All history files are written atomically.
> * **Token usage is never rolled back**: a rewind does not un-spend tokens, and the session
>   total is preserved. See
>   [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#rewinding-reset-and-what-it-does-to-the-files)
>   for the files involved.

## Development

Linting, the test suite, CI and the release process are documented in
[**docs/README_DEV.md**](docs/README_DEV.md).

```bash
python3 scripts/lint.py                                # lint (standard library only)
python3 -m unittest discover -s tests -p "test_*.py"    # run the test suite
```

Releases are cut by tagging `main`: pushing a `v*` tag triggers
[`.github/workflows/release.yml`](.github/workflows/release.yml), which builds the sdist + wheel,
verifies them with `twine check`, and attaches them to a new GitHub release.

## Documentation

Everything beyond this README lives in [`docs/`](docs/):

| Document | What it covers |
| --- | --- |
| [**docs/README_DEV.md**](docs/README_DEV.md) | developer guide — local setup, repository layout, linting, the test suite, CI and the release / PyPI upload process |
| [**docs/ARCHITECTURE.md**](docs/ARCHITECTURE.md) | design — module layout, on-disk conversation state (`history_{n}.json`, `summary_history.json`, usage files), the sub-agent framework and how to extend it |

This README stays the user-facing entry point (install, CLI flags, tools, sub-agents,
in-session commands, modes &amp; safety); the two documents above go deeper on the developer
and design side.

## Modes & safety

### Shell execution — `-s` / `--enable-shell`

Everything the agent runs from the shell first passes a **safety classifier** that sorts the
command into one of three tiers:

| Tier | Examples | What happens |
| --- | --- | --- |
| **DENY** | `rm -rf /`, `dd of=/dev/...`, `mkfs.*`, fork bombs | refused outright, never run |
| **ASK** | `rm file`, `sudo ...`, `chmod`, `curl ... \| sh` | requires your confirmation |
| **ALLOW** | `ls`, `cat`, `git status` | runs without prompting |

> [!WARNING]
> The classifier is **always enforced** — even in auto-mode and with `--sudo` — so the agent
> cannot run destructive commands unattended. It is a heuristic, not a sandbox: use it with care.
> `--allowed-dir` sandboxes **file tools**, not the shell.

### Plan mode — `-p` / `--plan`

The loop keeps two human-readable tracking files in the session directory: `CURRENT_GOAL.md`
(your latest instruction) and `CURRENT_PLAN.md` (a step-by-step plan for that goal, marked
finished when it is done). They are authored by the *loop*, not the agent, live outside
`--allowed-dir`, and can be inspected any time with `/plan`.

> [!NOTE]
> Plan mode also changes **history compaction**: after the transcript is compacted to a summary,
> `CURRENT_PLAN.md` and `CURRENT_GOAL.md` are re-injected, so the agent keeps working against the
> same goal. When the work looks done the plan is reviewed and any outstanding task is re-queued
> before the agent returns to you.

### Sessions — resume, or start fresh

Every session is saved under `--cache-dir` (default `~/.cache/mini/sessions/`).

> [!TIP]
> If a previous session exists but you did **not** ask to resume one (neither
> `--load-last-session` nor `--restore-session`), the prompt offers **Resume / Start new / Exit**
> before the conversation begins. Ctrl-C or EOF there is a graceful exit. This is the *asking*
> variant; `--load-last-session` is the *silent* one.


| Flag | Behaviour |
| --- | --- |
| `--restore-session PATH` | continue that exact session (a session directory or a bare session id under `--cache-dir`) |
| `-l`, `--load-last-session` | continue the session recorded in the `.mini/last_session.json` pointer, without asking |

`--load-last-session` continues that session **in place** (same id, same folder, same
messages), so the conversation you were in is the one you get back; an explicit
`--restore-session` wins if both are given. The "most recent" session comes from a single
source — the tiny `.mini/last_session.json` pointer written when a session starts or is
restored inside `--allowed-dir` (deliberately no timestamp fallback: with no pointer, a fresh
session starts). That pointer lives in the same `.mini` directory as `--skills`, but as its
own dedicated file, so the two never clash.

A session folder is restorable only if it has what this version writes: a valid
`summary_history.json` recorder, plus the `history_{n}.json` segment its last record names
(usage files are optional). Anything else — a folder from an older release, a hand-assembled
transcript, a recorder that does not validate, a missing segment — is **refused with a note
saying why**, and a fresh session is started instead. Nothing is repaired, guessed at, or
written into the refused folder.

### Context compaction — `--max-context-tokens`

Compaction is decided by **tokens only**, never by message count: a few large tool outputs can
fill a window that hundreds of short turns would not. The next request is projected from the
server-reported `prompt_tokens`, plus an estimate of what was appended since, plus the output
reserve (`--max-tokens`). Unset (default) it is derived from the model's context window minus the
output reserve and a safety margin. Lower it to compact sooner, raise it to compact later — there
is no message-count cap to configure.

Compaction replaces the older turns with a summary, but it always carries the **turn in flight**
across the rollover: the message you just typed (or, mid-turn, the tool call the model is
waiting on) is kept verbatim after the summary. So a compaction that fires just as you send an
instruction cannot swallow it — the model still answers what you asked, and the turn stays in
the session file, in `/history` and in a later restore.

Because a rollover leaves no trace in the transcript (the old turns really are gone), the
console says in plain terms what happened — all of it without `--verbose`:

```
[!] Compacting the conversation: 42 messages -> a summary (51000 tokens in, 8192 reserved for the reply > 52000 budget). The current turn (1 message) is kept verbatim.
[!] Summary preview: The user asked for a token-based compaction trigger, then fixed ...
[!] Compacted history_0.json (42 messages) into a summary; history_1.json now starts with 3 messages (summary + 1 kept message).
```

So you can see *what* was folded away (a preview of the summary itself), *how much* (the
message count of the segment that was closed), *where it went* (`history_0.json` closed,
`history_1.json` started) and that nothing you just typed was lost. If the summariser answers
with an empty summary twice, or the call fails, compaction is abandoned and the note says so
explicitly — "nothing was lost, but the next request may still be too large" — with the
transcript and the segment left untouched.

### Exiting — the good-bye box

Quitting (`/quit`, `exit`, or EOF on stdin) saves the session and then prints a short yellow
box: the version, a sign-off, and what the session left behind — session id, folder, the
history segments on disk, how many turns were recorded, and the token / call totals.

```
╔══════════ MiniCodeCLI v0.1.1.2 - goodbye ═════════════════════╗
║ Goodbye, and thanks for using MiniCodeCLI!                    ║
║                                                               ║
║ session id   : 1792666273983619-36ca3c2e                      ║
║ saved under  : ~/.cache/mini/sessions/1792…-36ca3c2e/         ║
║ history      : history_0.json (segment 0), 5 messages         ║
║ conversation : 2 recorded turns                               ║
║ tokens       : 20 in / 10 out (30 total)                      ║
║ calls        : 2 LLM, 0 tool                                  ║
╚═══════════════════════════════════════════════════════════════╝
```

It is printed exactly once, and only on a normal exit: a Ctrl-C or an internal error already
says what went wrong in red, so it does not get a cheerful summary box on top. With `--verbose`
the old per-field `Token usage (this session)` block is printed above it as well.


## Benchmarks (academic evaluation)

`benchmarks/` runs coding agents against **Terminal-Bench 2.x** and **SWE-bench Verified** and
compares them — mini-code with several settings, Claude Code, Pi and a deterministic `mock`
agent — by wrapping each benchmark's own official harness (`harbor`, `tb`,
`swebench.harness.run_evaluation`). It supports Docker and Apptainer, exports a CSV, and ships an
analyser for the results.

Everything is documented in **[benchmarks/README.md](benchmarks/README.md)**: prerequisites, the
exact commands (`plan`, `run --csv`, `analyze`), the Docker and Apptainer variants, the fixed-model
smoke config (`benchmarks/configs/ornith_30b.json`), and what a run produces.

## Serving a model locally

Tested with vLLM v0.30 (tool calling enabled):

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

If you use `mini-code-cli` in your research or writing, please cite it:

```bibtex
@misc{mini-code-cli,
  author       = {Rozanov, Nikolai},
  title        = {mini-code-cli: A minimal, dependency-light coding agent},
  year         = {2026},
  version      = {0.1.1.2},
  howpublished = {\url{https://github.com/ai-nikolai/mini-code-cli}},
  note         = {A no-bloat, dependency-light agent CLI for experimenting.}
}
```

## License

[AGPL-3.0](LICENSE) — (C) Nikolai Rozanov, 2026 – present.
