# Architecture

A small, dependency-free (stdlib-only) CLI that drives an interactive loop over an
OpenAI-compatible LLM server (vLLM / SGLang / etc.). The agent reads user input, calls the
model, runs any tool calls the model asks for, persists the conversation, and repeats.

This document describes the module layout and, in particular, the **sub-agent** framework
and the options available for sub-agents.

## 1. Module layout

The agent loop was intentionally split out of one large `agent_loop.py` into small,
single-concern modules. `mini_code_cli/agent_loop.py` now owns only the top-level CLI
(argument parsing, banners, `AgentSession`) and *re-exports* the helpers the tests and any
external callers used to import from it, so `from mini_code_cli.agent_loop import ...` still
works.

| Module | Responsibility |
| --- | --- |
| `agent_loop.py` | CLI entry point (`main`, `parse_args`), the `AgentSession` loop, history compaction, plan-mode hooks, sub-agent delegation wiring. Re-exports helpers for backward compatibility. |
| `config.py` | Process-wide defaults (cache location, model URL, timeouts, `max_messages`). Shared by every module to avoid import cycles. |
| `prompts.py` | Builds the system prompt (tool list, allowed-dir note, plan-mode instructions) and the compaction/summary prompt. |
| `tools.py` | The tool *functions* the model can call (`read_file`, `write_file`, `grep`, `glob`, `web_search`, `read_website`, `run_shell_command`) plus `util_get_tools` / `util_get_tools_dict` which introspect their signatures into an OpenAI-style tool schema. |
| `persistence.py` | Everything about session histories on disk: `history_{n}.json` segments, the flat `summary_history.json` recorder, `usage.json`, and the restore/`find_last_session` helpers. Every loader is best-effort (never raises). |
| `sandbox.py` | The `--allowed-dir` filesystem sandbox (`is_within_allowed_dir`, `tool_paths_are_allowed`), using `os.path.realpath` to defeat symlinks / `..`. |
| `permissions.py` | Pure "does this tool call need a prompt?" helpers (`needs_permission`, `shell_forces_ask_permission`, `sudo_disables_prompts`). |
| `runner.py` | `call_function_with_timeout` -- runs a tool with a hard wall-clock limit via `SIGALRM`. |
| `shell_safety_utils.py` | Heuristic classifier that sorts shell commands into DENY / ASK / ALLOW tiers. |
| `plan.py` | Plan-mode bookkeeping: reads/writes `CURRENT_GOAL.md` and `CURRENT_PLAN.md`. Every method is a no-op when disabled. |
| `ui.py` | The whole terminal renderer (banner, turns, notes, permission box, the "… thinking" spinner, transient/non-scrolling tool-call region, history views). Stdlib only; degrades gracefully to ASCII / no-colour. |
| `utils.py` | `call_openai_server` (the one place we touch `requests`) and `determine_vllm_model`. |
| `subagents/` | The pluggable sub-agent framework (see §2). |

### Conversation state on disk

A session owns a folder under `cache_dir/{session_id}/` and keeps:

* **`history_{n}.json`** -- one file per conversation *segment*. Messages only ever append
  within a segment, so any earlier message index still marks a faithful prefix; a new
  segment (`n -> n+1`) is started every time the history is compacted.
* **`summary_history.json`** -- a flat, high-level *recorder*: one record per turn (the turn
  number, *what* happened, and which `history_{n}.json` + message index the turn lives in).
  This powers `/history`, `/history <n>` and `/reset <n>` without reading the full histories.
* **`usage.json`** -- accumulated token usage.

## 2. The sub-agent framework

A **sub-agent** is a self-contained agent the main agent can *delegate a task to*. It is
conceptually a *narrowed* `AgentSession`: its own system prompt, its own (usually smaller)
set of tools, it runs unattended in auto-mode, and it hands a text result back to the caller.

The design goal is **reuse**: a sub-agent re-implements *nothing*. It spins up a real nested
`AgentSession` and reuses every existing mechanism -- LLM calls, tools, the shell-safety
classifier, the `--allowed-dir` sandbox, history compaction and `--plan` mode.

### 2.1 The pieces

| File | Role |
| --- | --- |
| `subagents/base.py` | The `SubAgent` base class every sub-agent implements. Owns the tool schema, the nested system prompt, the child `arg` namespace and the `run()` that drives the nested session to completion. |
| `subagents/builtin.py` | Two worked examples: `ResearchSubAgent` and `PlanSubAgent`. They are registered on import and double as documentation ("a sub-agent is just a small class configuring a name, a description, and a few knobs"). |
| `subagents/__init__.py` | The pluggable registry: `register`, `unregister`, `get_registry`, `clear_registry` (plus the `SubAgent` re-export). |

### 2.2 How a sub-agent becomes a tool

When `--enable-subagents` is set, `AgentSession.__init__` pulls `subagents.get_registry()`
and, for each registered sub-agent, appends its `tool_schema()` to the tools offered to the
LLM and adds a *sentinel* entry to `tools_dict` (`function = None`, flagged
`subagent=True`). When the model calls that tool, `_run_one_tool_call` notices the sentinel
and delegates to `_run_subagent`, which calls `subagent.run(task, parent, max_steps)`.

`subagent.run()` builds the child `arg` namespace (`_build_child_args`), spins up a real
`AgentSession` for the child with a **swallowed** transcript (its output is captured to a
`StringIO` so the parent's screen is not spammed), drives it in auto-mode
(`_drive_auto`, capped at `max_steps`), and returns a text result assembled by
`_build_result` from the child's final answer and any `report.md` it wrote.

Two safety properties:

* **No recursion.** A child session is built with `enable_subagents` unset, so a sub-agent
  cannot itself spawn sub-agents.
* **Plan mode composes.** A sub-agent keeps its *own* `CURRENT_PLAN.md` / `CURRENT_GOAL.md`
  in its *own* session dir; the parent keeps its own. The `planner` sub-agent forces plan
  mode on for the nested session.

### 2.3 The report.md channel

A sub-agent hands work back to the parent in two ways: the nested session's final assistant
text, and an optional `report.md` file.

The child's system prompt tells it exactly where to write the report
(`self._report_path(parent)` = `<child cache dir>/report.md`), and `run()` reads it back
from that same path -- so the two always agree.

Because the child cache dir lives under the parent's cache dir, which is *usually outside*
`--allowed-dir`, the child's `write_file` of `report.md` would normally be blocked by the
sandbox. This is fixed by a scoped `_is_report_file` exemption (mirroring the plan-file
logic in `agent_loop.py`): the child session is built with a `report_path`, and only a
`read_file`/`write_file` of *exactly that file* is exempt. Any other path under the cache
dir stays fully sandboxed.

### 2.4 Adding a new sub-agent

1. Subclass `SubAgent` and set `name`, `description`, and the knobs you want
   (`tools`, `enable_shell`, `max_steps`, `system_prompt`, ...).
2. Register it: `subagents.register(MySubAgent())` (typically in `subagents/builtin.py`,
   or in your own module that you import).

The agent loop then discovers it automatically and exposes it as a tool.

## 3. Options for sub-agents

Every sub-agent advertises a small, common parameter set to the model in its tool schema
(defined by `SubAgent.tool_schema()`), and subclasses can extend it. Options fall into two
groups: the knobs a *caller* (the main agent, or `register()`) can set on the sub-agent,
and the parameters the *LLM* can pass when it invokes the tool.

### 3.1 Knobs a caller / subclass can configure (on the class or via `register()`)

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | str | `"subagent"` | Tool name the model calls. Must be unique in the registry. |
| `description` | str | generic text | One-line description shown to the model in the tool schema (the help text / sub-agents list). |
| `tools` | list[str] or None | `None` (inherit parent's full set) | Restricts which tools the nested agent may use, e.g. `["web_search", "read_website"]`. `None` inherits everything. |
| `enable_shell` | bool | `False` | Whether the nested agent may run shell commands. Always intersected with the parent's `--enable-shell` -- a sub-agent can never widen the parent's permissions. |
| `max_steps` | int or None | `None` (unbounded) | Cap on how many model calls the nested agent may make before returning. Guards against loops. |
| `system_prompt` | method | default | Override to customise the nested system prompt (called with `task, parent`). See `research` / `planner` for `super().system_prompt(...)` extension. |
| `wants_plan_mode` | method | matches parent | Whether the nested session runs in plan mode. |
| `_extra_tool_parameters` | method | `{}` | Extend the tool schema with sub-agent-specific parameters (see `sources` below). |

In addition, a subclass may set per-instance options through the constructor, e.g.
`register(MySubAgent(sources="wikipedia.org"))`; those land in `self.options` and can be
read inside `system_prompt`, `_resolve_tools`, etc.

### 3.2 Parameters the LLM can pass when calling the tool

| Parameter | Type | Meaning |
| --- | --- | --- |
| `task` (required) | string | A clear, self-contained description of the task the sub-agent should complete. |
| `max_steps` (optional) | integer | Per-invocation cap on model calls. Overrides the subclass default for this one call; leaving it out uses the subclass default. |
| *(subclass-defined)* | varies | Any parameters added via `_extra_tool_parameters`, e.g. `sources` (a domain hint the `research` sub-agent folds into its prompt). Advisory only. |

### 3.3 Inheritance: what the child gets from the parent

A child session inherits the parent's server/cache/allowed-dir/timeout configuration; only
the narrow things a sub-agent controls (its tools, shell setting, optional model/url
override, plan-mode behaviour and its own report path) differ. Inherited values:
`url`, `api_key`, `max_tokens`, `temperature`, `llm_timeout`, `tool_timeout`, `allowed_dir`,
`cache_dir` (relocated under the parent session), `plan` (unless the subclass overrides).
`auto_mode` is always forced `True` for children; `ask_permission` / `sudo` are always
`False`.

### 3.4 Worked examples

**`research`** (`ResearchSubAgent`): read-only web sub-agent. `tools = ["web_search",
"read_website"]`, `enable_shell = False`, `max_steps = 8`, and an extra `sources` parameter
(a domain hint folded into the prompt). Safe to enable even in non-sandboxed sessions since
it cannot touch the filesystem or run shell commands.

**`planner`** (`PlanSubAgent`): drafts/refines a step-by-step plan. `tools = ["read_file",
"write_file"]`, `max_steps = 5`, and `wants_plan_mode` forced `True` so the nested agent
keeps a tidy `CURRENT_PLAN.md` / `CURRENT_GOAL.md`. It writes the plan both to
`CURRENT_PLAN.md` and to `report.md` so the parent receives it.

## 4. Combining sub-agents with plan mode

Sub-agents and plan mode compose cleanly:

* The **parent** keeps its own `CURRENT_GOAL.md` / `CURRENT_PLAN.md` in its session dir.
* Each **sub-agent** keeps its own copies in its own isolated session dir (a fresh folder
  under the parent's cache dir), typically working on a single plan item.
* After a compaction, plan mode re-injects the running plan + goal; the nested session does
  the same for its own plan/goal.
* A sub-agent can additionally return its result via `report.md`, which the parent surfaces
  to the model (with a sandbox exemption for that single file).

## 5. Extending beyond sub-agents

The same "small focused module + re-export" layout keeps the surface for further changes
small:

* **New tools** live only in `tools.py` (and show up in the schema automatically via
  `util_get_tools`, which introspects the signature). The sandbox stays correct because
  `_FILE_TOOL_PATH_ARGS` in `sandbox.py` enumerates every path-bearing argument.
* **New permissions behaviour** lives in `permissions.py` and `shell_safety_utils.py`.
* **New persistence shapes** live in `persistence.py` (best-effort loaders throughout).
* **New sub-agents** live in `subagents/` (see §2.4).
