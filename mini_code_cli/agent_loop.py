"""mini_code_cli.agent_loop -- the interactive agent loop.

Usage:
    mini-code

This module owns the top-level CLI: argument parsing, the welcome/help banners
and the :class:`AgentSession` that drives the read-input -> call-the-LLM -> run
tool-calls -> persist loop.

The pieces that had grown organically inside this file have been extracted into
smaller, focused modules so each concern is easy to read and test:

  * :mod:`mini_code_cli.persistence` -- session histories / recorder / usage on disk.
  * :mod:`mini_code_cli.sandbox`     -- the ``--allowed-dir`` path sandbox.
  * :mod:`mini_code_cli.permissions` -- pure "does this need a prompt?" helpers.
  * :mod:`mini_code_cli.runner`      -- running a tool with a wall-clock timeout.
  * :mod:`mini_code_cli.config`      -- shared default settings.
  * :mod:`mini_code_cli.subagents`   -- delegating a task to a sub-agent.
  * :mod:`mini_code_cli.skills`      -- the ``.mini`` skills feature.

For backward compatibility (and the test-suite) the most commonly imported
helpers are re-exported from here, so ``from mini_code_cli.agent_loop import ...``
keeps working.
"""

import os
import json
import argparse

from . import prompts
from . import shell_safety_utils
from . import tools
from . import utils
from .utils import oneline
from . import ui
from . import subagents
from . import skills
from . import plan as plan_utils
from .config import (
    DEFAULT_CACHE_LOCATION,
    DEFAULT_MODEL_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_MAX_MESSAGES,
)
from .persistence import (
    save_history,
    save_summary,
    save_usage_segment,
    save_total_usage,
    load_history,
    load_history_from_dir,
    load_summary_from_dir,
    load_usage_segment,
    load_total_usage_from_dir,
    empty_usage,
    find_last_session_dir,
    save_last_session_pointer,
    accumulate_usage,
    generate_unique_id,
    resolve_session_dir,
)
from .sandbox import is_within_allowed_dir, tool_paths_are_allowed, _FILE_TOOL_PATH_ARGS
from .permissions import (
    needs_permission,
    shell_forces_ask_permission,
    sudo_disables_prompts,
)
from .runner import call_function_with_timeout


__all__ = [
    # re-exported helpers (kept for the test-suite / external callers)
    "is_within_allowed_dir",
    "tool_paths_are_allowed",
    "_FILE_TOOL_PATH_ARGS",
    "maybe_compact",
    "accumulate_usage",
    "shell_forces_ask_permission",
    "sudo_disables_prompts",
    "needs_permission",
    "call_function_with_timeout",
    "save_history",
    "save_summary",
    "load_history",
    "generate_unique_id",
    "find_last_session_dir",
    # public classes / entry points
    "AgentSession",
    "main",
    "parse_args",
]


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    # MODEL / LLM related
    parser.add_argument("--url", type=str, default=f"{DEFAULT_MODEL_URL}", help=f"API host, default={DEFAULT_MODEL_URL}")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for LLM response, default=None")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature for LLM, default={DEFAULT_TEMPERATURE}")
    parser.add_argument("--model", type=str, help="Model name, default=None")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication. If not set, it tries to find it in env variable: MINI_CODE_API_KEY.")

    # Other settings:
    parser.add_argument("--cache-dir", type=str, help=f"The cache dir for the session histories. default={DEFAULT_CACHE_LOCATION}")
    parser.add_argument("--restore-session", type=str, help="The location of a previous session to continue. (e.g. ~/.cache/mini_code/sessions/1792028090823027-a9d77653/)")
    parser.add_argument("-l","--load-last-session", action="store_true", help="Continue the most recent previous session recorded in the --allowed-dir .mini pointer, in place and without the resume prompt, instead of starting fresh. --restore-session takes precedence if also given.")

    # Agent behaviour related
    parser.add_argument("--system-prompt", type=str, help="Replace system prompt, with a custom system prompt.")
    parser.add_argument("-a", "--auto-mode", action="store_true", help="Whether to run the agent in `auto-mode'. Or default: `manual-mode'.")
    parser.add_argument("--agent-md", type=str, help="If the agent should use an agent-md file... (it will added after system message.)")
    parser.add_argument("--prompt", type=str, help="An initial prompt from the user...")

    # Agent permissions related
    parser.add_argument("--ask-permission", action="store_true", help="Ask for permission before any tool call.")
    parser.add_argument("--sudo", action="store_true", help="DANGEROUS: never ask for permission. Every non-destructive tool call and shell command runs automatically without confirmation. Destructive (DENY-tier) commands are still refused.")
    parser.add_argument("--allowed-dir", type=str, default=".", help="Allowed directory for file operations, default='./'")
    parser.add_argument("-s", "--enable-shell", action="store_true", help="Allow shell execution. default= False")
    parser.add_argument("-g","--enable-subagents", action="store_true", help="Allow the agent to delegate work to sub-agents (exposed as extra tools). default= False")
    parser.add_argument("--skills", action="store_true", help="Enable the skills feature: a local .mini directory (inside --allowed-dir) of skill.md files, skill sub-dirs, a high-level SKILLS.md summary, and executable Python scripts, exposed to the agent as list_skills/read_skill/create_skill/run_skill tools. default= False")
    parser.add_argument("--llm-timeout", type=int, default=DEFAULT_HTTP_TIMEOUT, help=f"HTTP timeout in seconds for LLM requests (reasoning models can be slow). default={DEFAULT_HTTP_TIMEOUT}")
    parser.add_argument("--tool-timeout", type=int, default=DEFAULT_TOOL_TIMEOUT, help=f"Timeout in seconds for a single tool call. default={DEFAULT_TOOL_TIMEOUT}")
    parser.add_argument("--max-messages", type=int, default=DEFAULT_MAX_MESSAGES, help=f"History length (in messages) before the conversation is compacted into a summary. Lower it to compact sooner on small context windows. default={DEFAULT_MAX_MESSAGES}")
    parser.add_argument("-p","--plan", action="store_true", help="Enable plan mode: the agent keeps a CURRENT_PLAN.md (its step-by-step plan, marked finished when the goal is done) and CURRENT_GOAL.md (the user's latest instruction) in the session directory. Also changes how history compaction works.")

    parser.add_argument("--all", action="store_true", help="Enable: --plan, --subagents, --shell, --automode, --sudo")
    parser.add_argument("--all-last", action="store_true", help="Enable: --plan, --subagents, --shell, --automode, --sudo, --load-last-session")

    # Output verbosity
    parser.add_argument("--verbose", action="store_true", help="Print the full, detailed output. Without it (default), only user/assistant turns, token usage and tool calls are shown.")

    args = parser.parse_args()

    if args.all or args.all_last:
        args.plan = True
        args.enable_subagents = True
        args.enable_shell = True
        args.sudo = True
        args.auto_mode = True
        args.skills = True
    
    if args.all_last:
        args.load_last_session = True

    return args


# ---------------------------------------------------------------------------
# banners / helpers
# ---------------------------------------------------------------------------

def style_line_prompt(text):
    """A short, styled inline prompt (used for the 'You:' line)."""
    return ui.console.style(text, bold=True, fg="cyan") + " "


def print_help(allowed_dir, server_url, args, unique_id, cache_dir, model_name):
    ui.console.rule("help", fg="cyan")
    if not args.auto_mode:
        help_message = "Manual-mode: the agent asks for your input after each LLM prediction."
    else:
        help_message = "Auto-mode: the agent keeps calling the LLM until no further tool calls are possible."
    ui.console.emit(ui.console.style(help_message, dim=True, fg="cyan"))
    ui.console.emit(ui.console.style("History / navigation commands:", bold=True, fg="cyan"))
    ui.console.emit("  /history         list the conversation (turn number, what happened, where)")
    ui.console.emit("  /history <n>      show the full messages recorded for turn n")
    ui.console.emit("  /plan             show the current goal + plan (plan mode only)")
    ui.console.emit("  /reset [<n>]      rewind to the end of turn n (drops everything after it)")
    ui.console.emit("  /clear            clear the screen (keeps the conversation history)")
    ui.console.emit("  /help             show this help")
    ui.console.emit("  /quit, /exit      save the session and exit")
    if args.enable_subagents or args.verbose:
        ui.console.emit(ui.console.style("Sub-agents:", bold=True, fg="cyan"))
        for sa in subagents.get_registry().values():
            ui.console.emit(f"  {sa.name:16} {sa.description}")
    if args.enable_skills or args.verbose:
        ui.console.emit(ui.console.style("Skills:", bold=True, fg="cyan"))
        for name in skills.SKILL_TOOL_NAMES:
            ui.console.emit(f"  {name:16} (from the local .mini directory)")
    ui.console.rule(char="═", fg="cyan")


def print_welcome(allowed_dir, server_url, args, unique_id, cache_dir, model_name=None):
    """Render the startup info banner through the UI."""
    version = __import__('mini_code_cli').__version__

    if getattr(args, "sudo", False):
        permission_mode = "SUDO MODE: no permission prompts (DENY commands still blocked)."
    elif args.ask_permission:
        permission_mode = "All tool calls require permission."
    else:
        permission_mode = "Tool calls execute without asking."
    agent_mode = "Agent is in manual-mode." if not args.auto_mode else "Agent is in auto-mode."
    shell_mode = "Shell disabled." if not args.enable_shell else "Shell enabled."

    ui.console.welcome(version, unique_id, cache_dir, allowed_dir,
                       server_url, model_name, permission_mode, agent_mode, shell_mode)


def _make_tool_call_note(console, count):
    console.note(f"LLM is trying to call {count} tool(s).", "info")


# ---------------------------------------------------------------------------
# history compaction
# ---------------------------------------------------------------------------

def maybe_compact(messages, server_url, model, api_key, llm_timeout, token_stats, history_index=0, max_messages=DEFAULT_MAX_MESSAGES, plan_hook=None, console=None):
    """Compact the conversation history once it grows beyond ``max_messages``.

    Long (especially auto-mode) sessions can overflow the model's context window.
    When the history gets too large we ask the LLM to summarize it and replace the
    old turns with that summary, keeping the original system prompt as the first
    message. The compaction call itself is a single, tool-free request, so it
    cannot trigger further tool calls or recurse into compaction.

    ``plan_hook`` is an optional callable the caller supplies only when running
    under plan mode (in practice ``AgentSession._handle_compaction``). After the
    summary is built the hook re-injects the running goal + plan so the compacted
    messages carry them forward. Passing it in -- instead of referencing the
    owning session -- keeps ``maybe_compact`` a small, module-level, network-free
    helper that the test-suite can import and exercise directly.
    """
    # ``console`` defaults to None so callers may omit it, but every branch below
    # reports through a console. Falling back to the shared one keeps the
    # documented signature usable -- without this a bare maybe_compact() call hit
    # "'NoneType' object has no attribute 'start_thinking'" inside the try block,
    # which the except then swallowed as a failed compaction.
    if console is None:
        console = ui.console

    if len(messages) <= max_messages:
        # No compaction: keep the same segment (history_index) and signal the
        # caller with (messages, did_compact=False, unchanged_index). Returning
        # the full triple here is required -- callers unpack all three, so a
        # bare messages would crash them.
        return messages, False, history_index

    try:
        console.note(f"Compacting conversation history ({len(messages)} messages -> summary)...", "warn")
        summary_prompt = prompts.compaction_prompt(messages)

        # The summariser sometimes answers with nothing at all. An empty summary
        # must never be mistaken for a successful compaction: the caller would
        # think the history had been shrunk while the over-long transcript was
        # kept, and the rollover would be lost. Retry once, then give up loudly
        # (keeping the messages and the segment index) so the user can see that
        # compaction is not working -- previously this fell through silently and
        # the over-cap transcript was retried, and paid for, on every step.
        summary = ""
        for attempt in (1, 2):
            console.start_thinking("compacting")
            try:
                reply, _, compact_usage = utils.call_openai_server(
                    [{"role": "user", "content": summary_prompt}],
                    max_tokens=1024,
                    temperature=0.0,
                    model=model,
                    server_url=server_url,
                    api_key=api_key,
                    timeout=llm_timeout,
                )
            finally:
                console.stop_thinking()

            accumulate_usage(token_stats, compact_usage)
            summary = (reply or "").strip()
            if summary:
                break
            if attempt == 1:
                console.note("The summariser returned an empty summary; retrying once.", "warn")

        if not summary:
            console.note(
                f"Could not compact history: the summariser returned an empty "
                f"summary twice. Keeping all {len(messages)} messages "
                f"(over the {max_messages}-message cap).", "warn")
            return messages, False, history_index

        system_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        compacted = []
        if system_content:
            compacted.append({"role": "system", "content": system_content})
        compacted.append({
            "role": "user",
            "content": f"Here is a summary of our earlier conversation so far:\n{summary}",
        })

        # In plan mode, re-inject the running plan + current goal so a
        # compaction (which would otherwise drop them) keeps the model
        # working against the goal it is still trying to finish. The hook is
        # wired in only when the caller runs under plan mode; otherwise
        # compaction degrades to a plain summary.
        if plan_hook is not None:
            plan_hook(compacted)

        console.note(f"History compacted to {len(compacted)} messages.", "warn")
        # A new segment (history_index + 1) is started on compaction: the
        # compacted messages are its snapshot, and every later turn appends
        # to it. The caller records a single high-level 'summary' turn for
        # this index so the recorder stays flat across the rollover.
        return compacted, True, history_index + 1
    except Exception as e:
        console.note(f"Failed to compact history: {e}", "error")

    return messages, False, history_index


# ---------------------------------------------------------------------------
# the agent session
# ---------------------------------------------------------------------------

class AgentSession:
    """A single interactive agent session.

    The whole loop used to live in one ~300 line ``main()``. Splitting it into
    small, named methods keeps each step (read input -> call the LLM -> run tool
    calls -> persist) easy to follow and gives the interesting bits somewhere to
    be unit-tested.
    """

    EXIT_COMMANDS = {"quit", "exit", "/quit", "/exit", "exit()", "quit()"}
    CLEAR_COMMANDS = {"clear", "/clear"}

    def __init__(self, args, console=None, background_mode = False):
        self.args = args
        self.allowed_dir = os.path.abspath(args.allowed_dir)
        self.cache_dir = args.cache_dir if args.cache_dir else DEFAULT_CACHE_LOCATION

        self.console = console if console is not None else ui.console
        self.console.verbose = getattr(args, "verbose", False)

        # LLM / server configuration.
        self.server_url = f"{args.url}"
        self.api_key = args.api_key if args.api_key else os.environ.get("MINI_CODE_API_KEY", "")
        self.model = self._resolve_model(args)
        
        self._register_tools(args)

        # report_path is set on a *sub-agent child* session (by
        # SubAgent._build_child_args) 
        self.report_path = getattr(args, "report_path", None)

        self.plan_mode = bool(getattr(self.args, "plan", False))

        # Token accounting. ``token_stats`` covers the *current history
        # segment* and is reset when compaction rolls over to a new one;
        # ``total_usage`` is the whole-session accumulator and never decreases.
        # Keeping the two apart is what lets usage survive a compaction.
        self.token_stats = empty_usage()
        self.total_usage = empty_usage()

        self.background_mode = background_mode

        self.restore_session = getattr(self.args, "restore_session", None) #specific session
        # --load-last-session is an explicit *instruction* to continue the session
        # recorded in --allowed-dir/.mini, so it resumes without the interactive
        # offer (a piped/non-TTY stdin would otherwise answer that offer with EOF
        # and exit without doing anything). --restore-session still wins.
        self.load_last_session = bool(getattr(self.args, "load_last_session", False))
        
    def start_session(self):
        """Loads existing session, detects existing session and asks or starts a new session.

        A *background* session (a sub-agent's, see ``background_mode``) short
        circuits the detection entirely: it neither consults the ``.mini`` resume
        pointer nor offers the interactive resume prompt, so it always begins a
        fresh isolated session. That is what makes delegation safe to run
        unattended -- there is a single place that decides this, rather than
        every caller having to remember.
        """
        loading_success = False
        if self.background_mode and not self.restore_session:
            # Unattended: no pointer lookup, no resume prompt, no user input.
            self.start_new_session()
            return

        if self.restore_session:
            try:
                restore_directory = resolve_session_dir(self.restore_session, self.cache_dir)        
                loading_success = self.restore_previous_session(restore_directory)
            except Exception as e:
                pass
            
        else:
            try:
                detected = self._detect_last_session()
                if detected and not self.background_mode and self.load_last_session:
                    self.console.note("Continuing the last session (--load-last-session).", "warn")
                    restore_directory = resolve_session_dir(detected, self.cache_dir)
                    loading_success = self.restore_previous_session(restore_directory)
                elif detected and not self.background_mode:
                    answer = self.show_resume_choice()
                    if answer == "resume":
                        self.console.note("Resuming the session.", "warn")
                        restore_directory = resolve_session_dir(detected, self.cache_dir)     
                        loading_success = self.restore_previous_session(restore_directory)
                    elif answer == "exit":
                        self.console.note("User chose to exit! Exiting.", "warn")
                        raise SystemExit("") # Exit gracefully instead of crashing
                    else:
                        self.console.note("Will start with new session!", "warn")

            except Exception as e:
                pass
                   
        if not loading_success:
            self.start_new_session()

    def start_new_session(self):
        """A function to start a new session."""
        self.unique_id = generate_unique_id()
        self.session_dir = os.path.join(self.cache_dir, self.unique_id)

        self.plan_manager = plan_utils.PlanManager(
            self.session_dir, enabled=self.plan_mode)

        self.history_index = 0
        self.summary = []

        self.messages = self._build_initial_messages()

        save_history(self.messages, self.cache_dir, self.unique_id, self.history_index)
        save_summary(self.summary, self.cache_dir, self.unique_id)

        # Record this session in --allowed-dir/.mini so a later
        # --load-last-session (or the interactive resume offer) can find it.
        self._save_last_session_pointer()


    def restore_previous_session(self, restore_directory):
        """A function to restore from a directory.

        Order matters here. ``self.messages`` must be loaded *before* the
        recorder is restored, because the legacy fallback in
        :meth:`_load_summary_if_restored` rebuilds the recorder by scanning the
        restored messages. ``self.summary`` / ``self.history_index`` are seeded
        first so that fallback can append to them and so ``_persist()`` has a
        segment index to write to. Previously all three were unset on this path,
        so restoring a session folder that had a ``history_N.json`` but no
        ``summary_history.json`` (older code, or a hand-made folder) raised
        AttributeError and the caller silently started a *new* session instead.
        """
        try:
            self.unique_id = os.path.basename(os.path.normpath(restore_directory))
            self.session_dir = restore_directory
            self.plan_manager = plan_utils.PlanManager(
                self.session_dir, enabled=self.plan_mode)
            self.summary = []
            self.history_index = 0
            restored = load_history_from_dir(restore_directory) if restore_directory else None

            self.messages = restored if restored is not None else self._build_initial_messages()
            # The recorder restore decides which segment we are on, so usage --
            # which is stored per segment -- has to be restored after it.
            self._load_summary_if_restored(restore_directory)
            self._restore_usage_if_available(restore_directory)

            # Continuing a session in place makes it the latest one again: keep
            # the pointer aimed at it so a later --load-last-session finds the
            # conversation the user is actually working in.
            self._save_last_session_pointer()

            return True

        except Exception as e:
            self.unique_id = None
            self.session_dir = None
            self.plan_manager = None
            return False


    def _detect_last_session(self):
        """Resolve the most recent previous session dir, once.

        There is deliberately NO timestamp/mtime fallback over the cache dir: a
        previous session is restored only via the ``.mini`` pointer (an explicit
        record of the session the user last worked in). Returns the path, or None.
        Called at most once per session (see __init__).
        """
        return find_last_session_dir(self.allowed_dir)

    def _resolve_model(self, args):
        """Pick the model id: the one the user named, else the server's default.

        The ``/v1/models`` probe is *lazy*: it only runs when ``--model`` was not
        given, so a normal run with an explicit model makes no extra request and
        cannot be delayed by an unreachable server. It is also exception-safe --
        a failed probe warns once and falls back to ``args.model`` (possibly
        None) instead of killing the session. It used to be a bare ``except:``
        around an unconditional probe, which swallowed KeyboardInterrupt and
        turned every construction into a network round-trip.
        """
        if args.model:
            return args.model
        try:
            detected = utils.determine_vllm_model(
                self.server_url, api_key=self.api_key or None,
                timeout=getattr(args, "llm_timeout", DEFAULT_HTTP_TIMEOUT))
        except Exception as e:
            self.console.note(
                f"Could not detect a model served at {self.server_url} ({e}); "
                f"falling back to --model.", "warn")
            return args.model
        if detected:
            self.console.note(f"Using the model served at {self.server_url}: {detected}", "info")
        return detected or args.model

    def _register_tools(self, args):
        """Wire up which tools the LLM may call, and register the opt-in ones.

        Base tools depend on --enable-shell (the shell tool is withheld unless
        shell is explicitly on). Then two opt-in features each add their own
        sandbox-safe tools:

          * Sub-agents (--enable-subagents): every registered sub-agent becomes
            an extra tool. Its tools_dict entry has ``function=None`` so
            _run_one_tool_call knows to *delegate* to a nested session rather
            than call a real function. A child session never sets
            enable_subagents, so sub-agents cannot recursively spawn sub-agents.
          * Skills (--skills): the .mini directory (inside --allowed-dir) exposes
            list/read/create/run_skill tools. They are sandbox-safe -- names are
            validated as single path segments and resolved under .mini, so they
            never accept an arbitrary path from the model and are usable even
            without --enable-shell.
        """
        forbidden = None if args.enable_shell else tools.shell_tools
        self.tools_dict = tools.util_get_tools_dict(forbidden_tools=forbidden)
        self.llm_tools = tools.util_get_tools(forbidden_tools=forbidden)

        self.enable_subagents = bool(getattr(args, "enable_subagents", False))
        self.subagents_registry = subagents.get_registry() if self.enable_subagents else {}
        for sa in self.subagents_registry.values():
            self.llm_tools.append(sa.tool_schema())
            self.tools_dict[sa.name] = {"function": None, "description": sa.description, "subagent": True}

        self.enable_skills = bool(getattr(args, "enable_skills", False))
        if self.enable_skills:
            # list/create/run_skill + the scoped read/write_skill_file tools.
            self.tools_dict.update(skills.get_skills_tools_dict(self.allowed_dir))
            self.llm_tools.extend(skills.get_skills_tools(self.allowed_dir))
            self.tools_dict.update(skills.get_skills_file_tools_dict(self.allowed_dir))
            self.llm_tools.extend(skills.get_skills_file_tools(self.allowed_dir))


    def _build_initial_messages(self):
        """Assemble the system prompt and optional Agent.md into the first turns."""
        if self.args.system_prompt:
            system_prompt = self.args.system_prompt
        else:
            system_prompt = prompts.system_prompt(
                tools_dict=self.tools_dict, allowed_dir=self.allowed_dir,
                plan_mode=self.plan_mode,
                skills_dir=self.allowed_dir if self.enable_skills else None,
                session_dir=self.session_dir,
            )

        messages = [{"role": "system", "content": system_prompt}]

        if self.args.agent_md:
            self.console.note(f"AGENT.md '{self.args.agent_md}' specified.", "warn")
            if not os.path.isfile(self.args.agent_md):
                self.console.note(f"Error: The AGENT.md '{self.args.agent_md}' does not exist.", "error")
            else:
                with open(self.args.agent_md, "r", encoding="utf-8") as f:
                    agent_md_content = f.read()
                self.console.note(f"AGENT.md '{self.args.agent_md}' loaded.", "ok")
                # Append Agent.md as a user turn, right after the system prompt.
                messages.insert(1, {"role": "user", "content": agent_md_content})

        return messages

    def _restore_usage_if_available(self, restore_directory):
        """Restore the per-segment and whole-session token totals.

        ``token_stats`` comes from the segment we restored onto
        (``usage_{history_index}.json``) and ``total_usage`` from
        ``total_usage.json`` (falling back to the sum of all segments), so the
        running summary stays continuous across a restore *and* across the
        compaction that may have happened before it. Best-effort: any problem
        leaves the totals at zero rather than breaking the restore.
        """
        segment = load_usage_segment(restore_directory, self.history_index)
        total = load_total_usage_from_dir(restore_directory)
        # A restored segment can never have cost more than the recorded total.
        for key in total:
            total[key] = max(total[key], segment[key])
        self.token_stats = segment
        self.total_usage = total

    def _record_usage(self, usage):
        """Bill one LLM call to both the current segment and the session total."""
        accumulate_usage(self.token_stats, usage)
        accumulate_usage(self.total_usage, usage)

    def _bill_compaction(self, segment_before):
        """Carry the compaction call's spend over into the session total.

        :func:`maybe_compact` is only handed the segment dict, so the tokens the
        summariser spent land there and nowhere else. The rollover that follows
        zeroes the segment counters, so without this the summarisation cost would
        silently vanish from the session total. Deltas are per key because
        :func:`accumulate_usage` only ever adds.
        """
        for key, before in segment_before.items():
            delta = self.token_stats.get(key, 0) - before
            if delta > 0:
                self.total_usage[key] = self.total_usage.get(key, 0) + delta

    def _load_summary_if_restored(self, restore_directory):
        """Restore the high-level recorder and segment index for a restore.

        Prefers the saved ``summary_history.json`` so the recorder stays flat and
        contiguous across the restore, seeding ``self.summary`` and the next
        ``history_index`` from its last record. If no recorder was saved (e.g. a
        session created by older code, or only a classic ``history.json``), we
        rebuild a minimal recorder by scanning the restored messages for
        assistant turns, so navigation still works. Best-effort only.
        """
        saved = load_summary_from_dir(restore_directory)

        saved = [r for r in saved if isinstance(r, dict)]
        if saved:
            self.summary = saved
            self.history_index = int(saved[-1].get("history_file", 0))
            return
        turn = 0
        for i, m in enumerate(self.messages):
            if m.get("role") == "assistant":
                turn += 1
                self.summary.append({"turn": turn, "what": "assistant",
                                     "history_file": 0, "index": i})

    def _record_turn(self, what, messages, history_index=None):
        """Append one entry to the high-level recorder.

        Each entry records the running turn number, a one-line *what* (assistant
        reply, tool call(s), or a history compaction summary) and exactly where
        the turn lives -- ``history_{history_index}.json`` at message ``index``.
        ``index`` points at the last message appended this step, i.e. the
        end-of-turn snapshot that replaying ``messages`` reconstructs, which is
        what lets :meth:`_reset_history` rewind to that exact state.
        """
        if history_index is None:
            history_index = self.history_index
        self.summary.append({
            "turn": len(self.summary) + 1,
            "what": what,
            "history_file": history_index,
            "index": max(len(messages) - 1, 0),
        })

    def _show_history(self):
        """Print the high-level recorder as a numbered list of turns."""
        rows = [(str(r.get("turn", i + 1)),
                 r.get("what", "?"),
                 f"history_{r.get('history_file', 0)}.json#{r.get('index', 0)}")
                for i, r in enumerate(self.summary)]
        self.console.history_list(rows)

    def _inspect_history(self, arg):
        """Print the full messages recorded for a single turn."""
        try:
            record = self.summary[-1] if arg == "last" else self.summary[int(arg) - 1]
        except (ValueError, IndexError):
            self.console.note(f"No such turn {arg!r} (have 1..{len(self.summary)}).", "error")
            return
        messages = self._segment_upto(record)
        if messages is None:
            return
        self.console.history_detail(record, messages, total=len(self.summary))

    def _segment_upto(self, record):
        """Load a recorded turn's segment and truncate it at its index.

        Returns the messages reproducing the conversation state at the end of
        that turn, or None if the segment could not be loaded (never raises: a
        missing or moved segment must not crash navigation).
        """
        directory = os.path.join(self.cache_dir, self.unique_id)
        segment_path = os.path.join(directory, f"history_{record.get('history_file', 0)}.json")
        try:
            with open(segment_path, "r", encoding="utf-8") as f:
                segment = json.load(f)
        except Exception:
            self.console.note("Could not load the history segment for that turn.", "error")
            return None
        idx = max(0, min(int(record.get("index", 0)), len(segment) - 1))
        return segment[:idx + 1]

    def _reset_history(self, keep, record):
        """Rewind the conversation to the end of a recorded turn.

        Rebuilds ``self.messages`` from the turn's recorded segment + index,
        moves the session back onto that segment (``history_index``) and drops
        everything after it -- both the extra messages and the later recorder
        entries. New turns then continue from that state. Best-effort only: a
        segment that cannot be loaded leaves the session unchanged.
        """
        messages = self._segment_upto(record)
        if messages is None:
            return
        self.messages = messages
        self.history_index = int(record.get("history_file", 0))
        self.summary = self.summary[:keep]
        save_history(self.messages, self.cache_dir, self.unique_id, self.history_index)
        save_summary(self.summary, self.cache_dir, self.unique_id)
        self.console.note(
            f"Reset to turn {keep}: kept {len(self.messages)} messages "
            f"(history_{self.history_index}.json up to index "
            f"{int(record.get('index', 0))}).", "ok")
        self.console.note("Continuing: new turns will append to this state.", "info")

    def _warn_sudo(self):
        """Print a prominent red warning about --sudo mode.

        --sudo means the agent will never stop to ask before acting, so the user
        is effectively handing over unattended control of their machine (within
        the DENY-tier limits). We render it as a red box so it stands out on any
        real terminal (it degrades to a plain ASCII box when colour is off).
        """
        self.console.print_box(
            [
                "SUDO MODE: the agent will NEVER ask for permission.",
                "All tool calls and shell commands run automatically and unattended.",
                "Only clearly-destructive (DENY-tier) commands are still refused.",
                "You are responsible for everything it does on this machine.",
            ],
            title="DANGER",
            fg="red",
        )

    def print_announcements(self):
        """Print the info / warning banners that depend on the CLI flags."""
        if self.api_key:
            self.console.announce_api_key(present=True)
        else:
            self.console.announce_api_key(present=False)

        if shell_forces_ask_permission(self.args):
            self.console.announce_shell()

        # --sudo hands the agent unattended control, so make that impossible to miss.
        if sudo_disables_prompts(self.args):
            self._warn_sudo()

        if self.enable_subagents:
            self.console.note(
                f"Sub-agents enabled ({len(self.subagents_registry)} available: "
                f"{', '.join(self.subagents_registry)}).", "info")

    def show_resume_choice(self):
        """Prompt for the resume choice and return its label.

        Returns one of ``"resume"``, ``"new"`` or ``"exit"``. The prompt
        degrades cleanly on a non-TTY / piped console: an EOF / Ctrl-C (``read_line``
        returns None) is treated as ``"exit"`` so the loop never spins on unread input.
        """
        self.console.emit(
            self.console.style(
                "A previous session was found. What would you like to do?", dim=True,
                fg="cyan"))
        self.console.emit("  1) Resume the previous session (continue where you left off)")
        self.console.emit("  2) Start a new session (empty history)")
        self.console.emit("  3) Exit (start nothing)")
        while True:
            answer = self.console.read_line(
                self.console.style("Resume previous session? Type 1, 2, or 3: ", fg="cyan"))
            if answer is None:
                # EOF / Ctrl-C at the offer -> graceful exit, no session written.
                return "exit"
            choice = answer.strip().lower()
            if choice in ("1", "resume"):
                return "resume"
            if choice in ("2", "new"):
                return "new"
            if choice in ("3", "exit"):
                return "exit"
            self.console.note("Please choose 1, 2, or 3.", "warn")

    def run(self, initial_prompt=None):
        """Drive the agent loop until the user quits."""
        # Called in the beginning to init the session.
        self.start_session()


        # Hint at startup so the high-level history feature is discoverable:
        # /history browses the recorder, /history <n> inspects a turn, and
        # /reset <n> rewinds the conversation to that turn's state.
        tip = ("Tip: /history to browse the conversation, /history <n> to inspect a "
               "turn, /reset <n> to rewind. Type /help for all commands.")
        if getattr(self.args, "plan", False):
            tip = ("Tip: /plan to show the current goal + plan (plan mode), "
                   "/history to browse the conversation, /history <n> to inspect a "
                   "turn, /reset <n> to rewind. Type /help for all commands.")
        self.console.emit(self.console.style(tip, dim=True, fg="cyan"))

        # The --prompt value is a one-shot initial turn: it is sent verbatim and
        # then cleared, so later turns come from the user (manual mode) or are
        # produced automatically by the loop (auto mode).
        pending = initial_prompt

        running = True
        while running:
            try:
                # 1) Get the user's turn: the one-shot preset, or a fresh line.
                if pending is not None:
                    self.console.rule(title="Preset prompt", char="=", fg="cyan")
                    self.console.short_divider()
                    self.console.turn("You", pending, accent="cyan")
                    user_input = pending
                    pending = None
                else:
                    user_input = self._read_user_line()
                    if user_input is None or user_input.lower() in self.EXIT_COMMANDS:  # EOF / Ctrl-C at the prompt
                        self.console.note("Thank you. Goodbye! Saving history...", "info")
                        running = False
                        break

                    # Interactive slash commands are handled in place and never
                    # reach the LLM. _handle_command returns True when the input
                    # was such a command, so we just re-prompt (continue).
                    if user_input.startswith("/") and self._handle_command(user_input):
                        continue
                
                self.messages.append({"role": "user", "content": user_input})

                if self.plan_mode:
                    self.console.note(
                            "Generating plan for agent.", "info")
                    self._record_goal_request(user_input)
                    self._generate_plan_and_inject(user_input)
                    self.console.note(
                            "Plan generated.", "info")
                running_turn = True
                while running_turn:
                    self.console.note(
                            "Entering single turn loop.", "info")
                    # actually stepping through the agentic steps.
                    should_continue = self._step()
                    self._persist()
                    if should_continue:
                        self.console.note(
                            "Continuing with LLM calls until no more tool calls.", "info")
                    else:
                        self.console.note(
                            "LLM seems to have to an end of the task.", "info")

                        if self.plan_mode:
                            self.console.note(
                                "Reviewing the plan, if everything is done.", "info")
                            self._review_plan_completion() 
                            should_continue = self._step()
                            self._persist()

                        # A response with no tool calls means the turn is over. This must
                        # happen regardless of plan mode: previously the assignment sat inside
                        # `if self.plan_mode`, so non-plan sessions never left the turn loop and
                        # re-called the model forever.
                        if not should_continue:
                            running_turn = False  # escape the turn loop
            
            except _Abort:
                # Ctrl-C during a permission prompt already shut the session
                # down; just unwind the loop quietly.
                running = False

            except KeyboardInterrupt:
                self.console.note("Interrupting: Mini Code Exiting...", "error")
                self.shutdown()
                running = False
                # Ctrl-C mid-loop just saves and re-prompts rather than crashing.
            except Exception as e:
                self.console.note(f"Exception {e}. Exiting...", "error")
                self.shutdown()
                running = False
        self.console.note(
            "EOF.", "info")

    def _print_session_id(self):
        """Print the session's unique id as a minimal "Assistant" turn.

        Instead of a banner/header the id is rendered as an assistant-styled
        (green) turn carrying *only* the id in brackets, so the user can see /
        copy the id this conversation is being saved under without the noise of
        the extra "saved under" line.
        """
        self.console.turn("Assistant", f"[{self.unique_id}]", accent="green")

    def _save_last_session_pointer(self):
        """Record this session's directory in --allowed-dir/.mini.

        Writes the .mini pointer that a later --load-last-session (or the resume
        offer) reads via :func:`persistence.find_last_session_dir`. Best-effort:
        never raises, so a pointer hiccup can never break the session.

        Skipped for *background* sessions: a sub-agent's isolated child session
        must never become the user's "last session", or delegating once would
        silently redirect the user's next --load-last-session into a throwaway
        sub-agent folder.
        """
        if self.background_mode or not self.allowed_dir:
            return
        save_last_session_pointer(self.allowed_dir, self.session_dir)

    # -- autonomous drive (used by sub-agents) ------------------------------

    def _drive_auto(self, initial_task, max_steps=None):
        """Run unattended to completion and return the final assistant answer.

        Appends ``initial_task`` as a user turn, then keeps stepping until the
        model makes no further tool calls (auto-mode). ``max_steps`` caps the
        number of model calls so a misbehaving sub-agent cannot loop forever.

        The sub-agent records its work in its own cache dir: every step persists
        history / recorder / usage into the child's own session folder, and it
        seeds CURRENT_GOAL.md / CURRENT_PLAN.md (in plan mode) so each sub-session
        is self-contained and easy to inspect or resume. A final record is left
        even on a mid-run failure (the seeded goal is marked finished).
        """
        self.messages.append({"role": "user", "content": initial_task})
        # Seed the goal up front so the child keeps a self-contained record in
        # its own cache dir (a sub-agent always records a goal, independent of
        # the parent's --plan flag). We use the child's own plan_manager here so
        # nothing is written to the parent's session folder.
        from .plan import PlanManager
        child_plan = PlanManager(os.path.join(self.cache_dir, self.unique_id), enabled=True)
        child_plan.write_goal(initial_task)
        steps = 0
        # How the run ended, so the caller can explain a truncated answer
        # instead of reporting "no output" (the common case when the cap is hit
        # mid-tool-call, where there is no final assistant text at all).
        self.stop_reason = "finished"
        self.steps_taken = 0
        try:
            while True:
                should_continue = self._step()
                self._persist()
                steps += 1
                self.steps_taken = steps
                if max_steps and steps >= max_steps:
                    self.console.note(
                        f"Sub-agent reached max_steps ({max_steps}); stopping.", "warn")
                    self.stop_reason = "max_steps"
                    break
                if not should_continue:
                    break
            return self._last_assistant_text()
        finally:
            # Leave a clean, finished record in the child's cache dir even on an
            # error / early stop, so the sub-session is never left open.
            child_plan.mark_finished()
            self._persist()

    def _last_assistant_text(self):
        """Return the content of the most recent assistant (non tool-call) turn."""
        for m in reversed(self.messages):
            if m.get("role") == "assistant" and not m.get("tool_calls"):
                return m.get("content") or ""
        return ""

    # -- plan mode ---------------------------------------------------------

    def _record_goal_request(self, user_input):
        """Record the user's latest request in the plan-mode tracking files.
        """
        stripped = user_input.strip()
        if not stripped or stripped.startswith("/"):
            return
        self.plan_manager.write_goal(stripped)

    def _generate_plan_and_inject(self, current_instruction):
        """Ask the model to author a plan for the current instruction.
        """
        plan_prompt_text = prompts.plan_prompt(self.messages, current_instruction)
        plan_text = ""
        try:
            ui.console.note("Generating a plan for the current goal...", "info")
            self.console.start_thinking("planning")
            response, _, usage = utils.call_openai_server(
                [{"role": "user", "content": plan_prompt_text}],
                # The generated plan can be long: leave it unlimited (the server
                # decides the cap) rather than capping the step-by-step authoring.
                max_tokens=None,
                temperature=0.0,
                model=self.model,
                server_url=self.server_url,
                api_key=self.api_key,
                timeout=self.args.llm_timeout,
            )
            self.console.stop_thinking()

            self._record_usage(usage)
            plan_text = response or ""
        except Exception as exc:
            ui.console.note("Could not generate a plan: %s" % exc, "warn")

        self.plan_manager.write_plan(plan_text)
        if plan_text and plan_text.strip():
            self.messages.append({
                "role": "user",
                "content": "Current plan for the goal (from CURRENT_PLAN.md):\n%s" % plan_text.strip(),
            })

    def _finish_goal_and_shutdown(self):
        """Mark the active goal finished in CURRENT_PLAN.md, then quit/save.

        The goal is whatever the loop last recorded; if the loop never recorded
        one we fall back to reading the goal file back so a one-shot --prompt
        preset still marks its goal finished. A no-op in method behaviour when
        --plan is off (PlanManager no-ops), so it is safe to call unconditionally.
        """
        self.plan_manager.mark_finished(self.plan_manager.goal)
        self.shutdown()

    def _review_plan_completion(self):
        """Once the work looks done, review the plan's completion.
        """
        plan_prompt_text = prompts.plan_review_prompt(
            self.messages, self.plan_manager.goal or "")
        summary = ""
        try:
            ui.console.note("Reviewing the current plan...", "info")
            self.console.start_thinking("reviewing plan")
            response, _, usage = utils.call_openai_server(
                [{"role": "user", "content": plan_prompt_text}],
                max_tokens=1024,
                temperature=0.0,
                model=self.model,
                server_url=self.server_url,
                api_key=self.api_key,
                timeout=self.args.llm_timeout,
            )
            self.console.stop_thinking()
            self._record_usage(usage)
            summary = response or ""
        except Exception as exc:
            ui.console.note("Could not review the plan: %s" % exc, "warn")

        # Persist the review summary to the plan document and re-inject it so
        # the model works from the review result.
        self.plan_manager.write_plan_summary(summary)
        if summary and summary.strip():
            self.messages.append({
                "role": "user",
                "content": "Plan completion summary (from CURRENT_PLAN.md):\n%s"
                % summary.strip(),
            })

        return True


    def _handle_compaction(self, compacted):
        """Re-inject goal + plan into the compacted messages (plan mode).

        Thin plan-mode counterpart to the inlined plan handling inside
        :meth:`maybe_compact`, which takes a ``plan_hook`` rather than the
        owning session so it can stay a network-free module-level helper. The
        loop wires in this method only when plan mode is on; it simply
        re-injects the goal + plan via :meth:`_apply_plan_to_compacted`, which
        also rewrites the tracking files to match.
        """
        self.messages = self._apply_plan_to_compacted(compacted)

    def _apply_plan_to_compacted(self, messages):
        """Re-inject the current goal + plan after a history compaction.

        Compaction replaces the transcript with a summary, which would otherwise
        drop the goal the loop recorded and the plan it maintains. We read them
        back off disk and append them as fresh user turns so the model keeps
        working against the goal it is still trying to finish. The goal is
        appended last so it reads as the most recent instruction the model
        should act on.
        """
        plan_text = self.plan_manager.read_plan()
        goal_text = self.plan_manager.read_goal()

        # Recover the user's latest instruction from the pre-compaction tail, in
        # case the loop had not recorded a goal yet, so there is always a goal to
        # restate after compaction.
        current_input = None
        for m in reversed(messages):
            if m.get("role") == "user":
                current_input = m.get("content")
                break
        if current_input and not goal_text:
            self.plan_manager.write_goal(current_input)
            goal_text = self.plan_manager.read_goal()

        rebuilt = list(messages)
        if plan_text and plan_text.strip():
            rebuilt.append({"role": "user",
                            "content": f"Here is the current plan for the active goal (from CURRENT_PLAN.md):\n{plan_text}"})
        if goal_text and goal_text.strip():
            rebuilt.append({"role": "user",
                            "content": f"The current goal / latest user instruction (from CURRENT_GOAL.md):\n{goal_text}"})
        elif current_input:
            rebuilt.append({"role": "user", "content": current_input})
        return rebuilt

    def _is_report_file(self, name, tool_params):
        """Whether a write targets this session's own sub-agent report.md.

        A sub-agent writes its report.md under the parent's cache dir, which
        normally sits outside --allowed-dir. This exempts that single, known file
        so the sub-agent can actually write it (otherwise the sandbox would block
        the write and the parent would never receive the report). Scoped tightly:
        only when this is a child session (report_path set), only read_file /
        write_file, and only when the resolved path is exactly the report file.
        """
        if not self.report_path or name not in ("read_file", "write_file"):
            return False
        path = tool_params.get("file_path", "")
        if not path:
            return False
        try:
            resolved = os.path.realpath(path)
        except (ValueError, TypeError):
            return False
        try:
            expected = os.path.realpath(self.report_path)
        except (ValueError, TypeError):
            return False
        return resolved == expected

    # -- interactive input -------------------------------------------------

    def _read_user_line(self):
        """Read one turn from the user. Returns the text to send, or None to quit.

        An empty (blank) line is not a real instruction, so it is ignored and the
        user is asked again rather than sending an empty turn to the model.
        """
        if self.background_mode:
            # Nothing may block on input in an unattended session: report EOF so
            # any loop that reaches here unwinds instead of hanging.
            return None

        self.console.short_divider()

        user_input = self.console.read_line(style_line_prompt("You:"))

        if not user_input.strip():
            self.console.note("No input. (Type /help for commands, /quit to exit.)", "warn")
            return self._read_user_line()

        return user_input

    def _handle_command(self, user_input):
        """Handle an interactive slash command before it reaches the LLM.

        Supported:
          /history           list the high-level recorder (turn number, what
                             happened, and which history_{n}.json + index)
          /history <n>       show the full messages recorded for turn <n>
          /reset [<n>]       rewind the conversation to the end of turn <n>
                             (no <n> rewinds to the most recent turn)
          /plan              show the current goal + plan the loop maintains
                             (plan mode only)
          /clear             clear the screen (keeps the conversation)
          /help              show the command help

        Returns True when the input was a recognised command and was fully
        handled, so the caller re-prompts instead of sending it to the model.
        Unknown slash commands print a note and are swallowed (also True).
        """
        stripped = user_input.strip()
        if not stripped.startswith("/"):
            return False
        parts = stripped[1:].split(None, 1)
        cmd, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""

        if cmd == "history":
            self._inspect_history(arg) if arg else self._show_history()
            return True
        if cmd in ("reset", "rewind"):
            self._do_reset(arg)
            return True
        if cmd == "help":
            print_help(self.allowed_dir, self.server_url, self.args,
                       self.unique_id, self.cache_dir, self.model)
            return True
        if cmd in self.CLEAR_COMMANDS:
            self._clear_and_show_last()
            return True
        if cmd == "plan":
            self._show_plan()
            return True
        self.console.note(f"Unknown command {stripped!r}. Try /help.", "warn")
        return True

    def _do_reset(self, arg):
        """Rewind the conversation to the end of a recorded turn.

        No argument resets to the most recent turn (a no-op that just replays the
        current state). With <n>, rewinds to that turn, dropping the messages and
        recorder entries after it.
        """
        if not self.summary:
            self.console.note("No history has been recorded yet.", "warn")
            return
        target = self.summary[-1]
        if arg:
            try:
                idx = int(arg) - 1
                if not (0 <= idx < len(self.summary)):
                    raise ValueError
                target = self.summary[idx]
            except ValueError:
                self.console.note(
                    f"No such turn {arg!r} (have 1..{len(self.summary)}).", "error")
                return
        self._reset_history(target["turn"], target)

    def _show_plan(self):
        """Show the current plan/goal the loop maintains (``/plan`` command).

        Plan mode keeps two human-readable tracking files in the session
        directory (CURRENT_GOAL.md / CURRENT_PLAN.md) that the loop -- not the
        agent -- author. Those files live outside ``--allowed-dir``, so the
        model has no tool to read them back; the simplest way for the user to
        inspect progress is this command, which renders the goal, the current
        plan and the last completion review as one box, matching how
        ``/history`` shows the transcript. A no-op (just a note) when --plan is
        off, since the manager has nothing to read.
        """
        if not self.plan_mode:
            self.console.note("Plan mode is not enabled (--plan).", "warn")
            return
        status = self.plan_manager.read_current_status()
        if status is None:
            self.console.note("No plan information available for this session.", "warn")
            return
        self.console.rule(title="Plan", char="═", fg="cyan")
        self.console.status_box(
            status.splitlines(),
            title="Current goal + plan",
            fg="green",
            dim=True,
        )

    def _clear_and_show_last(self):
        """Handle /clear: wipe the screen and re-print the last turn.

        The conversation itself is unchanged; we just redraw the most recent
        exchange so the user re-orients after clearing the view. If there is no
        assistant reply yet, the last user turn is shown instead.
        """
        self.console.clear_screen()
        last_role = last_text = None
        for m in reversed(self.messages):
            role = m.get("role")
            content = m.get("content")
            if role == "assistant" and not m.get("tool_calls") and content:
                last_role, last_text = "Assistant", content
                break
            if role == "user" and content:
                last_role, last_text = "You", content
                break
        if last_role is not None:
            self.console.divider()
            accent = "green" if last_role == "Assistant" else "cyan"
            self.console.turn(last_role, last_text, accent=accent)

    # -- tool execution ----------------------------------------------------

    def _run_tool_calls(self, tool_calls):
        _make_tool_call_note(self.console, len(tool_calls))
        # Show a "… calling tool" spinner (in-place, non-scrolling) while the
        # tool(s) execute, mirroring the thinking spinner that covers the LLM
        # call. Always stopped (finally) so it can never leak into the next turn.
        
        try:
            for call in tool_calls:
                self._run_one_tool_call(call)
        finally:
            self.console.stop_tool_call()

    def _run_one_tool_call(self, call):

        call_id = call.get("id", "")
        name = call.get("function", {}).get("name", "")
        params_str = call.get("function", {}).get("arguments", "{}")
        self.console.start_tool_call(tool_name=name)

        try:
            tool_params = json.loads(params_str)
        except json.JSONDecodeError:
            self.console.note(f"Error parsing arguments for tool {name}", "error")
            self._append_tool_result(call_id,
                f"Error: Unknown tool: {name}, params {params_str}")
            return

        # Sub-agent: delegate the task to a sub-agent (which spins up its own
        # nested AgentSession) and feed the result back as the tool result.
        if name in self.subagents_registry:
            result = self._run_subagent(name, tool_params)
            self._append_tool_result(call_id, result)
            return

        if name not in self.tools_dict:
            self.console.note(f"Unknown tool: {name}, params {params_str}", "error")
            self._append_tool_result(call_id,
                f"Error: Unknown tool: {name}, params {params_str}")
            return

        # 1) Sandbox: keep every filesystem tool inside --allowed-dir. (In plan
        #    mode the tracking files are maintained by the loop directly -- not
        #    through agent tools -- so nothing here needs an exemption.)
        if not tool_paths_are_allowed(name, tool_params, self.allowed_dir):
            if self._is_report_file(name, tool_params):
                pass  # allow a sub-agent's own report.md under the cache dir
            else:
                self.console.note(
                    f"File operation not allowed outside allowed directory: {self.allowed_dir}", "error")
                self._append_tool_result(call_id,
                    f"Error: File operation not allowed outside allowed directory: {self.allowed_dir}")
                return

        # 2) Shell-safety classifier (DENY / ASK / ALLOW), always enforced.
        shell_cmd = None
        if name == "run_shell_command":
            shell_cmd = str(tool_params.get("command", ""))
            tier, reason = shell_safety_utils.classify_command(shell_cmd)
            if shell_safety_utils.is_deny(tier):
                self.console.note(f"Refusing to run dangerous command: {shell_cmd!r} ({reason})", "error")
                self._append_tool_result(call_id,
                    f"Command refused for safety reasons ({reason}): {shell_cmd!r}. Please choose a safer command.")
                return
        else:
            tier = reason = None

        # 3) Permission prompt. Either the user asked for prompts on every call,
        #    or the shell-safety classifier flagged this command as risky (ASK).
        if needs_permission(self.args, shell_cmd, tier):
            if self.background_mode:
                # A sub-agent has no human to ask (and forwarding the prompt to
                # the parent would hang an unattended parent). Refuse instead of
                # running: background mode never widens what the session may do.
                self.console.note(
                    f"Background session: refusing {name} (would need permission).", "warn")
                self._append_tool_result(call_id,
                    f"Refused: tool {name} requires user confirmation, which is not "
                    f"available in this unattended (background) session. Use a "
                    f"non-interactive alternative.")
                return
            ok = self._ask_permission(name, params_str, shell_cmd, tier, reason)
            if not ok:
                self._append_tool_result(call_id,
                    f"User denied permission for tool {name} and params {params_str}.")
                return

        # 4) Execute the tool. In non-verbose mode the tool call (and the usage
        #    line above) lives in the transient region: it flashes while the
        #    agent works and is cleared in place, keeping the transcript down to
        #    just the turns. In verbose mode everything is printed permanently.
        transient = not self.console.verbose
        func = self.tools_dict[name]["function"]
        result = call_function_with_timeout(func, tool_params, name, timeout_seconds=self.args.tool_timeout)

        self.console.tool_call_display(name, tool_params, result, transient=transient)
        self._append_tool_result(call_id, str(result))

    def _run_subagent(self, name, tool_params):
        """Run a sub-agent for ``task`` and return the text to feed back."""
        subagent = self.subagents_registry[name]
        task = tool_params.get("task", "")
        max_steps = tool_params.get("max_steps")
        self.console.note(f"Delegating to sub-agent '{name}' (task: {oneline(task, 80)})", "info")
        try:
            result = subagent.run(task, parent=self, max_steps=max_steps)
            self.console.note(f"Sub-agent '{name}' finished.", "ok")
            return result
        except Exception as e:
            self.console.note(f"Sub-agent '{name}' failed: {e}", "error")
            return f"Error: sub-agent '{name}' failed: {e}"

    def _ask_permission(self, name, params_str, shell_cmd, tier, reason):
        """Prompt the user to confirm a tool call (ask-permission mode).

        Delegates the actual prompting -- a styled box plus the answer input --
        to the console so all interactive input lives in one place. Returns True
        to proceed."""
        try:
            return self.console.ask_permission(name, params_str, shell_cmd, tier, reason)
        except (EOFError, KeyboardInterrupt):
            self.console.note("Thank you. Goodbye! Saving history...", "info")
            self.shutdown()
            raise _Abort

    def _append_tool_result(self, call_id, content):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": str(content)})

    def _print_assistant(self, text):
        # A delimiter line before every agent turn keeps consecutive turns
        # visually separated (the assistant may reply several times in a row
        # in auto-mode, or right after a tool-call summary).
        self.console.divider()
        self.console.turn("Assistant", text, accent="green")

    def _persist(self):
        save_history(self.messages, self.cache_dir, self.unique_id, self.history_index)
        save_summary(self.summary, self.cache_dir, self.unique_id)
        usage_path = save_usage_segment(self.token_stats, self.cache_dir,
                                        self.unique_id, self.history_index)
        total_path = save_total_usage(self.total_usage, self.cache_dir, self.unique_id)
        self.console.note(
            f"Saved usage to: {usage_path} (session total: {total_path})", "info")

    def shutdown(self):
        """Persist and print a final summary (called on quit / exit)."""
        self._persist()
        self.console.note(
            f"Saved session to {self.cache_dir}/{self.unique_id}/ "
            f"(history_0..{self.history_index}.json + summary_history.json).", "ok")
        print_usage_summary(self.total_usage)

    # -- single LLM turn ---------------------------------------------------

    def _step(self):
        """Run a single LLM turn: call the model, render its text, run its tool
        calls. Returns True if the LLM made tool calls and we're in auto-mode,
        signalling the loop to continue without asking the user."""
        # Keep long sessions within the model's context window. Compaction
        # rolls the session over to a fresh history segment (history_index + 1)
        # and returns that bump so the recorder can stay flat across it.
        segment_before = dict(self.token_stats)
        self.messages, did_compact, new_index = maybe_compact(
            self.messages, self.server_url, self.model, self.api_key,
            self.args.llm_timeout, self.token_stats, self.history_index,
            # getattr keeps argparse-like namespaces without the attribute
            # (older callers / tests) working, falling back to the default.
            max_messages=getattr(self.args, "max_messages", DEFAULT_MAX_MESSAGES),
            plan_hook=self._handle_compaction if self.plan_mode else None, console=self.console
        )
        # The compaction call itself was billed to the segment that is being
        # closed, so fold its delta into the session total here (the rollover
        # below zeroes the segment counters).
        self._bill_compaction(segment_before)

        if did_compact:
            # Close the old segment's usage file before moving on, so its spend
            # is preserved on disk even though the next segment starts at zero.
            save_usage_segment(self.token_stats, self.cache_dir, self.unique_id,
                               self.history_index)
            self.history_index = new_index
            self.token_stats = empty_usage()
            # Snapshot the compacted messages into the new segment so its turns
            # are preserved, and record the summary as the first turn of it.
            save_history(self.messages, self.cache_dir, self.unique_id, self.history_index)
            self._record_turn(
                f"summary: history compacted to {len(self.messages)} messages",
                self.messages)

        # Show a "… thinking" spinner while the LLM works (in-place, non-scrolling;
        # a no-op on non-TTY / ASCII output). Stopped right after the call returns.
        self.console.start_thinking()
        response_text, tool_calls, usage = utils.call_openai_server(
            self.messages,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            model=self.model,
            server_url=self.server_url,
            tools=self.llm_tools,
            api_key=self.api_key,
            timeout=self.args.llm_timeout,
        )
        self.console.stop_thinking()

        # A failed LLM call returns (None, None, None): treat it as "nothing
        # happened this step" so the loop degrades gracefully (records a note
        # and re-prompts) instead of crashing.
        if usage is None:
            response_text = response_text or ""
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "tool_calls": self.total_usage["tool_calls"],
            }

        self._record_usage(usage)
        made_tool_calls = bool(tool_calls)
        if made_tool_calls:
            self.token_stats["tool_calls"] += len(tool_calls)
            self.total_usage["tool_calls"] += len(tool_calls)
            # The model predicted more tool calls: once the work settles down a
            # completion review is due again (finalized_plan).
            self.finalized_plan = True

        if response_text:
            self._print_assistant(response_text)

        if tool_calls:
            self.messages.append({
                "role": "assistant",
                "content": response_text,
                "tool_calls": tool_calls,
            })
            self._run_tool_calls(tool_calls)
        else:
            self.messages.append({"role": "assistant", "content": response_text})

        if not response_text and not made_tool_calls:
            self.console.note("Warning: no response was given by the LLM.", "error")

        # Mirror the running tool-call total into this call's usage dict so the
        # per-call usage line reports the actual tool-call count (the server
        # does not, so usage.get("tool_calls") would always be 0 otherwise).
        usage["tool_calls"] = self.total_usage["tool_calls"]
        # Print the usage line LAST, transiently, so it always sits at the very
        # bottom of the screen. That way the next step can clear it in place
        # (cursor up + wipe + reprint) instead of jumping up past the assistant
        # turn above it -- which is what made the screen keep scrolling.
        self.console.usage_line(usage, transient=False)

        # Append this step's turn to the high-level recorder. On compaction the
        # recorder was already seeded for the new segment above, so this becomes
        # that segment's next turn (assistant / tool call), pointing at the
        # freshly appended message -- the end-of-turn snapshot /reset targets.
        if tool_calls:
            self._record_turn(
                f"tool call(s): {', '.join(tc.get('function', {}).get('name', '?') for tc in tool_calls)}",
                self.messages)
        elif response_text:
            self._record_turn("assistant", self.messages)

        return made_tool_calls and self.args.auto_mode


class _Abort(Exception):
    """Raised to unwind the permission prompt after a Ctrl-C / shutdown."""


# ---------------------------------------------------------------------------
# module-level helpers kept for readability / backward compatibility
# ---------------------------------------------------------------------------

def print_usage_summary(token_stats):
    ui.console.usage_summary(token_stats)


def main():
    args = parse_args()
    session = AgentSession(args)
    session.run(initial_prompt=args.prompt)


if __name__ == "__main__":
    main()
