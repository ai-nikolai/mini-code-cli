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

The helpers the test-suite imports from here are re-exported through
``__all__``, so ``from mini_code_cli.agent_loop import save_history, ...`` works
without the tests reaching into every module.
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
from . import tokens as tokens_mod
from .config import (
    DEFAULT_CACHE_LOCATION,
    DEFAULT_MODEL_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOOL_TIMEOUT,
    DEFAULT_HTTP_TIMEOUT,
    compaction_budget,
)
from .persistence import (
    save_history,
    save_summary,
    save_usage_segment,
    save_total_usage,
    load_history_with_index,
    load_summary_from_dir,
    load_usage_segment,
    load_total_usage_from_dir,
    delete_segments_above,
    history_segment_indices,
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
    # re-exported helpers (imported from here by the test-suite)
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
    "generate_unique_id",
    "find_last_session_dir",
    # public classes / entry points
    "AgentSession",
    "main",
    "parse_args",
]


# ---------------------------------------------------------------------------
# recorder helpers (shared by the class and the tests)
# ---------------------------------------------------------------------------

def _safe_int(value, default=0):
    """Coerce ``value`` to ``int``, returning ``default`` when that is impossible.

    Used by the *display / navigation* paths, which read a recorder that was
    already validated at restore time but may have been hand-edited since: a
    ``/history`` listing must render a junk field rather than crash the CLI.
    A restore itself never comes through here -- it validates strictly instead
    (see ``AgentSession._validate_records``).
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Agent loop for interacting with a language model")
    # MODEL / LLM related
    parser.add_argument("--url", type=str, default=f"{DEFAULT_MODEL_URL}", help=f"API host, default={DEFAULT_MODEL_URL}")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for a single LLM response (the reply budget), default=None. Also used as the output headroom reserved when deriving the compaction limit from the model's context window.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature for LLM, default={DEFAULT_TEMPERATURE}")
    parser.add_argument("--model", type=str, help="Model name, default=None")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication. If not set, it tries to find it in env variable: MINI_CODE_API_KEY.")

    # Other settings:
    parser.add_argument("--cache-dir", type=str, help=f"The cache dir for the session histories. default={DEFAULT_CACHE_LOCATION}")
    parser.add_argument("--restore-session", type=str, help="A previous session to continue: its directory (e.g. ~/.cache/mini/sessions/1792028090823027-a9d77653/) or a bare session id under --cache-dir. A session folder must hold a summary_history.json recorder plus the history_{n}.json segment it names; anything else starts a fresh session.")
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
    parser.add_argument("--skills", dest="enable_skills", action="store_true", help="Enable the skills feature: a local .mini directory (inside --allowed-dir) of skill.md files, skill sub-dirs, a high-level SKILLS.md summary, and executable Python scripts, exposed to the agent as list_skills/read_skill/create_skill/run_skill tools. default= False")
    parser.add_argument("--llm-timeout", type=int, default=DEFAULT_HTTP_TIMEOUT, help=f"HTTP timeout in seconds for LLM requests (reasoning models can be slow). default={DEFAULT_HTTP_TIMEOUT}")
    parser.add_argument("--tool-timeout", type=int, default=DEFAULT_TOOL_TIMEOUT, help=f"Timeout in seconds for a single tool call. default={DEFAULT_TOOL_TIMEOUT}")
    parser.add_argument("--max-context-tokens", type=int, default=None, help="Input-token budget before the conversation is compacted into a summary. This is the primary compaction trigger: it is compared against the projected size of the next request (the server-reported prompt tokens, plus an estimate for whatever was appended since) plus the output reserve (--max-tokens, or a default). Unset (default): derived from the model's context window minus the output reserve and a safety margin, falling back to a conservative window for unknown models. Lower it to compact sooner.")
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
        args.enable_skills = True

    if args.all_last:
        args.load_last_session = True

    return args


# ---------------------------------------------------------------------------
# banners / helpers
# ---------------------------------------------------------------------------

def print_help(args):
    """Print the interactive command list (``/help``)."""
    ui.console.rule("help", fg="cyan")
    if args.auto_mode:
        help_message = ("Auto-mode: the agent keeps calling the LLM until no "
                        "further tool calls are possible.")
    else:
        help_message = "Manual-mode: the agent asks for your input after each LLM prediction."
    ui.console.emit(ui.console.style(help_message, dim=True, fg="cyan"))
    ui.console.emit(ui.console.style("History / navigation commands:", bold=True, fg="cyan"))
    for command, description in (
        ("/history", "list the conversation (turn number, what happened, where)"),
        ("/history <n>", "show the full messages recorded for turn n"),
        ("/history last", "...for the most recent turn"),
        ("/plan", "show the current goal + plan (plan mode only)"),
        ("/reset [<n>|last]", "rewind to the end of turn n (drops everything after it);"),
        ("", "a bare /reset only re-saves the current state (no data loss)"),
        ("/clear", "clear the screen (keeps the conversation history)"),
        ("/help", "show this help"),
        ("/quit, /exit", "save the session and exit"),
    ):
        ui.console.emit(f"  {command:16} {description}")
    if getattr(args, "enable_subagents", False) or args.verbose:
        ui.console.emit(ui.console.style("Sub-agents:", bold=True, fg="cyan"))
        for sa in subagents.get_registry().values():
            ui.console.emit(f"  {sa.name:16} {sa.description}")
    if getattr(args, "enable_skills", False) or args.verbose:
        ui.console.emit(ui.console.style("Skills:", bold=True, fg="cyan"))
        for name in skills.SKILL_TOOL_NAMES:
            ui.console.emit(f"  {name:16} (from the local .mini directory)")
    ui.console.rule(char="═", fg="cyan")


def print_welcome(allowed_dir, server_url, args, unique_id, cache_dir, model_name=None,
                  console=None):
    """Render the startup info banner through the UI.

    Everything the session was configured with (where it saves, which server and
    model it talks to, and -- most importantly -- whether tools run unattended)
    is summarised in one box before the first prompt, so a surprising setting is
    visible immediately instead of only being discoverable via /help.
    """
    from . import __version__ as version

    if getattr(args, "sudo", False):
        permission_mode = "SUDO MODE: no permission prompts (DENY commands still blocked)."
    elif getattr(args, "ask_permission", False):
        permission_mode = "All tool calls require permission."
    else:
        permission_mode = "Tool calls execute without asking."
    auto_mode = bool(getattr(args, "auto_mode", False))
    agent_mode = "Agent is in auto-mode." if auto_mode else "Agent is in manual-mode."
    shell_mode = "Shell disabled." if not getattr(args, "enable_shell", False) else "Shell enabled."

    (console or ui.console).welcome(version, unique_id, cache_dir, allowed_dir,
                                    server_url, model_name, permission_mode,
                                    agent_mode, shell_mode)


# ---------------------------------------------------------------------------
# history compaction
# ---------------------------------------------------------------------------

def _compaction_trigger(messages, token_limit, reserved_output_tokens, current_tokens):
    """Decide whether the history is too big, and say why.

    Token pressure is the *only* trigger: an over-full context window makes the
    next request fail outright, and it is the number of tokens -- not the number
    of turns -- that decides that. A handful of large tool outputs can fill a
    window that a hundred short exchanges would not, so counting messages is a
    poor proxy for it (and compacts perfectly healthy conversations).

    The projection adds ``reserved_output_tokens`` to the current input size,
    because the reply has to fit in the same window: a request that leaves no
    room to answer is already too big.

    Returns ``(triggered, reason)``; ``reason`` is a short human-readable string
    for the console note when it fires.
    """
    if token_limit and int(token_limit) > 0:
        limit = int(token_limit)
        reserve = max(int(reserved_output_tokens or 0), 0)
        tokens = current_tokens
        if tokens is None:
            tokens = tokens_mod.estimate_messages_tokens(messages)
        projected = max(int(tokens), 0) + reserve
        if projected > limit:
            return True, (f"{projected} tokens in, {reserve} reserved for the reply "
                          f"> {limit} budget")
    return False, ""


def _in_flight_tail(messages):
    """Return the messages of the turn that is still in flight, or ``[]``.

    Compaction rewrites the transcript, so whatever it drops is gone from the
    conversation *and* from everything saved afterwards. The last turn is not
    finished work, though: it is either the instruction being answered right now
    (the transcript ends with a ``user`` message) or the tool exchange the model
    is waiting on (it ends with a ``tool`` result). Either way it has to be
    carried across verbatim, or the model answers the summary instead of the
    user's request -- and the typed line never reaches the recorder, the new
    segment or a later restore.

    The suffix starts at the last ``user`` message, which always begins a
    self-contained turn: every ``assistant`` tool call inside it is followed by
    its ``tool`` results, so the slice is a valid transcript on its own.

    ``[]`` when the transcript already ends with an ``assistant`` reply: that
    turn is answered, and re-sending it would ask the model to answer it twice.
    """
    if not messages:
        return []
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") == "assistant":
        return []
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            return list(messages[index:])
    return []


def _plural(count, noun, suffix="s"):
    """``"1 message"`` / ``"4 messages"`` -- for messages the user reads."""
    return f"{count} {noun}" + ("" if count == 1 else suffix)


def maybe_compact(messages, server_url, model, api_key, llm_timeout, token_stats,
                  history_index=0, token_limit=None, reserved_output_tokens=0,
                  current_tokens=None, plan_hook=None, console=None):
    """Compact the conversation history once it no longer fits the context window.

    Long (especially auto-mode) sessions can overflow the model's context window.
    The trigger is the *projected input size* of the next request -- the estimate
    for ``messages`` (or ``current_tokens``, a better number the caller already
    has) plus ``reserved_output_tokens`` headroom for the reply -- compared
    against ``token_limit``. Nothing else triggers compaction: in particular the
    number of messages is not a criterion. Compaction is then done by asking the
    LLM to summarize the history and replacing the old turns with that summary,
    keeping the original system prompt as the first message *and the turn that is
    still in flight* (see :func:`_in_flight_tail`) verbatim after it -- a summary
    of the user's current request is not the request. The compaction call
    itself is a single, tool-free request, so it cannot trigger further tool
    calls or recurse into compaction.

    ``token_limit=None`` disables the guard entirely (a no-op, useful to callers
    that only sometimes enable compaction). See :func:`_compaction_trigger`.

    ``plan_hook`` is an optional callable the caller supplies only when running
    under plan mode (in practice ``AgentSession._handle_compaction``). After the
    summary is built the hook re-injects the running goal + plan so the compacted
    messages carry them forward. It receives the compacted list *and* the
    in-flight tail (see below) and must update the compacted list in place,
    because that list is what gets returned to the caller. Passing it in --
    instead of referencing the owning session -- keeps ``maybe_compact`` a
    small, module-level, network-free helper that the test-suite can import and
    exercise directly.
    """
    # ``console`` defaults to None so callers may omit it, but every branch below
    # reports through a console. Falling back to the shared one keeps the
    # documented signature usable -- without this a bare maybe_compact() call hit
    # "'NoneType' object has no attribute 'start_thinking'" inside the try block,
    # which the except then swallowed as a failed compaction.
    if console is None:
        console = ui.console

    triggered, reason = _compaction_trigger(
        messages, token_limit, reserved_output_tokens, current_tokens)
    if not triggered:
        # No compaction: keep the same segment (history_index) and signal the
        # caller with (messages, did_compact=False, unchanged_index). Returning
        # the full triple here is required -- callers unpack all three, so a
        # bare messages would crash them.
        return messages, False, history_index

    # The turn this step is answering -- the instruction just typed, or the tool
    # exchange the model is waiting on -- is carried across the rollover (see
    # :func:`_in_flight_tail`). It is computed *before* the slow summary call so
    # the console can tell the user right away that nothing they just sent is at
    # stake; a compaction note that stays silent about that is exactly what makes
    # the rollover look like it might have eaten the message.
    tail = _in_flight_tail(messages)
    kept_note = (f" The current turn ({_plural(len(tail), 'message')}) is kept "
                 f"verbatim." if tail else "")

    try:
        console.note(
            f"Compacting the conversation: {_plural(len(messages), 'message')} "
            f"-> a summary ({reason}).{kept_note}", "warn")
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
            console.start_thinking("compacting history")
            try:
                reply, _, compact_usage, _ = utils.unpack_llm_result(
                    utils.call_openai_server(
                        [{"role": "user", "content": summary_prompt}],
                        max_tokens=1024,
                        temperature=0.0,
                        model=model,
                        server_url=server_url,
                        api_key=api_key,
                        timeout=llm_timeout,
                    ))
            finally:
                console.stop_thinking()

            accumulate_usage(token_stats, compact_usage)
            summary = (reply or "").strip()
            if summary:
                break
            if attempt == 1:
                console.note("The summariser returned an empty summary; retrying once.", "warn")

        if not summary:
            # Say plainly what did *not* happen: nothing was dropped and the
            # segment did not move -- only the shrink did, so the next request
            # may still be too large.
            console.note(
                f"Could not compact history: the summariser returned an empty "
                f"summary twice. Keeping all {_plural(len(messages), 'message')} "
                f"and history_{history_index}.json as they are -- nothing was "
                f"lost, but the next request may still be too large "
                f"({reason}).", "warn")
            return messages, False, history_index

        system_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
        compacted = []
        if system_content:
            compacted.append({"role": "system", "content": system_content})
        compacted.append({
            "role": "user",
            "content": f"Here is a summary of our earlier conversation so far:\n{summary}",
        })

        # The turn in flight has to survive the rollover. Compaction runs at the
        # *start* of a step, so ``messages`` normally ends with the user message
        # this step is meant to answer, or with the tool result the model is
        # waiting on. Replacing the transcript with just (system, summary)
        # therefore dropped the user's instruction on the floor: the model
        # answered the summary -- "Understood. What would you like me to do?" --
        # and the typed line never reached the new segment, the recorder or a
        # later restore (it survived only inside the *closed* segment, which no
        # recorder record points at any more). ``tail`` was computed before the
        # summary call so the opening note could already promise this.

        # In plan mode, re-inject the running plan + current goal so a
        # compaction (which would otherwise drop them) keeps the model
        # working against the goal it is still trying to finish. The hook is
        # wired in only when the caller runs under plan mode; otherwise
        # compaction degrades to a plain summary. It is also handed the tail so
        # it can restate the *real* instruction rather than the summary text.
        if plan_hook is not None:
            plan_hook(compacted, tail)

        # ... and only then the live turn, so the user's actual instruction
        # reads as the most recent thing the model has to act on.
        compacted.extend(tail)

        # What the user is shown has to answer the three questions a rollover
        # raises: *what* was folded away (the summary itself), *how much* (the
        # closing segment's message count, and what the new one starts with),
        # and *where it went* (which segment file was closed and which was
        # started).
        console.note(f"Summary preview: {utils.oneline(summary, 120)}", "warn")
        kept = (f"summary + {_plural(len(tail), 'kept message')}" if tail
                else "summary only")
        # Phrased as "old segment -> new segment" rather than "N messages -> M
        # messages": the replacement can be *longer* than what it replaced (a
        # system prompt, the summary and the kept turn), and a bare pair of
        # counts that sometimes grows reads like a bug.
        console.note(
            f"Compacted history_{history_index}.json "
            f"({_plural(len(messages), 'message')}) into a summary; "
            f"history_{history_index + 1}.json now starts with "
            f"{_plural(len(compacted), 'message')} ({kept}).", "warn")
        # A new segment (history_index + 1) is started on compaction: the
        # compacted messages are its snapshot, and every later turn appends
        # to it. The caller records a single high-level 'summary' turn for
        # this index so the recorder stays flat across the rollover.
        return compacted, True, history_index + 1
    except Exception as e:
        console.note(
            f"Failed to compact history: {e}. Keeping all "
            f"{_plural(len(messages), 'message')} unchanged.", "error")

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

        # Mint the session id eagerly: the startup banner is printed *before* the
        # resume offer is answered, so it must be able to name the session that
        # early. start_new_session() reuses this id; a successful restore
        # overwrites it, which is why the "[<id>]" line (printed after
        # start_session) can differ from the id in the banner.
        self.unique_id = generate_unique_id()
        self.session_dir = os.path.join(self.cache_dir, self.unique_id)

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

        # The good-bye box is printed once, from shutdown(). Several exits
        # funnel through there (quit, exit, EOF, a /quit command) and a repeated
        # box would read like a second session had ended.
        self._farewell_shown = False

        # Token accounting. ``token_stats`` covers the *current history
        # segment* and is reset when compaction rolls over to a new one;
        # ``total_usage`` is the whole-session accumulator and never decreases.
        # Keeping the two apart is what lets usage survive a compaction.
        self.token_stats = empty_usage()
        self.total_usage = empty_usage()

        # Compaction budget. History is compacted when the *projected input size*
        # of the next request would not leave room for the reply inside the
        # model's context window -- not when the message count crosses a
        # threshold, because it is tokens that make a request fail. Resolving it
        # once, here, means every step (and the console) agrees on one number.
        self.compaction_budget = compaction_budget(
            model=self.model,
            explicit_limit=getattr(args, "max_context_tokens", None),
            max_output_tokens=getattr(args, "max_tokens", None),
        )
        # How large the last request's prompt was, per the server's own count,
        # and how many messages that request held. Together these let
        # project_input_tokens() track the growing transcript instead of
        # re-estimating from characters every step (see _note_prompt_size).
        self.last_prompt_tokens = 0
        self.last_prompt_message_count = 0

        self.background_mode = background_mode

        self.restore_session = getattr(self.args, "restore_session", None) #specific session
        # --load-last-session is an explicit *instruction* to continue the session
        # recorded in --allowed-dir/.mini, so it resumes without the interactive
        # offer (a piped/non-TTY stdin would otherwise answer that offer with EOF
        # and exit without doing anything). --restore-session still wins.
        self.load_last_session = bool(getattr(self.args, "load_last_session", False))

    def start_session(self):
        """Restore a previous session when there is one, else start fresh.

        A *background* session (a sub-agent's, see ``background_mode``) short
        circuits the detection entirely: it neither consults the ``.mini`` resume
        pointer nor offers the interactive resume prompt, so it always begins a
        fresh isolated session. That is what makes delegation safe to run
        unattended -- there is a single place that decides this.

        All three ways in (``--restore-session``, ``--load-last-session`` and the
        interactive offer) go through :meth:`_try_restore`, which reports *why* a
        restore was refused and then leaves the session clean for
        :meth:`start_new_session`. A refused restore therefore never silently
        reuses the old folder.
        """
        if self.background_mode and not self.restore_session:
            self.start_new_session()
            return

        if self.restore_session:
            if self._try_restore(resolve_session_dir(self.restore_session, self.cache_dir)):
                return
        elif not self.background_mode:
            # Resolving or offering the resume must never stop the CLI from
            # starting: any hiccup there is reported and we start fresh.
            try:
                detected = find_last_session_dir(self.allowed_dir)
                if detected and self._resume_previous(detected):
                    return
            except Exception as e:
                self.console.note(f"Could not resume the previous session: {e}", "warn")
        self.start_new_session()

    def _try_restore(self, restore_directory):
        """Restore a session in place, returning whether it worked.

        A failed restore is not fatal: the caller falls back to a fresh session,
        so a folder this version cannot read (or a typo'd path) never blocks the
        CLI from starting. It is not silent either -- the reason is noted by
        :meth:`restore_previous_session` and the fallback is announced here,
        because a user who asked to continue a conversation needs to know that
        they are not looking at it.
        """
        try:
            if self.restore_previous_session(restore_directory):
                return True
        except Exception as e:
            self.console.note(
                f"Could not restore {restore_directory}: {e}", "error")
            self._fail_restore()
        self.console.note(
            f"Starting a fresh session instead of restoring {restore_directory}.",
            "warn")
        return False

    def _resume_previous(self, detected):
        """Continue the detected session if the user wants to; True if restored.

        ``--load-last-session`` makes that explicit, so it continues without
        asking (a piped/non-TTY stdin would otherwise answer the offer with EOF
        and exit without doing anything). Otherwise the interactive offer
        decides, and "exit" leaves the CLI without starting a session.
        """
        if self.load_last_session:
            self.console.note("Continuing the last session (--load-last-session).", "warn")
            return self._try_restore(resolve_session_dir(detected, self.cache_dir))

        answer = self.show_resume_choice()
        if answer == "exit":
            self.console.note("User chose to exit! Exiting.", "warn")
            raise SystemExit("")
        if answer != "resume":
            self.console.note("Will start with new session!", "warn")
            return False
        self.console.note("Resuming the session.", "warn")
        return self._try_restore(resolve_session_dir(detected, self.cache_dir))

    def start_new_session(self):
        """Start a fresh session under ``self.unique_id``.

        The id is normally minted in ``__init__`` (see there) and simply reused
        here; it is re-minted only when it is missing, which is what happens
        after a failed restore has cleared it.

        The complete on-disk set is written straight away -- the segment, the
        recorder and the (all-zero) usage files -- so a session this version
        creates always satisfies the restore contract even if the process dies
        before the first quit. A restorable session needs the recorder and the
        segment it names; the usage files are optional, and writing them here
        keeps the folder self-describing rather than half-populated.
        """
        self.unique_id = self.unique_id or generate_unique_id()
        self.session_dir = os.path.join(self.cache_dir, self.unique_id)

        self.plan_manager = plan_utils.PlanManager(
            self.session_dir, enabled=self.plan_mode)

        self.history_index = 0
        self.summary = []

        self.messages = self._build_initial_messages()

        save_history(self.messages, self.cache_dir, self.unique_id, self.history_index)
        save_summary(self.summary, self.cache_dir, self.unique_id)
        save_usage_segment(self.token_stats, self.cache_dir, self.unique_id,
                           self.history_index)
        save_total_usage(self.total_usage, self.cache_dir, self.unique_id)

        # Record this session in --allowed-dir/.mini so a later
        # --load-last-session (or the interactive resume offer) can find it.
        self._save_last_session_pointer()


    def restore_previous_session(self, restore_directory):
        """Restore a session from ``restore_directory``, strictly, or refuse it.

        A restorable folder is exactly what this version writes: a
        ``summary_history.json`` recorder whose records name where each turn
        lives, plus a readable ``history_{n}.json`` for the segment the *last*
        record points at. Nothing else is accepted, and each refusal says which
        part was wrong:

          * the directory does not exist              -> refuse;
          * no recorder (or not a JSON list)          -> refuse;
          * a recorder that does not validate         -> refuse (``_validate_records``);
          * the named segment missing/empty/garbled   -> refuse.

        There is deliberately no "rebuild the recorder from the transcript" path
        and no single-file ``history.json`` fallback: both had to *invent* turn
        boundaries, which made a rebuilt session behave differently under
        ``/reset`` than a recorded one -- and only sessions this version wrote
        are being restored.

        Order matters, and it is recorder-first. The recorder says which segment
        the user was really looking at, and the messages are loaded from *that*
        one, so a ``/reset`` that rewound onto an older segment is not undone by
        a stale higher segment left on disk. The loaded index becomes
        ``history_index`` -- the segment later saves append to -- and only then
        are the usage totals restored (they are looked up by that index).

        Failure is atomic: :meth:`_fail_restore` clears everything a restore
        touches, so the caller's fallback is a clean fresh session that never
        half-adopts this folder or reuses its id.
        """
        prefix = f"Cannot restore {restore_directory}"
        if not restore_directory or not os.path.isdir(restore_directory):
            self.console.note(f"{prefix}: no such session directory.", "error")
            return self._fail_restore()

        saved_summary = load_summary_from_dir(restore_directory)
        if saved_summary is None:
            self.console.note(
                f"{prefix}: it has no readable summary_history.json "
                "(the recorder every restorable session has).", "error")
            return self._fail_restore()
        try:
            records = self._validate_records(saved_summary)
        except ValueError as e:
            self.console.note(f"{prefix}: {e}.", "error")
            return self._fail_restore()

        # A recorder with no turns is a session that quit before finishing one:
        # it is restorable, on the segment it started on.
        preferred = records[-1]["history_file"] if records else 0
        restored, loaded_index = load_history_with_index(restore_directory, preferred)
        if restored is None:
            self.console.note(
                f"{prefix}: history_{preferred}.json is missing, empty or "
                "unreadable.", "error")
            return self._fail_restore()

        try:
            self.unique_id = os.path.basename(os.path.normpath(restore_directory))
            self.session_dir = restore_directory
            self.plan_manager = plan_utils.PlanManager(
                self.session_dir, enabled=self.plan_mode)
            self.messages = restored
            self.summary = records
            self.history_index = int(loaded_index)
            # No prompt has been sent in *this* process yet, so any remembered
            # size would be stale: start from a plain estimate of the restored
            # transcript (which is what project_input_tokens falls back to).
            self.last_prompt_tokens = 0
            self.last_prompt_message_count = 0
            self._restore_usage_if_available(restore_directory)

            # Continuing a session in place makes it the latest one again: keep
            # the pointer aimed at it so a later --load-last-session finds the
            # conversation the user is actually working in.
            self._save_last_session_pointer()
            return True
        except Exception as e:
            self.console.note(f"{prefix}: {e}", "error")
            return self._fail_restore()

    def _fail_restore(self):
        """Clear every field a restore may have touched; always returns ``False``.

        A restore is all-or-nothing. Dropping ``unique_id`` is what guarantees
        the fallback really is a *fresh* session: ``start_new_session`` re-mints
        the id when it is missing, so a refused restore can never append to (or
        overwrite) the folder it just declined to adopt.
        """
        self.unique_id = None
        self.session_dir = None
        self.plan_manager = None
        self.messages = []
        self.summary = []
        self.history_index = 0
        self.token_stats = empty_usage()
        self.total_usage = empty_usage()
        self.last_prompt_tokens = 0
        self.last_prompt_message_count = 0
        return False

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
        ``total_usage.json`` (reconstructed from the segment files when it is
        missing), so the running totals stay continuous across a restore *and*
        across the compaction that may have happened before it. Usage is
        optional -- a folder that has no usage files at all still restores, with
        zeros -- and any problem there leaves the totals at zero rather than
        failing the restore.
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

    def _note_prompt_size(self, usage, message_count=None):
        """Remember how large the last request's prompt was.

        The server's reported ``prompt_tokens`` is exact, so it is worth far more
        than a characters/4 guess -- but it only describes the transcript as it
        was at that moment. Recording the message count alongside it is what lets
        :func:`tokens.project_input_tokens` keep using that number later in the
        turn: it adds an estimate for the messages appended since, instead of
        throwing the real count away and re-guessing the whole transcript.

        ``message_count`` defaults to the current transcript length. A call that
        reports nothing (many local servers) or fails leaves the previous
        estimate in place, which then simply ages into the heuristic path.
        """
        if not usage:
            return
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        if prompt_tokens <= 0:
            return
        if message_count is None:
            message_count = len(self.messages)
        self.last_prompt_tokens = prompt_tokens
        self.last_prompt_message_count = int(message_count)

    def _projected_input_tokens(self):
        """Best estimate of how large the next request's prompt will be."""
        return tokens_mod.project_input_tokens(
            self.messages, self.last_prompt_tokens, self.last_prompt_message_count)

    def _context_fraction(self):
        """How much of the model's context window the next request would fill.

        Returns a 0.0-1.0 fraction (clamped), or ``None`` when it cannot be
        computed. It divides the *same* projection the compaction trigger uses
        against the same window, so the bar the user sees and the moment
        compaction fires can never disagree -- a bar stuck at 20% while the
        transcript is compacted would be worse than no bar at all.

        Note this counts input only, i.e. it answers "how full is the window
        with what we are about to send", not "including the reply that is about
        to be written"; that reserve is what the compaction limit already
        accounts for.
        """
        window = self.compaction_budget.context_window
        if not window or window <= 0:
            return None
        try:
            used = self._projected_input_tokens()
        except Exception:
            # A display detail must never be able to break a turn.
            return None
        return min(max(used / float(window), 0.0), 1.0)

    def _context_badge(self):
        """The context-occupancy badge for the Assistant header ("" if unusable).

        The percentage is occupancy of the whole window (what the user asked
        for: "how much of the context is in use"), but the colour thresholds are
        derived from the compaction limit, so amber/red mean "compaction is
        near" -- i.e. they point at the point where something actually happens.
        With a fixed 60/85% of the window the red zone would be unreachable: the
        budget keeps a reserve plus a margin, so history is compacted at roughly
        84% of the window and never gets near 100%.
        """
        fraction = self._context_fraction()
        if fraction is None:
            return ""
        window = self.compaction_budget.context_window
        danger = min(max(self.compaction_budget.token_limit / float(window), 0.0), 1.0)
        return self.console.context_bar(fraction, warn=danger * 0.85, danger=danger)

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

    @staticmethod
    def _validate_records(saved):
        """Validate a loaded recorder strictly; return a clean copy of its records.

        A ``summary_history.json`` is the *contract* the navigation layer runs
        on: ``/history <n>`` and ``/reset <n>`` index into the list positionally
        and every record says exactly where its turn lives on disk
        (``history_{n}.json`` at message ``index``). A malformed record
        therefore cannot be repaired without inventing a location, so this
        raises ``ValueError`` naming the offending record instead of normalising
        it -- the restore is then refused and the user starts a fresh session,
        which is honest. (The old lenient pass silently dropped junk entries,
        clamped negatives at 0 and renumbered turns ``1..N``; one dropped record
        shifted every later turn number and made a rewind land in the wrong
        place while still reporting success.)

        Requirements: a JSON list of objects, each with integer ``turn`` /
        ``history_file`` / ``index``, no negative ``history_file`` or ``index``,
        and ``turn == position + 1`` (turns run contiguously from 1). An empty
        list is valid -- a session that quit before finishing a turn.
        """
        if not isinstance(saved, list):
            raise ValueError("summary_history.json does not hold a list of turns")
        records = []
        for position, entry in enumerate(saved, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"record {position} is not a JSON object")
            for key in ("turn", "history_file", "index"):
                value = entry.get(key)
                if not isinstance(value, int) or isinstance(value, bool):
                    raise ValueError(
                        f"record {position} has a non-integer {key} ({value!r})")
            if entry["turn"] != position:
                raise ValueError(
                    f"record {position} is numbered turn {entry['turn']} "
                    f"(turns must run 1..{len(saved)})")
            if entry["history_file"] < 0 or entry["index"] < 0:
                raise ValueError(
                    f"record {position} has a negative history_file/index")
            records.append({
                "turn": entry["turn"],
                "what": str(entry.get("what") or "?"),
                "history_file": entry["history_file"],
                "index": entry["index"],
            })
        return records

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
        if not self.summary:
            self.console.note("No history has been recorded yet.", "warn")
            return
        rows = [(str(_safe_int(r.get("turn", i + 1), i + 1)),
                 str(r.get("what") or "?"),
                 f"history_{_safe_int(r.get('history_file'), 0)}.json"
                 f"#{_safe_int(r.get('index'), 0)}")
                for i, r in enumerate(self.summary) if isinstance(r, dict)]
        self.console.history_list(rows)

    def _resolve_turn_record(self, arg):
        """Resolve a turn selector to ``(position, record)``, or ``None``.

        One rule for both ``/history <n>`` and ``/reset <n>``: accept ``last``
        (case-insensitive) or a **positive** 1-based integer, and reject
        everything else with the same message -- ``0``, negatives, non-integers
        and out-of-range values. Previously ``/history`` computed
        ``summary[int(arg) - 1]`` without a lower bound, so ``/history 0``
        silently showed the *last* turn and ``/history -2`` the third-to-last,
        while ``/reset`` rejected exactly those inputs: the same user mistake got
        two different (and one silently wrong) answers.
        """
        count = len(self.summary)
        if not count:
            self.console.note("No history has been recorded yet.", "warn")
            return None
        text = (arg or "").strip().lower()
        if text in ("last", ""):
            return count - 1, self.summary[-1]
        try:
            position = int(text) - 1
        except (TypeError, ValueError):
            position = -1
        if not 0 <= position < count:
            self.console.note(
                f"No such turn {arg!r} (have 1..{count}, or 'last').", "error")
            return None
        return position, self.summary[position]

    def _inspect_history(self, arg):
        """Print the full messages recorded for a single turn."""
        resolved = self._resolve_turn_record(arg)
        if resolved is None:
            return
        _, record = resolved
        messages = self._segment_upto(record)
        if messages is None:
            return
        self.console.history_detail(record, messages, total=len(self.summary))

    def _load_segment(self, history_index):
        """Load ``history_{history_index}.json`` as a message list, or ``None``.

        Never raises and never trusts the argument: an out-of-range, negative or
        non-integer index, a missing file, unreadable JSON or a payload that is
        not a non-empty list all yield ``None``. Navigation and the rewind only
        ever *read* through here, so a damaged segment can never crash the CLI.
        """
        index = _safe_int(history_index, -1)
        if index < 0 or not self.session_dir:
            return None
        path = os.path.join(self.session_dir, f"history_{index}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                segment = json.load(f)
        except Exception:
            return None
        if not isinstance(segment, list) or not segment:
            return None
        return [m for m in segment if isinstance(m, dict)]

    def _segment_for(self, history_index):
        """Return the messages of a segment, preferring the live transcript.

        ``_persist()`` only runs on quit (and at a compaction rollover), so the
        ``history_{n}.json`` file on disk can lag arbitrarily far behind the
        running conversation. For the *current* segment the in-memory transcript
        is therefore the authoritative copy -- reading the file instead used to
        silently drop everything said since the last save (a ``/reset`` could
        truncate to a state older than the turn it was asked to keep). Older
        segments are immutable within their lifetime, so they are read from disk.
        """
        index = _safe_int(history_index, -1)
        if index < 0:
            return None
        if index == self.history_index:
            live = [m for m in self.messages if isinstance(m, dict)]
            if live:
                return live
        return self._load_segment(index)

    def _segment_upto(self, record):
        """Load a recorded turn's segment and truncate it at its index.

        Returns the messages reproducing the conversation state at the end of
        that turn, or None if the segment could not be loaded (never raises: a
        missing or moved segment must not crash navigation). Both recorded
        numbers are coerced and the index is clamped into the segment, so a
        stale or hand-edited recorder entry cannot slice outside it.
        """
        segment = self._segment_for(record.get("history_file"))
        if not segment:
            self.console.note("Could not load the history segment for that turn.", "error")
            return None
        index = max(0, min(_safe_int(record.get("index"), 0), len(segment) - 1))
        return segment[:index + 1]

    def _reset_history(self, keep_count, record, force=True):
        """Rewind the conversation to the end of a recorded turn.

        ``keep_count`` is how many recorder entries to keep (the resolved
        position + 1 -- *not* ``record["turn"]``, which is display data and used
        to be trusted for slicing). The work is ordered so a failure cannot leave
        the live session remembering a state the disk did not accept:

          1. load the target segment and compute the truncated transcript;
          2. refuse (with a warning) when ``force`` is off and messages after the
             recorded turn exist -- an unrecorded tail must not be silently
             destroyed;
          3. persist the truncated ``history_{n}.json``, ``summary_history.json``
             and ``total_usage.json`` *first*, rolling the segment/summary back
             if either write fails;
          4. delete the now-orphaned higher segments (best-effort: a failure is
             a warning, not a rollback -- the rewind is already on disk);
          5. only then mutate the live session.

        Afterwards the session is really on segment ``n``: ``token_stats`` is
        reloaded from ``usage_{n}.json`` (so the next ``_persist()`` cannot write
        the later segment's counters over it), the remembered prompt size is
        dropped (it describes a transcript that no longer exists) and
        ``total_usage`` is left alone -- a rewind does not un-spend tokens.
        """
        target_index = _safe_int(record.get("history_file"), -1)
        full_segment = self._segment_for(target_index)
        if full_segment is None:
            self.console.note(
                "Could not load the history segment for that turn; "
                "the session is unchanged.", "error")
            return False

        index = max(0, min(_safe_int(record.get("index"), 0), len(full_segment) - 1))
        messages = full_segment[:index + 1]
        trailing = max(len(full_segment) - len(messages), 0)
        if not force and trailing:
            self.console.note(
                f"{trailing} message(s) after turn {keep_count} were never recorded; "
                f"refusing to discard them. Use /reset {keep_count} to rewind anyway.",
                "warn")
            return False

        new_summary = [r for r in self.summary[:keep_count] if isinstance(r, dict)]
        try:
            save_history(messages, self.cache_dir, self.unique_id, target_index)
            save_summary(new_summary, self.cache_dir, self.unique_id)
            save_total_usage(self.total_usage, self.cache_dir, self.unique_id)
        except Exception as e:
            # Put the untouched segment and recorder back, so the on-disk state
            # still matches the (unchanged) in-memory session.
            try:
                save_history(full_segment, self.cache_dir, self.unique_id, target_index)
                save_summary(self.summary, self.cache_dir, self.unique_id)
            except Exception:
                pass
            self.console.note(
                f"Could not persist the rewind ({e}); the session is unchanged.", "error")
            return False

        removed = delete_segments_above(self.session_dir, target_index)

        self.messages = messages
        self.history_index = target_index
        self.summary = new_summary
        self.token_stats = load_usage_segment(self.session_dir, target_index)
        self.last_prompt_tokens = 0
        self.last_prompt_message_count = 0

        self.console.note(
            f"Reset to turn {keep_count}: kept {len(self.messages)} messages "
            f"(history_{target_index}.json up to index {index}).", "ok")
        if removed:
            # Reported at "ok" level, not "info": info notes are hidden in the
            # default (non-verbose) view, which is exactly where the user needs to
            # know that files were deleted.
            self.console.note(
                f"Removed {len(removed)} later history/usage segment file(s).", "ok")
        self.console.note("Continuing: new turns will append to this state.", "info")
        return True

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

    def print_welcome(self):
        """Render the startup banner for this session."""
        print_welcome(self.allowed_dir, self.server_url, self.args,
                      self.unique_id, self.cache_dir, self.model,
                      console=self.console)

    def print_announcements(self):
        """Print the info / warning banners that depend on the CLI flags.

        All of these use the ``ok`` / ``warn`` levels (never ``info``) so they
        still reach the user in the quiet, default (non-verbose) view: a missing
        API key, unrestricted shell execution and --sudo are exactly the things
        that must not be hidden behind --verbose.
        """
        self.console.announce_api_key(present=bool(self.api_key))

        if shell_forces_ask_permission(self.args):
            self.console.announce_shell()

        # --sudo hands the agent unattended control, so make that impossible to miss.
        if sudo_disables_prompts(self.args):
            self._warn_sudo()

        if self.enable_subagents:
            self.console.note(
                f"Sub-agents enabled ({len(self.subagents_registry)} available: "
                f"{', '.join(self.subagents_registry)}).", "ok")

    def _print_session_id(self):
        """Print the session's unique id as a minimal "Assistant" turn.

        The id is rendered as an assistant-styled (green) turn carrying *only*
        the id in brackets, so the user can see / copy the id this conversation
        is being saved under without adding another line to the banner.
        """
        self.console.turn("Assistant", f"[{self.unique_id}]", accent="green")

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
        # Startup output comes FIRST, before the session is resolved. The banner
        # and the flag-dependent warnings (API key / shell / --sudo / an unknown
        # model) describe how this process is configured -- how much the agent
        # may do unattended -- so the user must see them *before* being asked
        # whether to resume an older conversation (and before anything can
        # consume stdin). None of them depends on which session is finally used:
        # they read args, allowed_dir, cache_dir, server_url and the resolved
        # model, plus the provisional id minted in __init__.
        self.print_welcome()
        self.print_announcements()
        self._warn_compaction_budget()

        # Only now resolve the session (resume offer / restore / fresh start),
        # because all of that can consume input and can replace the id. The id
        # line and the tip therefore come after it and show the id actually in
        # use (the banner may have shown the provisional one).
        self.start_session()
        self._print_session_id()
        self._emit_tip()

        # The --prompt value is a one-shot initial turn: it is sent verbatim and
        # then cleared, so later turns come from the user (manual mode) or are
        # produced automatically by the loop (auto mode).
        pending = initial_prompt
        try:
            while True:
                user_input = pending if pending is not None else self._read_user_line()
                if user_input is None or user_input.lower() in self.EXIT_COMMANDS:
                    self.console.note("Thank you. Goodbye! Saving history...", "info")
                    break
                if pending is not None:
                    self.console.rule(title="Preset prompt", char="=", fg="cyan")
                    self.console.short_divider()
                    self.console.turn("You", user_input, accent="cyan")
                    pending = None
                # Interactive slash commands are handled in place and never
                # reach the LLM.
                if user_input.startswith("/") and self._handle_command(user_input):
                    continue
                self._run_turn(user_input)
        except _Abort:
            # Ctrl-C during a permission prompt already saved the session.
            return
        except KeyboardInterrupt:
            self.console.note("Interrupting: Mini Code Exiting...", "error")
            self.shutdown(farewell=False)
            return
        except Exception as e:
            self.console.note(f"Exception {e}. Exiting...", "error")
            self.shutdown(farewell=False)
            return
        self.shutdown()

    def _emit_tip(self):
        """Point at the history/navigation commands once, at startup."""
        tip = ("Tip: /history to browse the conversation, /history <n> to inspect a "
               "turn, /reset <n> to rewind. Type /help for all commands.")
        if self.plan_mode:
            tip = ("Tip: /plan to show the current goal + plan (plan mode), "
                   "/history to browse the conversation, /history <n> to inspect a "
                   "turn, /reset <n> to rewind. Type /help for all commands.")
        self.console.emit(self.console.style(tip, dim=True, fg="cyan"))

    def _warn_compaction_budget(self):
        """Warn when the compaction budget came from the unknown-model fallback.

        An unknown model gets a deliberately conservative window, so the user
        should know the limit is a guess they can override rather than a
        measured value. This is a startup warning (it describes the run's
        configuration, not a turn), so it is printed with the other
        announcements rather than after the session has been settled.
        """
        if self.compaction_budget.source == "default":
            self.console.note(
                f"Unknown model {self.model!r}: assuming a "
                f"{self.compaction_budget.context_window:,}-token context window, so "
                f"history compacts at {self.compaction_budget.token_limit:,} input "
                f"tokens (+{self.compaction_budget.reserved_output_tokens:,} reserved "
                f"for the reply). Override with --max-context-tokens.", "warn")

    def _run_turn(self, user_input):
        """Run one user turn: record it, step until the model stops, then review.

        A step that makes no tool calls ends the turn. In plan mode the loop then
        asks the model to review the plan and takes one more step, so the review
        can drive further work. The extra step happens *outside* the plan-mode
        branch's result: a reply with no tool calls must always end the turn,
        otherwise non-plan sessions re-call the model forever.
        """
        self.messages.append({"role": "user", "content": user_input})

        if self.plan_mode:
            self.console.note("Generating plan for agent.", "info")
            self._record_goal_request(user_input)
            self._generate_plan_and_inject(user_input)
            self.console.note("Plan generated.", "info")

        while True:
            should_continue = self._step()
            self._persist()
            if should_continue:
                self.console.note(
                    "Continuing with LLM calls until no more tool calls.", "info")
                continue
            self.console.note("LLM seems to have to an end of the task.", "info")
            if not self.plan_mode:
                return
            self.console.note("Reviewing the plan, if everything is done.", "info")
            self._review_plan_completion()
            self._persist()
            # One more step so the review can drive further work. Persist before
            # deciding so the review step's messages are recorded even when it
            # ends the turn (a no-tool-call reply must never be left unsaved).
            should_continue = self._step()
            self._persist()
            if not should_continue:
                return

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
            self.console.note("Generating a plan for the current goal...", "info")
            self.console.start_thinking("planning")
            # The loop's own bookkeeping requests (plan authoring / review /
            # compaction) are single user turns with no ``tools``, so the
            # server's thinking-mode rule does not require any assistant
            # reasoning text here; the field is simply not forwarded.
            response, _, usage, _ = utils.unpack_llm_result(
                utils.call_openai_server(
                    [{"role": "user", "content": plan_prompt_text}],
                    # The generated plan can be long: leave it unlimited (the
                    # server decides the cap) rather than capping the
                    # step-by-step authoring.
                    max_tokens=None,
                    temperature=0.0,
                    model=self.model,
                    server_url=self.server_url,
                    api_key=self.api_key,
                    timeout=self.args.llm_timeout,
                ))
            self.console.stop_thinking()

            self._record_usage(usage)
            # The plan prompt is one message holding a flattened copy of the whole
            # transcript, so its prompt_tokens is a safe (slightly high) upper
            # bound on the transcript's own size -- worth keeping, since it is a
            # real count from this model rather than a character estimate.
            self._note_prompt_size(usage, len(self.messages))
            plan_text = response or ""
        except Exception as exc:
            self.console.note(f"Could not generate a plan: {exc}", "warn")

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
            self.console.note("Reviewing the current plan...", "info")
            self.console.start_thinking("reviewing plan")
            response, _, usage, _ = utils.unpack_llm_result(
                utils.call_openai_server(
                    [{"role": "user", "content": plan_prompt_text}],
                    max_tokens=1024,
                    temperature=0.0,
                    model=self.model,
                    server_url=self.server_url,
                    api_key=self.api_key,
                    timeout=self.args.llm_timeout,
                ))
            self.console.stop_thinking()
            self._record_usage(usage)
            self._note_prompt_size(usage, len(self.messages))
            summary = response or ""
        except Exception as exc:
            self.console.note(f"Could not review the plan: {exc}", "warn")

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


    def _handle_compaction(self, compacted, tail=()):
        """Re-inject goal + plan into the compacted messages (plan mode).

        Plan-mode counterpart to the plan handling inside :meth:`maybe_compact`,
        which takes a ``plan_hook`` rather than the owning session so it can stay
        a network-free module-level helper. The loop wires in this method only
        when plan mode is on; it re-injects the goal + plan via
        :meth:`_apply_plan_to_compacted`, which also rewrites the tracking files
        to match. ``tail`` is the turn that is still in flight, which
        ``maybe_compact`` appends after the hook's messages.

        The hook must update ``compacted`` **in place**: ``maybe_compact``
        returns *that* list to the caller, which then assigns it to
        ``self.messages``. Rebinding (``self.messages = ...``) instead would be
        discarded a moment later, silently dropping the goal and plan -- which is
        exactly what happened before this was a slice assignment.
        """
        compacted[:] = self._apply_plan_to_compacted(compacted, tail)

    def _apply_plan_to_compacted(self, messages, tail=()):
        """Re-inject the current goal + plan after a history compaction.

        Compaction replaces the transcript with a summary, which would otherwise
        drop the goal the loop recorded and the plan it maintains. We read them
        back off disk and append them as fresh user turns so the model keeps
        working against the goal it is still trying to finish. The goal is
        appended last so it reads as the most recent instruction the model
        should act on -- ``maybe_compact`` then appends the in-flight turn, which
        is the same instruction verbatim.
        """
        plan_text = self.plan_manager.read_plan()
        goal_text = self.plan_manager.read_goal()

        # Recover the user's latest instruction from the *in-flight turn* -- not
        # from ``messages``, which by now holds the summary -- so a session that
        # compacts before its goal was recorded still restates the real
        # instruction instead of the summary text.
        current_input = None
        for m in reversed(list(tail or ())):
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
        only when this is a child session (report_path set), only the file tools
        (read_file / write_file / edit), and only when the resolved path is
        exactly the report file.
        """
        if not self.report_path or name not in ("read_file", "write_file", "edit"):
            return False
        path = tool_params.get("file_path", "")
        if not path:
            return False
        try:
            return os.path.realpath(path) == os.path.realpath(self.report_path)
        except (ValueError, TypeError):
            return False

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

        # A loop, not recursion: holding Enter used to recurse once per blank
        # line, so a long run of empty input could exhaust the interpreter stack.
        while True:
            self.console.short_divider()

            user_input = self.console.read_line(
                self.console.style("You:", bold=True, fg="cyan") + " ")

            # EOF / Ctrl-C at the prompt: the caller treats None as "quit", so it
            # must not be dereferenced here (a piped or closed stdin would
            # otherwise crash the loop instead of exiting cleanly).
            if user_input is None:
                return None
            if user_input.strip():
                return user_input
            self.console.note("No input. (Type /help for commands, /quit to exit.)", "warn")

    def _handle_command(self, user_input):
        """Handle an interactive slash command before it reaches the LLM.

        Supported:
          /history           list the high-level recorder (turn number, what
                             happened, and which history_{n}.json + index)
          /history <n>       show the full messages recorded for turn <n>
          /history last      ...for the most recent turn
          /reset [<n>|last]  rewind the conversation to the end of turn <n>
                             (a bare /reset only re-saves the current state;
                             see _do_reset)
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
            if arg:
                self._inspect_history(arg)
            else:
                self._show_history()
            return True
        if cmd in ("reset", "rewind"):
            self._do_reset(arg)
            return True
        if cmd == "help":
            print_help(self.args)
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

        ``/reset <n>`` and ``/reset last`` are *explicit* rewinds: they truncate
        the transcript to the end of that turn and drop everything recorded after
        it. A bare ``/reset`` is the cautious variant -- it never truncates: when
        the transcript really ends at the last recorded turn it just reports
        "Already at turn N", and when unrecorded messages follow that turn it
        *warns* instead of destroying them, because the user never asked for a
        specific turn. Forcing the rewind is one explicit command away:
        ``/reset <n>``.
        """
        resolved = self._resolve_turn_record(arg)
        if resolved is None:
            return
        position, record = resolved
        explicit = bool((arg or "").strip())

        if not explicit and self._transcript_ends_at(record):
            self.console.note(
                f"Already at turn {position + 1} (the last recorded turn); "
                f"nothing to rewind. Use /reset <n> to rewind to an earlier turn.",
                "ok")
            return
        self._reset_history(position + 1, record, force=explicit)

    def _transcript_ends_at(self, record):
        """Whether the live transcript is exactly the state recorded for a turn.

        That is true when the session is still on the record's segment and its
        last message is the one the record points at -- i.e. a bare ``/reset``
        has nothing left to drop.
        """
        return (_safe_int(record.get("history_file"), -1) == self.history_index
                and len(self.messages) == _safe_int(record.get("index"), -1) + 1)

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
        self.console.note(f"LLM is trying to call {len(tool_calls)} tool(s).", "info")
        # Show a "… calling tool" spinner (in-place, non-scrolling) while the
        # tool(s) execute, mirroring the thinking spinner that covers the LLM
        # call. Always stopped (finally) so it can never leak into the next turn.

        try:
            for call in tool_calls:
                self._run_one_tool_call(call)
        finally:
            self.console.stop_tool_call()

    def _run_one_tool_call(self, call):

        call = call or {}
        # ``or {}`` (not merely a default) because a malformed server response can
        # carry an explicit ``"function": null``; indexing through that used to
        # raise AttributeError and kill the whole session instead of reporting a
        # bad tool call back to the model.
        function = call.get("function") or {}
        call_id = call.get("id", "")
        name = function.get("name", "") or ""
        params_str = function.get("arguments", "{}")
        self.console.start_tool_call(tool_name=name)

        try:
            tool_params = json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            self.console.note(f"Error parsing arguments for tool {name}", "error")
            self._append_tool_result(call_id,
                f"Error: the arguments for tool {name} were not valid JSON "
                f"({params_str!r}). Pass a JSON object of named parameters.")
            return

        # The arguments must be a JSON object (a model occasionally emits a bare
        # list/string). Treat that as a normal tool error rather than crashing on
        # ``tool_params.get(...)`` below.
        if not isinstance(tool_params, dict):
            self.console.note(f"Tool {name} was called with non-object arguments.", "error")
            self._append_tool_result(call_id,
                f"Error: the arguments for tool {name} must be a JSON object of "
                f"named parameters, got {type(tool_params).__name__}.")
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
        if (not tool_paths_are_allowed(name, tool_params, self.allowed_dir)
                and not self._is_report_file(name, tool_params)):
            # The one exemption is a sub-agent's own report.md, which lives under
            # the cache dir (outside --allowed-dir) so the parent can read it.
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
        # The context-occupancy bar rides on the header, so "how full is the
        # window" is visible exactly when the user is reading a reply -- and it
        # refreshes every turn because it is recomputed from the projection
        # (which is reset after a compaction, so the bar drops back down there).
        self.console.turn("Assistant", text, accent="green",
                          badge=self._context_badge())

    def _persist(self):
        """Write the current segment, recorder and usage totals to disk.

        Each write is attempted independently and none of them may raise: a full
        disk, a read-only cache dir or a permission hiccup has to degrade to a
        warning. Quitting must never crash, and a failed transcript save must not
        stop the usage totals (or the other way round) from being written.
        """
        def attempt(label, write):
            try:
                return write()
            except Exception as e:
                self.console.note(f"Could not save the {label}: {e}", "error")
                return None

        attempt("history segment", lambda: save_history(
            self.messages, self.cache_dir, self.unique_id, self.history_index))
        attempt("history recorder", lambda: save_summary(
            self.summary, self.cache_dir, self.unique_id))
        usage_path = attempt("segment usage", lambda: save_usage_segment(
            self.token_stats, self.cache_dir, self.unique_id, self.history_index))
        total_path = attempt("session usage", lambda: save_total_usage(
            self.total_usage, self.cache_dir, self.unique_id))
        if usage_path or total_path:
            self.console.note(
                f"Saved usage to: {usage_path} (session total: {total_path})", "info")

    def shutdown(self, farewell=True):
        """Persist and print a final summary (called on quit / exit).

        ``farewell=False`` is used by the interrupt / fatal paths in :meth:`run`,
        which have already told the user (in red) that the session is being torn
        down: a cheerful "thanks and here is your state" box on top of a crash is
        the wrong note, and the state it would recap is the state that just
        failed.
        """
        self._persist()
        segments = history_segment_indices(self.session_dir)
        # List what is really on disk. The old message claimed
        # "history_0..{history_index}.json", which was a lie after a /reset had
        # removed the higher segments (or moved the session back onto an earlier
        # one) -- it named files that no longer existed and hid the ones that did.
        if segments:
            present = (f"history_{segments[0]}.json"
                       if len(segments) == 1 else
                       f"history_{segments[0]}..{segments[-1]}.json")
        else:
            present = f"history_{self.history_index}.json"
        self.console.note(
            f"Saved session to {self.cache_dir}/{self.unique_id}/ "
            f"({present} + summary_history.json).", "ok")
        # The detailed per-field usage block is verbose-only (the console
        # documents it that way); in the quiet default view the good-bye box
        # below carries the same totals, so nothing is lost by leaving it out.
        if self.console.verbose:
            self.console.usage_summary(self.total_usage)
        if farewell:
            self._farewell(segments)

    def _farewell(self, segments=None):
        """Print the good-bye box: version, sign-off and what the session left.

        Printed once (see ``_farewell_shown``) and only on a normal exit. It
        restates the state in one glanceable place -- the same facts the user
        would otherwise have to piece together from the save note, ``/history``
        and the (verbose-only) usage readout, which in the quiet default view
        means they saw none of it.
        """
        if self._farewell_shown:
            return
        self._farewell_shown = True
        from . import __version__ as version

        if segments is None:
            segments = history_segment_indices(self.session_dir)
        if not segments:
            where = f"history_{self.history_index}.json"
        elif len(segments) == 1:
            where = f"history_{segments[0]}.json"
        else:
            where = f"history_{segments[0]}..{segments[-1]}.json"

        turns = len(self.summary)
        usage = self.total_usage or {}

        def count(key):
            # A display detail on the exit path must never raise: a server that
            # reported ``null`` usage leaves a None here, and crashing while
            # saying goodbye would be the worst possible time to fail.
            try:
                return int(usage.get(key) or 0)
            except (TypeError, ValueError):
                return 0

        rows = [
            ("session id", self.unique_id),
            ("saved under", f"{self.cache_dir}/{self.unique_id}/"),
            ("history", f"{where} (segment {self.history_index}), "
                        f"{_plural(len(self.messages), 'message')}"),
            ("conversation", _plural(turns, "recorded turn")),
            ("tokens", f"{count('prompt_tokens'):,} in / "
                       f"{count('completion_tokens'):,} out "
                       f"({count('total_tokens'):,} total)"),
            ("calls", f"{count('llm_calls'):,} LLM, "
                      f"{count('tool_calls'):,} tool"),
        ]
        self.console.farewell(version, rows)

    # -- single LLM turn ---------------------------------------------------

    def _step(self):
        """Run a single LLM turn: call the model, render its text, run its tool
        calls. Returns True if the LLM made tool calls and we're in auto-mode,
        signalling the loop to continue without asking the user."""
        # Keep long sessions within the model's context window. Compaction
        # rolls the session over to a fresh history segment (history_index + 1)
        # and returns that bump so the recorder can stay flat across it.
        segment_before = dict(self.token_stats)
        # The segment as it stands *before* compaction: it has to be flushed to
        # disk as it is being closed (see below), because nothing else writes it.
        closing_messages = self.messages
        projected = self._projected_input_tokens()
        self.messages, did_compact, new_index = maybe_compact(
            self.messages, self.server_url, self.model, self.api_key,
            self.args.llm_timeout, self.token_stats, self.history_index,
            token_limit=self.compaction_budget.token_limit,
            reserved_output_tokens=self.compaction_budget.reserved_output_tokens,
            current_tokens=projected,
            plan_hook=self._handle_compaction if self.plan_mode else None,
            console=self.console,
        )
        # The compaction call itself was billed to the segment that is being
        # closed, so fold its delta into the session total here (the rollover
        # below zeroes the segment counters).
        self._bill_compaction(segment_before)

        if did_compact:
            # The transcript was replaced, so the last prompt-size report now
            # describes a conversation that no longer exists. Dropping it makes
            # the next projection a plain estimate of the (much shorter)
            # compacted history rather than a stale, over-large number that
            # would trigger compaction again immediately.
            self.last_prompt_tokens = 0
            self.last_prompt_message_count = 0
            # Close the old segment's files before moving on: its spend is
            # preserved for the totals, and its *messages* are flushed so the
            # recorder keeps pointing at a transcript that really exists on disk.
            # Nothing else writes a closing segment -- the interactive loop only
            # persists on quit -- so without this history_{n}.json would be short
            # by every turn recorded in that segment, and /reset (or a restore)
            # would rewind to a state the user never saw.
            try:
                save_usage_segment(self.token_stats, self.cache_dir, self.unique_id,
                                   self.history_index)
                save_history(closing_messages, self.cache_dir, self.unique_id,
                             self.history_index)
            except Exception as e:
                self.console.note(
                    f"Could not save the closed history segment "
                    f"history_{self.history_index}.json: {e}. The compacted "
                    f"conversation is unaffected, but /reset back into that "
                    f"segment may be incomplete.", "warn")
            self.history_index = new_index
            self.token_stats = empty_usage()
            # Snapshot the compacted messages into the new segment so its turns
            # are preserved, and record the summary as the first turn of it.
            save_history(self.messages, self.cache_dir, self.unique_id, self.history_index)
            self._record_turn(
                f"summary: history_{new_index - 1}.json "
                f"({_plural(len(closing_messages), 'message')}) compacted into a "
                f"summary; history_{self.history_index}.json started",
                self.messages)

        # Show a "… thinking" spinner while the LLM works (in-place, non-scrolling;
        # a no-op on non-TTY / ASCII output). Stopped right after the call returns.
        # Capture the transcript length *before* the call: this is exactly how
        # many messages the reported prompt_tokens describe.
        sent_message_count = len(self.messages)
        self.console.start_thinking()
        response_text, tool_calls, usage, reasoning_content = utils.unpack_llm_result(
            utils.call_openai_server(
                self.messages,
                max_tokens=self.args.max_tokens,
                temperature=self.args.temperature,
                model=self.model,
                server_url=self.server_url,
                tools=self.llm_tools,
                api_key=self.api_key,
                timeout=self.args.llm_timeout,
            ))
        self.console.stop_thinking()

        # A failed LLM call returns (None, None, None, None): treat it as "nothing
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
        self._note_prompt_size(usage, sent_message_count)
        made_tool_calls = bool(tool_calls)
        if made_tool_calls:
            self.token_stats["tool_calls"] += len(tool_calls)
            self.total_usage["tool_calls"] += len(tool_calls)
            # The model predicted more tool calls: once the work settles down a
            # completion review is due again (finalized_plan).
            self.finalized_plan = True

        if response_text:
            self._print_assistant(response_text)

        # Append the assistant turn exactly as the server returned it. The
        # thinking-mode ``reasoning_content`` belongs on this message and must
        # be sent back verbatim on every later request that carries ``tools``:
        # DeepSeek V3.2+ rejects the follow-up with HTTP 400
        # ("The `reasoning_content` in the thinking mode must be passed back to
        # the API") when an earlier assistant turn dropped it. It is stored but
        # never rendered -- it is the model's private scratchpad, not the reply,
        # so ``_print_assistant`` is still fed only ``response_text``.
        # The field is added *only* when the server actually returned it (an
        # empty string counts: that is what such a model returns when it did no
        # reasoning), so non-thinking models and older transcripts keep exactly
        # the message shape they had before.
        assistant_message = {"role": "assistant", "content": response_text}
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content
        self.messages.append(assistant_message)

        if tool_calls:
            self._run_tool_calls(tool_calls)

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
        self.console.usage_line(usage, transient=False,
                                llm_calls=self.total_usage["llm_calls"])

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
        else:
            # A step that produced neither text nor tool calls (a failed call, or
            # a model that answered with nothing) still appended an assistant
            # message, so it is recorded as well. The recorder must cover the
            # *final* message: otherwise a bare /reset sees an unrecorded tail
            # and refuses to touch a conversation that is in fact settled.
            self._record_turn("assistant (no response)", self.messages)

        return made_tool_calls and self.args.auto_mode


class _Abort(Exception):
    """Raised to unwind the permission prompt after a Ctrl-C / shutdown."""


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    session = AgentSession(args)
    session.run(initial_prompt=args.prompt)


if __name__ == "__main__":
    main()
