"""
Usage:
    mini-code

Prompt:

write a similar model.py file but for a LSTM based language model. Make it minimal and call it model_lstm.py it should still be compatible with eval.py

"""

import os
import re
import json
import signal

import time
import uuid

import argparse

from . import prompts
from . import shell_safety_utils
from . import tools
from . import utils
from . import ui
from . import plan as plan_utils


# ANSI escape codes for colors and bold (kept for the couple of ad-hoc spots
# that still build their own strings; the rest of the package renders through
# the ui.Console in ui.py).
BOLD = '\033[1m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

DEFAULT_CACHE_LOCATION=os.path.expanduser("~/.cache/mini_code/sessions/")
DEFAULT_MODEL_URL="http://0.0.0.0:30000"
DEFAULT_TEMPERATURE=1.0
DEFAULT_TOOL_TIMEOUT=120  # seconds before a single tool call is aborted
DEFAULT_HTTP_TIMEOUT=500  # seconds before an LLM HTTP request is aborted
DEFAULT_MAX_MESSAGES=150  # messages in history before it is compacted to a summary

def history_dir(args, session_name=""):
    """Return (creating it if needed) this session's on-disk history folder.

    Everything a session writes lives under ``cache_dir/{session_name}/``: one
    ``history_{n}.json`` file per conversation *segment* -- a brand-new segment
    is started every time the history is compacted -- plus the high-level
    recorder kept in ``summary_history.json``.
    """
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = DEFAULT_CACHE_LOCATION
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir, session_name), exist_ok=True)
    return os.path.join(cache_dir, session_name)


def save_history(messages, args, session_name="", history_index=0):
    """Persist a detailed conversation-history *segment* to disk.

    A segment is written to ``history_{history_index}.json``. A session only
    ever has one *active* segment (the current ``history_index``), which is
    overwritten every step with the latest messages. Because messages only ever
    *append* within a segment, any earlier index in it still marks a faithful
    prefix -- so a recorded turn's ``index`` lets you rebuild the exact state at
    the end of that turn. Compaction rolls the session over to a higher index
    and freezes the old segment, so its turns stay intact snapshots.
    """
    d = history_dir(args, session_name)
    history_path = os.path.join(d, f"history_{history_index}.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)
    return history_path


def save_summary(summary, args, session_name=""):
    """Persist the high-level recorder to ``summary_history.json``.

    The recorder is the flat list of per-turn records built by
    :meth:`AgentSession._record_turn` (turn number, what happened, and where the
    turn lives in the history segments). It lets the user navigate and rewind
    the conversation without reading the full detailed histories.
    """
    d = history_dir(args, session_name)
    summary_path = os.path.join(d, "summary_history.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def load_history_from_dir(directory):
    """Load the most complete history for a session directory.

    Prefers the newest ``history_{n}.json`` segment (sessions created by the
    high-level recorder) and falls back to the classic ``history.json`` (older
    sessions). Returns the messages list, or None if nothing could be loaded
    (never raises: a failed restore must fall back to a fresh session).
    """
    if not directory or not os.path.isdir(directory):
        return None
    newest, highest = None, -1
    for fn in os.listdir(directory):
        m = re.fullmatch(r"history_(\d+)\.json", fn)
        if m and int(m.group(1)) > highest:
            highest, newest = int(m.group(1)), fn
    path = (os.path.join(directory, newest) if newest is not None
            else os.path.join(directory, "history.json"))
    try:
        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        return messages if isinstance(messages, list) else None
    except Exception as e:
        ui.console.note(f"Failed to load history from {directory}: {e}", "error")
        return None


def load_summary_from_dir(directory):
    """Load a session's high-level recorder (``summary_history.json``).

    Returns the list of records, or an empty list if there is none or it cannot
    be read. Best-effort only: a missing or garbled recorder must never break a
    restore.
    """
    if not directory or not os.path.isdir(directory):
        return []
    path = os.path.join(directory, "summary_history.json")
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        return saved if isinstance(saved, list) else []
    except Exception:
        return []


def accumulate_usage(token_stats, usage):
    """Add a single LLM call's token usage into the running session totals."""
    if not usage:
        return
    token_stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    token_stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    token_stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    token_stats["llm_calls"] += 1


def save_usage(token_stats, cache_dir, session_name=""):
    """Persist the accumulated token usage to the session folder."""
    os.makedirs(cache_dir, exist_ok=True)
    usage_path = os.path.join(cache_dir, session_name, f"usage.json")
    with open(usage_path, 'w') as f:
        json.dump(token_stats, f, indent=2)
    return usage_path


def print_usage_summary(token_stats):
    """Print the accumulated token usage for the session so far."""
    ui.console.usage_summary(token_stats)


def generate_unique_id():
    """Generate a unique ID based on current time and a random component."""
    timestamp = int(time.time() * 1000000)  # microseconds
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _resolve_history_path(restore_session):
    """Resolve the --restore-session path to the history.json file it holds.

    Accepts either the session directory itself (e.g.
    ~/.cache/mini_code/sessions/1792028090823027-a9d77653/) or a direct path to
    a history.json file. A leading ~ is expanded for convenience.
    """
    candidate = os.path.expanduser(restore_session)
    if os.path.isdir(candidate):
        candidate = os.path.join(candidate, "history.json")
    return candidate


def _restore_dir(restore_session):
    """Resolve a --restore-session value to the directory its history lives in.

    Accepts either the session directory itself or a direct path to a history
    file inside it (``history.json`` or ``history_{n}.json``). A leading ~ is
    expanded for convenience.
    """
    if not restore_session:
        return None
    candidate = os.path.expanduser(restore_session)
    if os.path.isfile(candidate):
        return os.path.dirname(candidate)
    return candidate


def find_last_session_dir(cache_dir):
    """Return the path to the most recently *active* session directory under
    ``cache_dir``, or None if there is none.

    "Most recent" is decided by the newest modification time of any file inside
    each session folder (a session rewrites its ``history_{n}.json`` every step),
    so this reflects the last session the user actually worked in rather than the
    one most recently *created*. Session folders with no files are ignored. Never
    raises: a missing or unreadable cache dir simply yields None so
    ``--load-last-session`` can fall back to a fresh session.
    """
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    try:
        entries = [
            os.path.join(cache_dir, name)
            for name in os.listdir(cache_dir)
            if os.path.isdir(os.path.join(cache_dir, name))
        ]
    except OSError:
        return None

    best_dir, best_time = None, -1.0
    for session in entries:
        try:
            files = [
                os.path.join(session, f)
                for f in os.listdir(session)
                if os.path.isfile(os.path.join(session, f))
            ]
            if not files:
                continue
            mtime = max(os.path.getmtime(f) for f in files)
        except OSError:
            continue
        if mtime > best_time:
            best_time, best_dir = mtime, session
    return best_dir


def load_history(restore_session):
    """Load a previous session's message history from disk.

    Returns the list of messages, or None if nothing could be loaded (missing
    file, wrong shape, or unreadable JSON). Never raises: a failed restore
    should let the caller fall back to a fresh session rather than crash.
    """
    if not restore_session:
        return None

    history_path = _resolve_history_path(restore_session)
    if not os.path.isfile(history_path):
        ui.console.note(f"No history found to restore from: {restore_session}", "warn")
        return None

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError("history file did not contain a message list")
    except Exception as e:
        ui.console.note(f"Failed to load history from {restore_session}: {e}", "error")
        return None

    return messages

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
    parser.add_argument("--restore-session", type=str, help=f"The location of the previous session. (e.g. ~/.cache/mini_code/sessions/1792028090823027-a9d77653/)")
    parser.add_argument("--load-last-session", action="store_true", help="Load the most recent previous session from the cache dir and continue it (under a new session id), instead of starting fresh. --restore-session takes precedence if also given.")

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
    parser.add_argument("--llm-timeout", type=int, default=DEFAULT_HTTP_TIMEOUT, help=f"HTTP timeout in seconds for LLM requests (reasoning models can be slow). default={DEFAULT_HTTP_TIMEOUT}")
    parser.add_argument("--tool-timeout", type=int, default=DEFAULT_TOOL_TIMEOUT, help=f"Timeout in seconds for a single tool call. default={DEFAULT_TOOL_TIMEOUT}")
    parser.add_argument("--plan", action="store_true", help="Enable plan mode: the agent keeps a CURRENT_PLAN.md (its step-by-step plan, marked finished when the goal is done) and CURRENT_GOAL.md (the user's latest instruction) in the session directory. Also changes how history compaction works.")



    # Output verbosity
    parser.add_argument("--verbose", action="store_true", help="Print the full, detailed output. Without it (default), only user/assistant turns, token usage and tool calls are shown.")

    args = parser.parse_args()
    return args


def style_line_prompt(text):
    """A short, styled inline prompt (used for the "You:" line)."""
    return f"{BOLD}{CYAN}{text}{RESET} "


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
    ui.console.emit("  /reset [<n>]      rewind to the end of turn n (drops everything after it)")
    ui.console.emit("  /clear            clear the screen (keeps the conversation history)")
    ui.console.emit("  /help             show this help")
    ui.console.emit("  /quit, /exit      save the session and exit")
    ui.console.rule(char="═", fg="cyan")

def call_function_with_timeout(func, tool_params, name, timeout_seconds=DEFAULT_TOOL_TIMEOUT):
    """calls function with params, aborting after `timeout_seconds`."""
    class TimeoutError(Exception):
        pass

    def timeout_handler(signum, frame):
        raise TimeoutError("Function call timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func(**tool_params)
    except TimeoutError:
        ui.console.note(f"Function {name} timed out after {timeout_seconds} seconds.", "error")
        result = f"Error: Function {name} timed out after {timeout_seconds} seconds."
    finally:
        signal.alarm(0)  # Disable the alarm
    
    return result



def is_within_allowed_dir(file_path, allowed_dir):
    """Return True if the resolved `file_path` lives inside `allowed_dir`.

    Uses os.path.realpath to resolve symlinks and '..' so path-traversal or
    symlink tricks cannot escape the allowed directory.
    """
    target = os.path.realpath(file_path)
    root = os.path.realpath(allowed_dir)
    try:
        return os.path.commonpath([target, root]) == root
    except (ValueError, TypeError):
        # Different drives (Windows) or a non-absolute path: fail closed.
        return False


# Maps each filesystem tool to the names of its path-bearing arguments, so the
# --allowed-dir sandbox can be enforced uniformly. Keeping this in one place
# means a future tool cannot silently bypass the sandbox by using a different
# argument name (e.g. glob's `pattern` instead of `file_path`).
_FILE_TOOL_PATH_ARGS = {
    "read_file": ["file_path"],
    "write_file": ["file_path"],
    "grep": ["file_path"],
    "glob": ["pattern"],
}


def tool_paths_are_allowed(name, tool_params, allowed_dir):
    """Return True if every filesystem path argument of `name` stays inside
    `allowed_dir`.

    Tools not listed in _FILE_TOOL_PATH_ARGS have no path to check and pass
    through (return True). A single path argument may itself be a list (e.g. a
    batch of files), so each entry is validated.
    """
    path_args = _FILE_TOOL_PATH_ARGS.get(name, [])
    for arg_name in path_args:
        value = tool_params.get(arg_name, "")
        values = value if isinstance(value, (list, tuple)) else [value]
        for path in values:
            if path and not is_within_allowed_dir(path, allowed_dir):
                return False
    return True


def maybe_compact(messages, server_url, model, api_key, llm_timeout, token_stats, history_index=0, max_messages=DEFAULT_MAX_MESSAGES):
    """Compact the conversation history once it grows beyond `max_messages`.

    Long (especially auto-mode) sessions can overflow the model's context window.
    When the history gets too large we ask the LLM to summarize it and replace the
    old turns with that summary, keeping the original system prompt as the first
    message. The compaction call itself is a single, tool-free request, so it
    cannot trigger further tool calls or recurse into compaction.
    """
    if len(messages) <= max_messages:
        # No compaction: keep the same segment (history_index) and signal the
        # caller with (messages, did_compact=False, unchanged_index). Returning
        # the full triple here is required -- callers unpack all three, so a
        # bare messages would crash them.
        return messages, False, history_index

    try:
        ui.console.note(f"Compacting conversation history ({len(messages)} messages -> summary)...", "warn")
        summary_prompt = prompts.compaction_prompt(messages)
        summary, _, compact_usage = utils.call_openai_server(
            [{"role": "user", "content": summary_prompt}],
            max_tokens=1024,
            temperature=0.0,
            model=model,
            server_url=server_url,
            api_key=api_key,
            timeout=llm_timeout,
        )
        accumulate_usage(token_stats, compact_usage)
        if summary:
            system_content = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
            compacted = []
            if system_content:
                compacted.append({"role": "system", "content": system_content})
            compacted.append({
                "role": "user",
                "content": f"Here is a summary of our earlier conversation so far:\n{summary}",
            })
            ui.console.note(f"History compacted to {len(compacted)} messages.", "warn")
            # A new segment (history_index + 1) is started on compaction: the
            # compacted messages are its snapshot, and every later turn appends
            # to it. The caller records a single high-level 'summary' turn for
            # this index so the recorder stays flat across the rollover.
            return compacted, True, history_index + 1
    except Exception as e:
        ui.console.note(f"Failed to compact history: {e}", "error")

    return messages, False, history_index


def shell_forces_ask_permission(args):
    """Whether enabling the shell should force per-command permission prompts.

    Running shell commands without asking is dangerous (the agent can run any
    command and escape --allowed-dir), so whenever the shell is enabled we
    force ask-permission for the whole session. Exposed as a pure function so it
    can be unit-tested independently of argparse / stdin.
    """
    return bool(getattr(args, "enable_shell", False))


def sudo_disables_prompts(args):
    """Whether --sudo was passed.

    In sudo mode the agent never stops to ask for confirmation: both the
    shell-safety ASK tier and --ask-permission are bypassed, so any
    non-destructive tool call / shell command runs unattended. Destructive
    DENY-tier commands are still refused outright -- this flag only removes the
    *prompts*, not the hard safety refusals. Pure function so it can be
    unit-tested independently of argparse / stdin.
    """
    return bool(getattr(args, "sudo", False))


class AgentSession:
    """A single interactive agent session.

    The whole loop used to live in one ~300 line ``main()``. Splitting it into
    small, named methods keeps each step (read input -> call the LLM -> run tool
    calls -> persist) easy to follow and gives the interesting bits somewhere to
    be unit-tested.
    """

    EXIT_COMMANDS = {"quit", "exit", "/quit", "/exit"}
    CLEAR_COMMANDS = {"clear", "/clear"}

    def __init__(self, args, console=None):
        self.args = args
        self.allowed_dir = os.path.abspath(args.allowed_dir)
        self.cache_dir = args.cache_dir if args.cache_dir else DEFAULT_CACHE_LOCATION

        # The console that renders everything the agent prints. Unit tests /
        # other callers can swap it out (e.g. AgentSession(args, console=...) with
        # a color=False console) but it defaults to the shared ui.console.
        self.console = console if console is not None else ui.console
        # --verbose toggles the amount of detail printed. Defaults to True so
        # existing behaviour / callers are unchanged when the flag is omitted.
        self.console.verbose = getattr(args, "verbose", False)

        # LLM / server configuration.
        self.server_url = f"{args.url}"
        model_name = utils.determine_vllm_model(self.server_url)
        self.model = model_name if model_name else args.model
        self.api_key = args.api_key if args.api_key else os.environ.get("MINI_CODE_API_KEY", "")

        # Which tools the LLM may call. The shell tool is only available when
        # --enable-shell was passed.
        forbidden = None if args.enable_shell else tools.shell_tools
        self.tools_dict = tools.util_get_tools_dict(forbidden_tools=forbidden)
        self.llm_tools = tools.util_get_tools(forbidden_tools=forbidden)

        # Token accounting for the whole session.
        self.token_stats = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
            "tool_calls": 0,
        }

        self.unique_id = generate_unique_id()

        # Plan-mode tracking (see plan.py). Always instantiated so the loop can
        # call it unconditionally; every method no-ops when --plan is disabled.
        self.plan_mode = bool(getattr(self.args, "plan", False))
        self.plan_manager = plan_utils.PlanManager(
            os.path.join(self.cache_dir, self.unique_id), enabled=self.plan_mode)

        # High-level recorder + history segments. Instead of a single ever-growing
        # history.json, a session keeps:
        #   * ``history_{n}.json`` files -- one per conversation *segment*; a brand
        #     new segment is started (n -> n+1) every time the history is compacted.
        #   * ``summary_history.json`` -- a flat, high-level *recorder*: one record
        #     per turn (the running turn number, *what* happened -- an assistant
        #     reply, a tool call, or a compaction summary -- and which
        #     ``history_{n}.json`` + message index the turn lives in).
        # The recorder lets the user inspect and rewind the conversation (/history,
        # /reset) without reading the full detailed histories, because every turn's
        # recorded index marks a faithful end-of-turn snapshot of its segment.
        self.history_index = 0
        self.summary = []

        # Conversation state. When --restore-session is given we continue that
        # session's history instead of starting fresh -- still under the brand
        # new unique id above, so the original session file is never
        # overwritten. Falls back to a freshly built history if the restore
        # file is missing or unreadable.
        # --load-last-session auto-detects the most recent previous session in the
        # cache dir and continues it -- exactly like --restore-session, but without
        # needing to know the folder name. An explicit --restore-session always wins
        # and is used verbatim; otherwise the newest session is resolved here.
        restore_session = getattr(self.args, "restore_session", None)
        self.loaded_last_session_dir = None
        if getattr(self.args, "load_last_session", False) and not restore_session:
            detected = find_last_session_dir(self.cache_dir)
            if detected:
                restore_session = detected
                self.loaded_last_session_dir = detected
        restore_directory = _restore_dir(restore_session)
        restored = load_history_from_dir(restore_directory) if restore_directory else None
        self.messages = restored if restored is not None else self._build_initial_messages()
        if restored is not None:
            self._restore_usage_if_available()
            self._load_summary_if_restored(restore_directory)
        else:
            # --load-last-session asked for a session but nothing usable was found:
            # don't advertise a load. The recorder starts empty and the first
            # assistant / tool-call turn seeds turn 1 -- mirroring the restore
            # path, which also has no leading system turn, so /history turn
            # numbers stay consistent whether or not we restored.
            self.loaded_last_session_dir = None

        # Snapshot the initial (or restored) messages as the first history segment
        # and persist the recorder, so a session starts with on-disk state to
        # inspect and rewind to.
        save_history(self.messages, self.args, self.unique_id, self.history_index)
        save_summary(self.summary, self.args, self.unique_id)

    def _build_initial_messages(self):
        """Assemble the system prompt and optional Agent.md into the first turns."""
        if self.args.system_prompt:
            system_prompt = self.args.system_prompt
        else:
            system_prompt = prompts.system_prompt(tools_dict=self.tools_dict, allowed_dir=self.allowed_dir, plan_mode=self.plan_mode)

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

    def _restore_usage_if_available(self):
        """Seed this session's token totals from a sibling usage.json, if the
        restored session had one, so the running summary stays continuous.

        Best-effort only: token accounting is loaded on top of (never in place
        of) the in-memory totals, and any problem here is ignored silently so a
        missing or malformed usage file can never break a restore.
        """
        restore_session = getattr(self.args, "restore_session", None)
        if not restore_session:
            return
        history_path = _resolve_history_path(restore_session)
        usage_path = os.path.join(os.path.dirname(history_path), "usage.json")
        if not os.path.isfile(usage_path):
            return
        try:
            with open(usage_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key in self.token_stats:
                if key in saved:
                    self.token_stats[key] += int(saved[key] or 0)
        except Exception:
            pass

    def _load_summary_if_restored(self, directory):
        """Restore the high-level recorder and segment index for a restore.

        Prefers the saved ``summary_history.json`` so the recorder stays flat and
        contiguous across the restore, seeding ``self.summary`` and the next
        ``history_index`` from its last record. If no recorder was saved (e.g. a
        session created by older code, or only a classic ``history.json``), we
        rebuild a minimal recorder by scanning the restored messages for
        assistant turns, so navigation still works. Best-effort only.
        """
        saved = load_summary_from_dir(directory)
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
        save_history(self.messages, self.args, self.unique_id, self.history_index)
        save_summary(self.summary, self.args, self.unique_id)
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

    def run(self, initial_prompt=None):
        """Drive the agent loop until the user quits."""
        self.print_announcements()
        print_welcome(self.allowed_dir, self.server_url, self.args,
                      self.unique_id, self.cache_dir, self.model)

        # Report the outcome of --load-last-session up front, so the user sees
        # whether we resumed an earlier conversation (under a brand-new id) or
        # started fresh because no previous session was found.
        if self.loaded_last_session_dir is not None:
            self.console.note(
                f"Loaded most recent session "
                f"({os.path.basename(self.loaded_last_session_dir)}) and continuing "
                f"it under a new session id.", "ok")
        elif getattr(self.args, "load_last_session", False):
            self.console.note(
                "No previous session found to load; starting a fresh session.", "warn")

        # Hint at startup so the high-level history feature is discoverable:
        # /history browses the recorder, /history <n> inspects a turn, and
        # /reset <n> rewinds the conversation to that turn's state.
        self.console.emit(
            self.console.style(
                "Tip: /history to browse the conversation, /history <n> to inspect a "
                "turn, /reset <n> to rewind. Type /help for all commands.",
                dim=True, fg="cyan"))

        # The --prompt value is a one-shot initial turn: it is sent verbatim and
        # then cleared, so later turns come from the user (manual mode) or are
        # produced automatically by the loop (auto mode).
        pending = initial_prompt

        while True:
            try:
                # 1) Get the user's turn: the one-shot preset, or a fresh line.
                if pending is not None:
                    self.console.rule(title="Preset prompt", char="=", fg="cyan")
                    self.console.turn("You", pending, accent="cyan")
                    user_input = pending
                    pending = None
                else:
                    user_input = self._read_user_line()
                    if user_input is None:  # quit / exit
                        self._finish_goal_and_shutdown()
                        break

                    # Interactive slash commands are handled in place and never
                    # reach the LLM. _handle_command returns True when the input
                    # was such a command, so we just re-prompt (continue).
                    if user_input.startswith("/") and self._handle_command(user_input):
                        continue

                # In plan mode, remember the user's latest instruction as the
                # active goal so the loop (and plan-mode compaction) can track it
                # even if the agent never writes it to disk itself.
                self._capture_goal(user_input)

                # 2) Hand the user turn to the model and act on its tool calls.
                self.messages.append({"role": "user", "content": user_input})
                while True:
                    should_continue = self._step()
                    self._persist()
                    if not should_continue:
                        break
                    self.console.note("Continuing with LLM calls until no more tool calls.", "info")
            except _Abort:
                # Ctrl-C during a permission prompt already shut the session
                # down; just unwind the loop quietly.
                break
            except KeyboardInterrupt:
                self.console.note("Interrupting: Mini Code Exiting...", "error")
                # Ctrl-C mid-loop just saves and re-prompts rather than crashing.
                self._persist()

    def _step(self):
        """Run a single LLM turn: call the model, render its text, run its tool
        calls. Returns True if the LLM made tool calls and we're in auto-mode,
        signalling the loop to continue without asking the user."""
        # Keep long sessions within the model's context window. Compaction
        # rolls the session over to a fresh history segment (history_index + 1)
        # and returns that bump so the recorder can stay flat across it.
        self.messages, did_compact, new_index = maybe_compact(
            self.messages, self.server_url, self.model, self.api_key,
            self.args.llm_timeout, self.token_stats, self.history_index,
        )
        if did_compact:
            self.history_index = new_index
            # In plan mode, re-inject the running plan + current goal so a
            # compaction (which would otherwise drop them) keeps the model
            # working against the goal it is still trying to finish.
            if self.plan_mode:
                self.messages = self._apply_plan_to_compacted(self.messages)
            # Snapshot the compacted messages into the new segment so its turns
            # are preserved, and record the summary as the first turn of it.
            save_history(self.messages, self.args, self.unique_id, self.history_index)
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
        # A failed LLM call returns (None, None, None): treat it as "nothing
        # happened this step" so the loop degrades gracefully (records a note
        # and re-prompts) instead of crashing.
        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                     "tool_calls": self.token_stats["tool_calls"]}
            # This shouldn't be set to None, as usage could be None, but there is still a response.
            # tool_calls = None
            # response_text = None
        # The LLM has answered (or failed); clear the spinner before rendering.
        self.console.stop_thinking()
        accumulate_usage(self.token_stats, usage)
        made_tool_calls = bool(tool_calls)
        if made_tool_calls:
            self.token_stats["tool_calls"] += len(tool_calls)

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
        usage["tool_calls"] = self.token_stats["tool_calls"]
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

    def _capture_goal(self, user_input):
        """Seed CURRENT_GOAL.md with the user's latest instruction (plan mode).

        The loop owns the goal (the user's last instruction); the agent owns the
        plan (CURRENT_PLAN.md). Overwriting on every turn keeps the goal current.
        Slash commands and the empty input are not real instructions, so they are
        ignored. No-op when --plan is off.
        """
        if not self.plan_mode or user_input is None:
            return
        stripped = user_input.strip()
        if not stripped or stripped.startswith("/"):
            return
        self.plan_manager.write_goal(user_input)

    def _finish_goal_and_shutdown(self):
        """Mark the active goal finished in CURRENT_PLAN.md, then quit/save."""
        self._mark_current_goal_finished()
        self.shutdown()

    def _mark_current_goal_finished(self):
        """Append a \"goal finished\" entry to the plan document (plan mode).

        Uses the goal the loop last recorded (self.plan_manager.goal); a no-op
        when --plan is off, so it is safe to call unconditionally.
        """
        self.plan_manager.mark_finished(self.plan_manager.goal)

    def _apply_plan_to_compacted(self, messages):
        """Re-inject the running plan + current goal after a history compaction.

        Compaction replaces the transcript with a summary, which would otherwise
        drop the agent's plan and the goal the loop seeded. We read them back off
        disk and append them as fresh user turns so the model keeps working
        against the goal it is still trying to finish. The goal is appended last
        so it reads as the most recent instruction the model should act on.
        """
        plan_text = self.plan_manager.read_plan()
        goal_text = self.plan_manager.read_goal()

        # Recover the user's latest instruction from the pre-compaction tail, in
        # case the loop had not seeded a goal file yet (e.g. the agent finished
        # everything before ever writing CURRENT_GOAL.md).
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

    def _is_plan_file(self, name, tool_params):
        """Whether a read/write targets a plan-mode file in the session dir.

        Lets the agent maintain CURRENT_PLAN.md / CURRENT_GOAL.md even though the
        session directory is usually outside --allowed-dir. Scoped tightly: plan
        mode only, only read_file / write_file, only when the resolved path is a
        plan file living directly in this session's directory.
        """
        if not self.plan_mode or name not in ("read_file", "write_file"):
            return False
        path = tool_params.get("file_path", "")
        if not path:
            return False
        try:
            resolved = os.path.realpath(path)
        except (ValueError, TypeError):
            return False
        session = os.path.realpath(os.path.join(self.cache_dir, self.unique_id))
        if os.path.dirname(resolved) != session:
            return False
        return os.path.basename(resolved) in (
            plan_utils.PLAN_FILENAME, plan_utils.GOAL_FILENAME)

    def _read_user_line(self):
        """Read one turn from the user. Returns the text to send, or None to quit."""
        user_input = self.console.read_line(style_line_prompt("You:"))
        if user_input is None:  # EOF / Ctrl-C at the prompt
            self.console.note("Thank you. Goodbye! Saving history...", "info")
            return None

        if user_input.lower() in self.EXIT_COMMANDS:
            return None

        if user_input == "/help":
            print_help(self.allowed_dir, self.server_url, self.args,
                       self.unique_id, self.cache_dir, self.model)
            # Nothing to send to the model; ask again.
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
            self.console.clear_screen()
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

    def _run_tool_calls(self, tool_calls):
        self.console.note(f"LLM is trying to call {len(tool_calls)} tool(s).", "info")
        for call in tool_calls:
            self._run_one_tool_call(call)

    def _run_one_tool_call(self, call):
        call_id = call.get("id", "")
        name = call.get("function", {}).get("name", "")
        params_str = call.get("function", {}).get("arguments", "{}")

        try:
            tool_params = json.loads(params_str)
        except json.JSONDecodeError:
            self.console.note(f"Error parsing arguments for tool {name}", "error")
            return

        if name not in self.tools_dict:
            self.console.note(f"Unknown tool: {name}, params {params_str}", "error")
            self._append_tool_result(call_id,
                f"Error: Unknown tool: {name}, params {params_str}")
            return

        # 1) Sandbox: keep every filesystem tool inside --allowed-dir (plan mode
        #    additionally lets the agent maintain CURRENT_PLAN.md /
        #    CURRENT_GOAL.md in the session directory, which normally sits
        #    outside --allowed-dir, so those writes are exempted).
        if not tool_paths_are_allowed(name, tool_params, self.allowed_dir):
            if self._is_plan_file(name, tool_params):
                pass  # allow the plan/goal files into the session directory
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
        save_history(self.messages, self.args, self.unique_id, self.history_index)
        save_summary(self.summary, self.args, self.unique_id)
        usage_path = save_usage(self.token_stats, self.cache_dir, self.unique_id)
        self.console.note(f"Saved usage to: {usage_path}", "info")

    def shutdown(self):
        """Persist and print a final summary (called on quit / exit)."""
        self._persist()
        self.console.note(
            f"Saved session to {self.cache_dir}/{self.unique_id}/ "
            f"(history_0..{self.history_index}.json + summary_history.json).", "ok")
        print_usage_summary(self.token_stats)

class _Abort(Exception):
    """Raised to unwind the permission prompt after a Ctrl-C / shutdown."""


def needs_permission(args, shell_cmd, shell_tier):
    """Whether a tool call needs a manual confirmation.

    True when either --ask-permission was passed, or the shell-safety classifier
    marked the command as risky (ASK tier) and we still need to confirm it.

    --sudo bypasses every prompt (both --ask-permission and the ASK tier), so it
    wins outright and this returns False. Destructive DENY-tier commands are
    refused earlier, independently of this function.
    """
    if getattr(args, "sudo", False):
        return False
    if getattr(args, "ask_permission", False):
        return True
    return shell_cmd is not None and shell_safety_utils.is_ask(shell_tier)


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


def main():
    args = parse_args()
    session = AgentSession(args)
    session.run(initial_prompt=args.prompt)

if __name__ == "__main__":
    main()
