"""
Usage:
    mini-code

Prompt:

write a similar model.py file but for a LSTM based language model. Make it minimal and call it model_lstm.py it should still be compatible with eval.py

"""

import os
import json
import signal

import time
import uuid

import argparse
import sys

from . import prompts
from . import shell_safety_utils
from . import tools
from . import utils
from . import ui


# ANSI escape codes for colors and bold (kept for the couple of ad-hoc spots
# that still build their own strings; the rest of the package renders through
# the ui.Console in ui.py).
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

DEFAULT_CACHE_LOCATION=os.path.expanduser("~/.cache/mini_code/sessions/")
DEFAULT_MODEL_URL="http://0.0.0.0:30000"
DEFAULT_TEMPERATURE=1.0
DEFAULT_TOOL_TIMEOUT=120  # seconds before a single tool call is aborted
DEFAULT_HTTP_TIMEOUT=500  # seconds before an LLM HTTP request is aborted
DEFAULT_MAX_MESSAGES=150  # messages in history before it is compacted to a summary

def save_history(messages, args, session_name=""):
    # Save messages to history.json
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = DEFAULT_CACHE_LOCATION
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cache_dir,session_name), exist_ok=True)

    history_path = os.path.join(cache_dir, session_name, f"history.json")
    with open(history_path, 'w') as f:
        json.dump(messages, f, indent=2)


def accumulate_usage(token_stats, usage):
    """Add a single LLM call's token usage into the running session totals."""
    if not usage:
        return
    token_stats["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
    token_stats["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
    token_stats["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    token_stats["llm_calls"] += 1


def format_usage(usage):
    """Human-readable one-line summary of a single call's token usage."""
    if not usage:
        return "tokens: n/a (server did not report usage)"
    return (f"tokens: input={usage.get('prompt_tokens', 0)} "
            f"output={usage.get('completion_tokens', 0)} "
            f"total={usage.get('total_tokens', 0)}")


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
        print(f"{BOLD}{YELLOW}Warning:{RESET} No history found to restore from: {restore_session}")
        return None

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list):
            raise ValueError("history file did not contain a message list")
    except Exception as e:
        print(f"{BOLD}{RED}Warning:{RESET} Failed to load history from {restore_session}: {e}")
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
    ui.console.note(help_message, "info")
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


def maybe_compact(messages, server_url, model, api_key, llm_timeout, token_stats, max_messages=DEFAULT_MAX_MESSAGES):
    """Compact the conversation history once it grows beyond `max_messages`.

    Long (especially auto-mode) sessions can overflow the model's context window.
    When the history gets too large we ask the LLM to summarize it and replace the
    old turns with that summary, keeping the original system prompt as the first
    message. The compaction call itself is a single, tool-free request, so it
    cannot trigger further tool calls or recurse into compaction.
    """
    if len(messages) <= max_messages:
        return messages

    try:
        print(f"{BOLD}{YELLOW}System:{RESET} Compacting conversation history ({len(messages)} messages -> summary)...")
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
            print(f"{BOLD}{YELLOW}System:{RESET} History compacted to {len(compacted)} messages.")
            return compacted
    except Exception as e:
        print(f"{BOLD}{RED}Warning:{RESET} Failed to compact history: {e}")

    return messages


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

        # Conversation state. When --restore-session is given we continue that
        # session's history instead of starting fresh -- still under the brand
        # new unique id above, so the original session file is never
        # overwritten. Falls back to a freshly built history if the restore
        # file is missing or unreadable.
        restored = load_history(getattr(self.args, "restore_session", None))
        self.messages = restored if restored is not None else self._build_initial_messages()
        if restored is not None:
            self._restore_usage_if_available()

    def _build_initial_messages(self):
        """Assemble the system prompt and optional Agent.md into the first turns."""
        if self.args.system_prompt:
            system_prompt = self.args.system_prompt
        else:
            system_prompt = prompts.system_prompt(tools_dict=self.tools_dict, allowed_dir=self.allowed_dir)

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

        # The --prompt value is a one-shot initial turn: it is sent verbatim and
        # then cleared, so later turns come from the user (manual mode) or are
        # produced automatically by the loop (auto mode).
        pending = initial_prompt

        while True:
            try:
                # 1) Get the user's turn: the one-shot preset, or a fresh line.
                if pending is not None:
                    print("=" * 30)
                    print(f"{BOLD}{CYAN}You (preset): {RESET}{pending}")
                    user_input = pending
                    pending = None
                else:
                    user_input = self._read_user_line()
                    if user_input is None:  # quit / exit
                        self.shutdown()
                        break

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
        # Keep long sessions within the model's context window.
        self.messages = maybe_compact(
            self.messages, self.server_url, self.model, self.api_key,
            self.args.llm_timeout, self.token_stats,
        )

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
        accumulate_usage(self.token_stats, usage)
        self.console.usage_line(usage, transient=not self.console.verbose)

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

        return made_tool_calls and self.args.auto_mode

    def _read_user_line(self):
        """Read one turn from the user. Returns the text to send, or None to quit."""
        user_input = self.console.read_line(style_line_prompt("You:"))
        if user_input is None:  # EOF / Ctrl-C at the prompt
            print("\nThank you. Goodbye! Saving history...")
            return None

        if user_input.lower() in self.EXIT_COMMANDS:
            return None

        if user_input == "/help":
            print_help(self.allowed_dir, self.server_url, self.args,
                       self.unique_id, self.cache_dir, self.model)
            # Nothing to send to the model; ask again.
            return self._read_user_line()

        return user_input

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

        # 1) Sandbox: keep every filesystem tool inside --allowed-dir.
        if not tool_paths_are_allowed(name, tool_params, self.allowed_dir):
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
            print("\nThank you. Goodbye! Saving history...")
            self.shutdown()
            raise _Abort

    def _append_tool_result(self, call_id, content):
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": str(content)})

    def _print_assistant(self, text):
        self.console.turn("Assistant", text, accent="green")

    def _persist(self):
        save_history(self.messages, self.args, self.unique_id)
        usage_path = save_usage(self.token_stats, self.cache_dir, self.unique_id)
        self.console.note(f"Saved usage to: {usage_path}", "info")
        print_usage_summary(self.token_stats)

    def shutdown(self):
        """Persist and print a final summary (called on quit / exit)."""
        self.console.note(f"Saving history to: {self.cache_dir}/history_{self.unique_id}.json", "info")
        self._persist()


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
