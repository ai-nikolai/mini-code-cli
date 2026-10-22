"""mini_code_cli.runner -- execute a tool function with a hard wall-clock limit.

Uses SIGALRM so a tool that hangs (a slow ``run_shell_command``, an unresponsive
website, ...) is aborted after ``timeout_seconds`` instead of stalling the whole
agent loop.
"""
import signal

from . import ui
from .config import DEFAULT_TOOL_TIMEOUT


# A *private* sentinel so a real ``TimeoutError`` raised by a tool is NOT
# mistaken for our wall-clock timeout. The old implementation redefined a local
# ``TimeoutError`` class, which shadowed the stdlib one and meant any tool that
# legitimately raised ``TimeoutError`` was silently reported as "timed out".
class _TimedOut(Exception):
    pass


def call_function_with_timeout(func, tool_params, name, timeout_seconds=DEFAULT_TOOL_TIMEOUT,
                               console=None):
    """Call ``func(**tool_params)``, aborting after ``timeout_seconds``.

    On timeout -- or if the tool raises *any* other exception -- a friendly
    ``Error:`` string is returned so the failure reaches the model as a tool
    result (letting it recover or retry) instead of propagating and crashing the
    whole agent loop. A timeout is reported distinctly from a generic failure.

    ``console`` is the session's console. It defaults to the shared one so a
    bare call still works, but the agent loop passes its own: a sub-agent's
    console redirects its output, and reporting to the module-level console
    would print a background child's tool errors onto the user's terminal.
    """
    def _handler(signum, frame):
        raise _TimedOut()

    try:
        previous = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, OSError):
        # signal only works in the main thread; if we cannot arm it, just run
        # the tool without the wall-clock guard rather than failing outright --
        # but still shield the loop from a tool that raises.
        return _call_tool_safely(func, tool_params, name, console)

    console = console if console is not None else ui.console
    signal.alarm(int(timeout_seconds) if timeout_seconds else 0)
    try:
        return func(**tool_params)
    except _TimedOut:
        console.note(f"Function {name} timed out after {timeout_seconds} seconds.", "error")
        return f"Error: Function {name} timed out after {timeout_seconds} seconds."
    except Exception as e:  # noqa: BLE001 -- a failing tool must not crash the loop
        console.note(f"Tool {name} raised an error: {e}", "error")
        return f"Error: tool {name} raised an error: {e}"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _call_tool_safely(func, tool_params, name, console=None):
    """Run a tool with no wall-clock guard (SIGALRM unavailable) but still
    shield the loop from a tool that raises.

    Used as the fallback when we could not arm SIGALRM (e.g. a non-main thread).
    """
    console = console if console is not None else ui.console
    try:
        return func(**tool_params)
    except Exception as e:  # noqa: BLE001 -- never let a tool exception propagate
        console.note(f"Tool {name} raised an error: {e}", "error")
        return f"Error: tool {name} raised an error: {e}"
