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


def call_function_with_timeout(func, tool_params, name, timeout_seconds=DEFAULT_TOOL_TIMEOUT):
    """Call ``func(**tool_params)``, aborting after ``timeout_seconds``.

    On timeout, returns a friendly ``Error:`` string (so the agent sees the
    failure as a tool result rather than crashing). Any *other* exception the
    tool raises is allowed to propagate -- the tool authors are responsible for
    returning ``Error`` strings, and surfacing genuine bugs here is intentional.
    """
    def _handler(signum, frame):
        raise _TimedOut()

    try:
        previous = signal.signal(signal.SIGALRM, _handler)
    except (ValueError, OSError):
        # signal only works in the main thread; if we cannot arm it, just run
        # the tool without the wall-clock guard rather than failing outright.
        return func(**tool_params)

    signal.alarm(int(timeout_seconds) if timeout_seconds else 0)
    try:
        return func(**tool_params)
    except _TimedOut:
        ui.console.note(f"Function {name} timed out after {timeout_seconds} seconds.", "error")
        return f"Error: Function {name} timed out after {timeout_seconds} seconds."
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
