"""Tests for ``runner`` -- the wall-clock guard around a tool call.

The interesting behaviour is the failure side: a hung tool must come back as a
friendly ``Error:`` string (so the model can recover) instead of stalling the
loop, and the note about it has to go to the *session's* console -- a
sub-agent's console is redirected, so reporting to the module-level one would
print a background child's errors onto the user's terminal.
"""

import io
import time
import unittest

from mini_code_cli import runner, ui


def make_console():
    """A quiet, non-verbose console writing into a buffer we can read."""
    stream = io.StringIO()
    return ui.Console(stream=stream, color=False, unicode=False, verbose=False), stream


class CallFunctionWithTimeoutTest(unittest.TestCase):

    def test_a_normal_call_returns_the_value(self):
        console, _ = make_console()
        result = runner.call_function_with_timeout(
            lambda a, b=1: a + b, {"a": 2, "b": 3}, "add", 5, console=console)
        self.assertEqual(result, 5)

    def test_a_hung_tool_times_out_with_a_friendly_error(self):
        console, stream = make_console()
        started = time.monotonic()
        result = runner.call_function_with_timeout(
            lambda: time.sleep(10), {}, "slow", 1, console=console)
        elapsed = time.monotonic() - started

        self.assertIsInstance(result, str)
        self.assertIn("timed out after 1 seconds", result)
        self.assertLess(elapsed, 5, "the wall-clock guard did not fire promptly")
        self.assertIn("timed out", stream.getvalue())

    def test_a_raising_tool_does_not_propagate(self):
        console, stream = make_console()
        result = runner.call_function_with_timeout(
            lambda: 1 / 0, {}, "boom", 5, console=console)

        self.assertEqual(result, "Error: tool boom raised an error: division by zero")
        self.assertIn("boom raised an error", stream.getvalue())

    def test_the_note_goes_to_the_passed_console_not_the_global_one(self):
        # The global console is the fallback; a call that passes its own console
        # must not write there (that is how a background sub-agent's tool errors
        # used to leak onto the user's terminal).
        global_stream = io.StringIO()
        original = ui.console
        ui.console = ui.Console(stream=global_stream, color=False,
                                unicode=False, verbose=False)
        try:
            console, stream = make_console()
            runner.call_function_with_timeout(lambda: 1 / 0, {}, "boom", 5,
                                              console=console)
        finally:
            ui.console = original

        self.assertIn("boom raised an error", stream.getvalue())
        self.assertEqual(global_stream.getvalue(), "")

    def test_without_a_console_the_module_console_is_used(self):
        # The default keeps a bare call (a script, a test) working.
        global_stream = io.StringIO()
        original = ui.console
        ui.console = ui.Console(stream=global_stream, color=False,
                                unicode=False, verbose=False)
        try:
            runner.call_function_with_timeout(lambda: 1 / 0, {}, "boom", 5)
        finally:
            ui.console = original

        self.assertIn("boom raised an error", global_stream.getvalue())


if __name__ == "__main__":
    unittest.main()
