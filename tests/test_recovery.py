"""Tests for the runtime recovery helpers and the settings that expose them."""
import unittest

from mini_code_cli import recovery


class ToolFailureTest(unittest.TestCase):
    def test_clear_failures_are_detected(self):
        self.assertTrue(recovery.tool_result_failed("Error: no such file: x"))
        self.assertTrue(recovery.tool_result_failed(
            "some output\nEXIT CODE: 1"))
        self.assertTrue(recovery.tool_result_failed(
            "Command produced no output (exit code 2)."))
        self.assertTrue(recovery.tool_result_failed(
            "Error: command timed out after 30 seconds: 'sleep'"))

    def test_success_and_benign_text_are_not_failures(self):
        self.assertFalse(recovery.tool_result_failed(""))
        self.assertFalse(recovery.tool_result_failed("read 12 lines"))
        self.assertFalse(recovery.tool_result_failed(
            "matched: this line mentions exit code: 1 in a log"))
        self.assertFalse(recovery.tool_result_failed(
            "Command produced no output (exit code 0)."))
        self.assertFalse(recovery.tool_result_failed(
            "the word Error appears in this file, not as the result"))

    def test_approach_key_groups_the_same_attempt(self):
        a = recovery.approach_key("run_shell_command",
                                  {"command": "pytest   -q"})
        b = recovery.approach_key("run_shell_command",
                                  {"command": "pytest -q"})
        c = recovery.approach_key("run_shell_command",
                                  {"command": "python solve.py"})
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(
            recovery.approach_key("read_file", {"file_path": "/app/x.py"}),
            recovery.approach_key("read_file", {"file_path": "/app/x.py",
                                                "offset": 3}))
        self.assertEqual(
            recovery.approach_key("edit", {"file_path": "/app/x.py",
                                           "old_string": "a" * 1000}),
            recovery.approach_key("edit", {"file_path": "/app/x.py",
                                           "old_string": "b" * 1000}))

    def test_recovery_prompt_names_the_tool_and_attempts(self):
        text = recovery.recovery_prompt("run_shell_command", 3,
                                        same_call=True,
                                        result="Error: nope")
        self.assertIn("run_shell_command", text)
        self.assertIn("3 times", text)
        self.assertIn("Rethink your approach", text)
        self.assertIn("Last failure", text)
        self.assertNotIn("documentation or the web", text)

    def test_search_hint_is_only_added_when_search_is_available(self):
        without = recovery.recovery_prompt("run_shell_command", 3)
        with_search = recovery.recovery_prompt("run_shell_command", 3,
                                               can_search=True)
        self.assertNotIn("documentation or the web", without)
        self.assertIn("documentation or the web", with_search)

    def test_unknown_strategy_falls_back_to_normal(self):
        self.assertEqual(recovery.normalise("nope"), recovery.NORMAL)
        self.assertEqual(recovery.normalise(recovery.PROMPT_CHANGE),
                         recovery.PROMPT_CHANGE)


if __name__ == "__main__":
    unittest.main()
