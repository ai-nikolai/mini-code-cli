"""Tests for ``permissions`` -- "does this tool call need a prompt?".

The module is deliberately pure (no I/O, no terminal), which is exactly what
makes this table cheap to pin: every combination of ``--ask-permission``,
``--sudo`` and shell-safety tier is decided here, and a regression means either
a prompt appearing when it should not (an unattended run stalls) or a prompt
disappearing when it should appear (a destructive command runs silently).
"""

import types
import unittest

from mini_code_cli import permissions, shell_safety_utils


def make_args(**overrides):
    """An argparse-shaped namespace with every flag defaulted to False."""
    base = dict(enable_shell=False, ask_permission=False, sudo=False)
    base.update(overrides)
    return types.SimpleNamespace(**base)


class NeedsPermissionTest(unittest.TestCase):
    """:func:`permissions.needs_permission` across the three inputs."""

    def test_plain_call_with_no_flags_never_needs_a_prompt(self):
        self.assertFalse(permissions.needs_permission(make_args(), None, None))

    def test_ask_permission_prompts_for_every_tool_call(self):
        args = make_args(ask_permission=True)
        self.assertTrue(permissions.needs_permission(args, None, None))
        self.assertTrue(permissions.needs_permission(
            args, "ls", shell_safety_utils.ALLOW))

    def test_an_ask_tier_shell_command_prompts_without_ask_permission(self):
        # The shell-safety classifier is the second, independent reason to ask:
        # a risky command is confirmed even when --ask-permission was not given.
        self.assertTrue(permissions.needs_permission(
            make_args(), "rm -rf build", shell_safety_utils.ASK))
        # ... but an ALLOW-tier command is not.
        self.assertFalse(permissions.needs_permission(
            make_args(), "ls -la", shell_safety_utils.ALLOW))

    def test_sudo_bypasses_every_prompt(self):
        args = make_args(sudo=True, ask_permission=True, enable_shell=True)
        self.assertFalse(permissions.needs_permission(args, None, None))
        self.assertFalse(permissions.needs_permission(
            args, "rm -rf build", shell_safety_utils.ASK))

    def test_sudo_does_not_override_the_deny_tier(self):
        # DENY is refused before any permission question is asked, so --sudo
        # "winning" here must never be read as "destructive commands are ok".
        self.assertTrue(shell_safety_utils.is_deny(shell_safety_utils.DENY))
        self.assertFalse(permissions.needs_permission(
            make_args(sudo=True), "rm -rf /", shell_safety_utils.DENY))

    def test_a_non_shell_call_is_not_treated_as_ask_tier(self):
        # ``shell_cmd is None`` means "not a shell call", whatever tier was
        # passed along with it; only a real shell command can trigger the tier.
        self.assertFalse(permissions.needs_permission(
            make_args(), None, shell_safety_utils.ASK))


class FlagHelpersTest(unittest.TestCase):

    def test_enabling_the_shell_forces_ask_permission(self):
        self.assertTrue(permissions.shell_forces_ask_permission(
            make_args(enable_shell=True)))
        self.assertFalse(permissions.shell_forces_ask_permission(make_args()))

    def test_sudo_is_reported_from_the_flag(self):
        self.assertTrue(permissions.sudo_disables_prompts(make_args(sudo=True)))
        self.assertFalse(permissions.sudo_disables_prompts(make_args()))

    def test_missing_attributes_read_as_false(self):
        # The helpers take argparse namespaces, but a caller (a sub-agent's
        # rebuilt args) may not have every attribute: that must not raise.
        empty = types.SimpleNamespace()
        self.assertFalse(permissions.shell_forces_ask_permission(empty))
        self.assertFalse(permissions.sudo_disables_prompts(empty))
        self.assertFalse(permissions.needs_permission(empty, None, None))


if __name__ == "__main__":
    unittest.main()
