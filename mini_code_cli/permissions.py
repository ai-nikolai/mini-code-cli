"""mini_code_cli.permissions -- pure helpers for deciding whether a tool call
needs a manual confirmation.

These are deliberately free of any I/O or terminal dependency so they can be
unit-tested in isolation. ``AgentSession`` is the only caller.
"""
from . import shell_safety_utils


def shell_forces_ask_permission(args):
    """Whether enabling the shell should force per-command permission prompts.

    Running shell commands without asking is dangerous (the agent can run any
    command and escape ``--allowed-dir``), so whenever the shell is enabled we
    force ask-permission for the whole session. Exposed as a pure function so it
    can be unit-tested independently of argparse / stdin.
    """
    return bool(getattr(args, "enable_shell", False))


def sudo_disables_prompts(args):
    """Whether ``--sudo`` was passed.

    In sudo mode the agent never stops to ask for confirmation: both the
    shell-safety ASK tier and ``--ask-permission`` are bypassed, so any
    non-destructive tool call / shell command runs unattended. Destructive
    DENY-tier commands are still refused outright -- this flag only removes the
    *prompts*, not the hard safety refusals. Pure function so it can be
    unit-tested independently of argparse / stdin.
    """
    return bool(getattr(args, "sudo", False))


def needs_permission(args, shell_cmd, shell_tier):
    """Whether a tool call needs a manual confirmation.

    True when either ``--ask-permission`` was passed, or the shell-safety
    classifier marked the command as risky (ASK tier) and we still need to
    confirm it.

    ``--sudo`` bypasses every prompt (both ``--ask-permission`` and the ASK
    tier), so it wins outright and this returns False. Destructive DENY-tier
    commands are refused earlier, independently of this function.
    """
    if sudo_disables_prompts(args):
        return False
    if getattr(args, "ask_permission", False):
        return True
    return shell_cmd is not None and shell_safety_utils.is_ask(shell_tier)
