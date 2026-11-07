"""mini_code_cli.permissions -- pure helpers for deciding whether a tool call
needs a manual confirmation.

These are deliberately free of any I/O or terminal dependency so they can be
unit-tested in isolation. ``AgentSession`` is the only caller.
"""
from . import shell_safety_utils


def shell_execution_enabled(args):
    """Whether shell execution was requested (``-s`` / ``--enable-shell``).

    This says the shell tool is *available*. It does not mean every command
    prompts: shell-safety decides that per command (ASK tier), ``--ask-permission``
    can ask for everything, and ``--sudo`` removes the prompts. The old name
    (``shell_forces_ask_permission``) claimed otherwise; it is kept as an alias
    so external callers do not break.
    """
    return bool(getattr(args, "enable_shell", False))


#: Backwards-compatible alias for the historical (misleading) name.
shell_forces_ask_permission = shell_execution_enabled


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

    True when either ``--ask-permission`` was passed (that asks before *every*
    tool call, including an ALLOW-tier shell command), or the shell-safety
    classifier marked the command as risky (ASK tier) and we still need to
    confirm it. With no flags, safe (ALLOW-tier) commands run unaided.

    ``--sudo`` bypasses every prompt (both ``--ask-permission`` and the ASK
    tier), so it wins outright and this returns False. Destructive DENY-tier
    commands are refused earlier, independently of this function.
    """
    if sudo_disables_prompts(args):
        return False
    if getattr(args, "ask_permission", False):
        return True
    return shell_cmd is not None and shell_safety_utils.is_ask(shell_tier)
