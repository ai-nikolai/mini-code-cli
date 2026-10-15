"""mini_code_cli.ui — a tiny, dependency-free terminal UI.

This module is responsible for everything the agent prints to the user: the
welcome banner, the user / assistant / system / tool turns, the token-usage
readouts and the interactive prompts (permission boxes, the "You:" line, ...).
It uses *only* the Python standard library.

Graceful degradation is a first-class goal:

  * When stdout is not a TTY (output piped to a file, captured by tests, ...)
    or the ``NO_COLOR`` environment variable is set, all ANSI colour is
    stripped and the unicode box-drawing characters fall back to plain ASCII,
    so piped logs stay clean, portable and greppable.
  * ``FORCE_COLOR`` / ``COLORTERM`` can be used to keep colour on even when the
    output is redirected (e.g. CI that renders colour).

The heavy lifting lives in a single :class:`Console` object that owns the
output stream plus the auto-detected ``color`` / ``unicode`` / ``width``
settings. The package uses one shared default console (``ui.console``), but the
class is trivial to instantiate elsewhere (for example with ``color=False`` in
a unit test) for full control.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

# --- ANSI SGR codes ---------------------------------------------------------
_RESET = "\033[0m"
_STYLE_CODES = {
    "bold": "1", "dim": "2", "italic": "3", "underline": "4",
}
_FG = {
    "black": 30, "red": 31, "green": 32, "yellow": 33, "blue": 34,
    "magenta": 35, "cyan": 36, "white": 37, "gray": 90, "default": 39,
}
_BG = {
    "black": 40, "red": 41, "green": 42, "yellow": 43, "blue": 44,
    "magenta": 45, "cyan": 46,
}

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# --- helpers ----------------------------------------------------------------

def _visual_len(text):
    """Length of *text* on screen, ignoring ANSI escape codes."""
    return len(_ANSI_RE.sub("", text))


def _pad_visual(text, width):
    """Right-pad *text* to *width* screen columns (ignoring ANSI codes)."""
    text = str(text)
    pad = width - _visual_len(text)
    return text + (" " * pad if pad > 0 else "")


# Glyph tables for the unicode and ascii (fallback) renderers.
_UNICODE_GLYPHS = {
    "tl": "╔", "tr": "╗", "bl": "╚", "br": "╝", "h": "═", "v": "║",
    "bullet": "•", "arrow": "↳", "info": "ℹ", "warn": "⚠",
    "error": "✖", "ok": "✔", "gear": "⚙", "role": "▸", "flag": "⚑",
}
_ASCII_GLYPHS = {
    "tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|",
    "bullet": "*", "arrow": "->", "info": "[i]", "warn": "[!]",
    "error": "[x]", "ok": "[v]", "gear": "#", "role": ">", "flag": "!",
}


class Console:
    """A small terminal renderer.

    Parameters
    ----------
    stream:
        Where to write. Defaults to ``sys.stdout``.
    color, unicode:
        Force colour / unicode on or off. ``None`` (the default) means
        auto-detect from the stream and environment.
    width:
        Force a terminal width. ``None`` auto-detects (falling back to 80).
    """

    def __init__(self, stream=None, color=None, unicode=None, width=None, verbose=True):
        self.stream = stream if stream is not None else sys.stdout
        self._color = color
        self._unicode = unicode
        self._width = width
        # verbose controls whether informational ("info") notes are printed.
        # When False, only user/assistant/tool turns and warn/error notes are
        # shown. Defaults to True so existing callers keep the full output.
        self.verbose = verbose
        # Live "transient" region used for the tool-call area in non-verbose
        # mode: content drawn here is overwritten in place (via cursor moves)
        # instead of scrolling, so the screen only keeps the final turns.
        # ``_transient_lines`` is how many screen lines the current block holds.
        self._transient_lines = 0

    # -- capability detection ----------------------------------------------
    @property
    def color_enabled(self):
        if self._color is not None:
            return self._color
        # The widespread NO_COLOR convention wins over everything.
        if os.environ.get("NO_COLOR") is not None:
            return False
        # Explicit opt-ins for environments that want colour while piped.
        if os.environ.get("FORCE_COLOR") or os.environ.get("COLORTERM"):
            return True
        try:
            return bool(self.stream.isatty())
        except (AttributeError, ValueError):
            return False

    @property
    def unicode_enabled(self):
        if self._unicode is not None:
            return self._unicode
        enc = (getattr(self.stream, "encoding", "") or "").lower()
        # Only draw box characters when we both have colour (i.e. a real
        # terminal) and a utf-8-capable stream, otherwise stick to ASCII.
        return bool(self.color_enabled and "utf" in enc)

    @property
    def width(self):
        if self._width is not None:
            return int(self._width)
        try:
            return shutil.get_terminal_size(self.stream).columns
        except (OSError, ValueError, TypeError):
            return 80

    def glyph(self, name):
        table = _UNICODE_GLYPHS if self.unicode_enabled else _ASCII_GLYPHS
        return table.get(name, "")

    # -- primitives ---------------------------------------------------------
    def style(self, text, bold=False, dim=False, italic=False,
              underline=False, fg=None, bg=None):
        """Wrap *text* in ANSI styling, or return it unchanged when colour is
        disabled (so the returned string always contains the literal text,
        which keeps it easy to test and to grep)."""
        if not self.color_enabled:
            return text
        codes = []
        if bold:
            codes.append(_STYLE_CODES["bold"])
        if dim:
            codes.append(_STYLE_CODES["dim"])
        if italic:
            codes.append(_STYLE_CODES["italic"])
        if underline:
            codes.append(_STYLE_CODES["underline"])
        if fg in _FG:
            codes.append(str(_FG[fg]))
        if bg in _BG:
            codes.append(str(_BG[bg]))
        if not codes:
            return text
        return "\033[" + ";".join(codes) + "m" + text + _RESET

    def emit(self, text=""):
        """Write a single line to the stream."""
        print(text, file=self.stream)

    def rule(self, title=None, char="═", fg="cyan"):
        """Print a horizontal rule, optionally with a centred title."""
        rule_char = char if self.color_enabled or not self.unicode_enabled else "-"
        if not title:
            line = rule_char * self.width
        else:
            seg = f" {title} "
            fill = max(self.width - _visual_len(seg), 0)
            left = fill // 2
            line = (rule_char * left) + seg + (rule_char * (fill - left))
        self.emit(self.style(line, fg=fg) if self.color_enabled else line)

    # -- transient (live, overwritten-in-place) region ----------------------
    def _can_clear_cursor(self):
        """Whether we can move/erase the cursor (a real colour, utf-8 TTY)."""
        return self.color_enabled and self.unicode_enabled

    def _clear_transient_locked(self):
        """Erase the previously emitted transient block.

        The cursor is assumed to be resting on the line just below the block
        (the normal state after :meth:`emit`). When we can move the cursor we
        walk up over the block, wipe each line and return below it, ready to
        print again. Otherwise (piped / ASCII output) we just push the block up
        with blank lines — there is no screen to clean anyway.
        """
        n = self._transient_lines
        self._transient_lines = 0
        if n <= 0:
            return
        if self._can_clear_cursor():
            self.stream.write(f"\033[{n}A")  # up to the first line of the block
            for _ in range(n):
                self.stream.write("\033[2K")  # clear the whole line
                self.stream.write("\033[1B")  # step down (ends below block)
            self.stream.flush()
        else:
            self.stream.write("\n" * n)
            self.stream.flush()

    def clear_transient(self):
        """Clear any pending transient output, leaving a clean screen."""
        self._clear_transient_locked()

    def emit_transient(self, text=""):
        """Emit *text* into the live region, replacing any previous transient
        content. Used for the tool-call area in non-verbose mode so those lines
        flash while the agent works instead of scrolling the whole transcript.
        """
        self._clear_transient_locked()
        lines = ("" if text is None else str(text)).split("\n")
        # Pad each line to the full width so a shorter follow-up fully overwrites
        # any leftover characters from a longer previous render.
        padded = [line + " " * max(0, self.width - _visual_len(line)) for line in lines]
        self.emit("\n".join(padded))
        self._transient_lines = len(lines)

    # -- building blocks ----------------------------------------------------
    def box(self, lines, title=None, fg="cyan"):
        """Render *lines* inside a bordered box and return it as a string."""
        lines = ["" if line is None else str(line) for line in lines]
        content_w = max((_visual_len(line) for line in lines), default=0)
        inner = min(max(content_w + 2, 12), max(self.width - 2, 12))
        if title is not None:
            inner = max(inner, _visual_len(title) + 4)

        tl, tr, bl, br, v, h = (self.glyph("tl"), self.glyph("tr"),
                                self.glyph("bl"), self.glyph("br"),
                                self.glyph("v"), self.glyph("h"))

        if title is None:
            top = tl + (h * inner) + tr
        else:
            seg = " " + title + " "
            fill = max(inner - _visual_len(seg), 0)
            # Split the free space roughly 1/3 left, 2/3 right so the title
            # sits wrapped by the border on both sides (e.g. ``╔══ Title ═══╗``).
            left = fill // 3
            top = tl + (h * left) + seg + (h * (fill - left)) + tr
        body = [v + " " + _pad_visual(line, inner - 2) + " " + v for line in lines]
        bottom = bl + (h * inner) + br

        if not self.color_enabled:
            return "\n".join([top] + body + [bottom])

        top_c = self.style(top, fg=fg, bold=True)
        bottom_c = self.style(bottom, fg=fg, bold=True)
        vc = self.style(v, fg=fg, bold=True)
        body_c = [vc + line[1:-1] + vc for line in body]
        return "\n".join([top_c] + body_c + [bottom_c])

    def print_box(self, lines, title=None, fg="cyan"):
        self.emit(self.box(lines, title=title, fg=fg))

    def truncate(self, text, limit):
        text = str(text)
        if len(text) > limit:
            return text[:limit] + "... (truncated)"
        return text

    @staticmethod
    def _oneline(text, limit=None):
        one = " ".join(line.strip() for line in str(text).splitlines())
        one = " ".join(one.split())
        if limit is not None and len(one) > limit:
            return one[:limit] + "..."
        return one

    # -- semantic renderers -------------------------------------------------
    def banner(self, title, lines, fg="cyan"):
        self.print_box(lines, title=title, fg=fg)

    def turn(self, role, content, accent="cyan"):
        """Render a single conversation turn (user / assistant / system)."""
        header = f"{self.glyph('role')} {role}"
        self.emit(self.style(header, fg=accent, bold=True) if accent
                  else self.style(role, bold=True))
        text = "" if content is None else str(content)
        if text == "":
            self.emit("   (no content)")
            return
        for line in text.splitlines():
            self.emit("   " + line)

    def note(self, msg, level="info"):
        """Render a one-line status message (info / ok / warn / error).

        Informational notes (the default level) are only printed in verbose
        mode; warn / error / ok notes are always shown so the user never misses
        something important in the quiet, default view.
        """
        if level == "info" and not self.verbose:
            return
        colors = {"info": "cyan", "ok": "green", "warn": "yellow", "error": "red"}
        icon = self.glyph(level)
        prefix = self.style(f"{icon} ", fg=colors.get(level, "cyan"), bold=True) \
            if self.color_enabled else f"{icon} "
        self.emit(prefix + str(msg))

    def tool_group(self, name, params=None, result=None, status="ran"):
        """Render one attempted tool call: its name, parameters and result."""
        self.emit(self.style(f"{self.glyph('gear')} {name}", fg="magenta", bold=True))
        for key, value in (params or {}).items():
            snippet = self.truncate(self._oneline(str(value), 100), 100)
            key_str = self.style(f"    {key}:", fg="gray", dim=True) if self.color_enabled \
                else f"    {key}:"
            self.emit(f"{key_str} {snippet}")
        if result is not None:
            text = str(result)
            if status == "refused":
                label = f"{self.glyph('error')} refused"
                self.emit(self.style(
                    f"    {label}: {self._oneline(text, 200)}", fg="red"))
            else:
                ok = not text.startswith("Error")
                color = "green" if ok else "red"
                first = text.splitlines()[0] if text.strip() else "(no output)"
                arrow = self.style(self.glyph("arrow"), fg=color) if self.color_enabled \
                    else self.glyph("arrow")
                self.emit(f"    {arrow} {self.truncate(first, 300)}")

    def tool_call_display(self, name, params=None, result=None, transient=True):
        """Render one attempted tool call.

        In verbose mode (``transient=False``) it is printed permanently via
        :meth:`tool_group`. Otherwise it is drawn in the live region that gets
        overwritten by the next tool call, so the transcript stays clean.
        """
        if not transient:
            self.tool_group(name, params, result)
            return
        lines = [f"{self.glyph('gear')} {name}"]
        for key, value in (params or {}).items():
            snippet = self.truncate(self._oneline(str(value), 100), 100)
            key_str = self.style(f"    {key}:", fg="gray", dim=True) if self.color_enabled \
                else f"    {key}:"
            lines.append(f"{key_str} {snippet}")
        if result is not None:
            text = str(result)
            ok = not text.startswith("Error")
            color = "green" if ok else "red"
            first = text.splitlines()[0] if text.strip() else "(no output)"
            arrow = self.style(self.glyph("arrow"), fg=color) if self.color_enabled \
                else self.glyph("arrow")
            lines.append(f"    {arrow} {self.truncate(first, 300)}")
        self.emit_transient("\n".join(lines))

    def usage_inline(self, usage):
        """A compact one-liner describing a single LLM call's token usage."""
        if not usage:
            text = "input=n/a (server did not report usage)"
        else:
            text = (f"input={usage.get('prompt_tokens', 0)} "
                    f"output={usage.get('completion_tokens', 0)} "
                    f"toolcalls={usage.get('tool_calls', 0)} "
                    f"total={usage.get('total_tokens', 0)}")
        label = self.style("tokens", fg="cyan", dim=True) if self.color_enabled else "tokens"
        return f"{label}: {text}"

    def usage_line(self, usage, transient=None):
        """Emit the one-line usage readout.

        Printed permanently in verbose mode; in non-verbose mode (or when
        ``transient`` is True) it goes into the live region that is cleared on
        the next transient emit, so per-call usage does not accumulate.
        """
        if transient is None:
            transient = not self.verbose
        self.emit_transient(self.usage_inline(usage)) if transient else self.emit(self.usage_inline(usage))

    def usage_summary(self, stats):
        """Render the accumulated token usage for the session so far (box).

        Only shown in verbose mode; call it via the console's ``verbose`` flag.
        """
        rows = [
            ("LLM calls", stats.get("llm_calls", 0)),
            ("Input tokens", stats.get("prompt_tokens", 0)),
            ("Output tokens", stats.get("completion_tokens", 0)),
            ("Total tokens", stats.get("total_tokens", 0)),
            ("Tool calls", stats.get("tool_calls", 0)),
        ]
        label_w = max(len(label) for label, _ in rows)
        self.emit(self.style("Token usage (this session):", bold=True, fg="cyan"))
        for label, value in rows:
            self.emit(f"  {label.ljust(label_w)} : {value}")

    def welcome(self, version, session_id, cache_dir, allowed_dir,
                server_url, model, permission_mode, agent_mode, shell_mode):
        """Render the startup info banner."""
        lines = [
            f"version     : v{version}",
            f"session id  : {session_id}",
            f"saved under : {cache_dir}",
            f"allowed dir : {allowed_dir}",
            f"server url  : {server_url}",
            f"model       : {model or '(auto)'}",
            "",
            f"permissions : {permission_mode}",
            f"agent mode  : {agent_mode}",
            f"shell       : {shell_mode}",
        ]
        self.banner(f"MiniCodeCLI v{version}", lines, fg="cyan")

    def announce_api_key(self, present):
        if present:
            self.note("API key found (MINI_CODE_API_KEY).", "ok")
        else:
            self.note("No API key found. Set MINI_CODE_API_KEY='sk-...' if your server needs one.",
                      "warn")

    def announce_shell(self):
        self.note("Shell execution is ENABLED. Every command is checked by the safety "
                  "classifier: destructive commands are blocked, risky ones ask for confirmation.",
                  "warn")

    # -- interactive input --------------------------------------------------
    def read_line(self, prompt_text=""):
        """Print an (already-styled) prompt and read one line from stdin.

        Returns the typed text, or ``None`` on EOF / Ctrl-C. Routing every
        interactive prompt (the "You:" line, permission answers, ...) through
        the console keeps all input handling in one place.
        """
        try:
            return input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            return None

    def ask_permission(self, name, params_str, shell_cmd=None, tier=None, reason=None):
        """Draw a permission-prompt box and read the user's answer (y / yes).

        The box *and* the input both go through the console. Any previous
        transient tool output is cleared first so the box renders cleanly.
        Returns True to proceed.
        """
        self.clear_transient()
        lines = [f"Tool       : {name}", f"Parameters : {params_str}"]
        if shell_cmd is not None:
            snippet = self._oneline(shell_cmd, 120)
            if tier == "ask":
                lines.append(f"Command    : {snippet}")
                lines.append(f"           : risky - needs your confirmation ({reason})")
            else:
                lines.append(f"Command    : {snippet}")
        self.print_box(lines, title="Permission needed", fg="yellow")
        prompt = self.style("Proceed? Type 'y' or 'yes' and press Enter: ",
                            fg="yellow", bold=True)
        answer = self.read_line(prompt)
        return (answer or "").strip().lower() in ("y", "yes")


# One shared default console used throughout the package.
console = Console(color=True, unicode=True)
