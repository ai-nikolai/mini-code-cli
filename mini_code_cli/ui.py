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
import signal
import sys
import textwrap
import threading
import unicodedata
from collections import deque

from .utils import oneline, plural

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
    """Width of *text* in terminal columns, ignoring ANSI escape codes.

    Wide East-Asian glyphs and emoji take two columns; combining/format marks
    take none. This keeps boxes aligned for CJK text and emoji instead of
    silently shifting every border by one column.
    """
    total = 0
    for char in _ANSI_RE.sub("", str(text)):
        if unicodedata.category(char) in ("Mn", "Me", "Cf"):
            continue
        if unicodedata.east_asian_width(char) in ("W", "F"):
            total += 2
        elif 0x1F000 <= ord(char) <= 0x1FAFF:
            total += 2
        else:
            total += 1
    return total


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
    # Context-usage bar (see Console.context_bar): filled / empty cells.
    "bar_full": "█", "bar_empty": "░",
}
_ASCII_GLYPHS = {
    "tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|",
    "bullet": "*", "arrow": "->", "info": "[i]", "warn": "[!]",
    "error": "[x]", "ok": "[v]", "gear": "#", "role": ">", "flag": "!",
    "bar_full": "#", "bar_empty": "-",
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

    def __init__(self, stream=None, color=None, unicode=None, width=None, verbose=True, parent_console=None):
        self.stream = stream if stream is not None else sys.stdout
        # ``parent_console`` lets a swallowed console (a sub-agent's own, whose
        # output is redirected away from the screen) route its *interactive
        # input* (the "You:" line, permission answers, ...) to the console that
        # actually owns the terminal -- the human-facing one. It is None for a
        # normal console, in which case input is read directly below.
        self.parent_console = parent_console
        # Bytes read by the multi-line editor but not consumed (the LF of a
        # CRLF, or a key that arrived right after Enter). It lives on the
        # console so it survives into the next read_line call.
        self._input_pending = []
        # TUI mode's up/down history. It lives on the console so it survives
        # across prompts for the whole session.
        self._input_history = []
        # Recent tool calls, for /expand in the fixed TUI (and normal mode).
        self._tool_history = deque(maxlen=20)
        self._color = color
        self._unicode = unicode
        self._width = width
        # verbose controls whether informational ("info") notes are printed.
        # When False, only user/assistant/tool turns and warn/error notes are
        # shown. Defaults to True so existing callers keep the full output.
        self.verbose = verbose
        # "… thinking" spinner state (see start_thinking / stop_thinking).
        self._thinking_thread = None
        self._thinking_stop = None
        # "… calling tool" spinner state (see start_tool_call / stop_tool_call).
        # Mirrors the thinking spinner: a daemon thread that rewrites a single
        # live line in place (via the transient region) so the tool-execution
        # phase looks alive without scrolling the transcript.
        self._tool_call_thread = None
        self._tool_call_stop = None
        # Live "transient" region (non-verbose mode) renders the tool-call area
        # and the usage line without scrolling the screen. We keep a single
        # invariant: whenever ``_transient_lines`` is > 0 a block of that many
        # lines sits at the *bottom* of the screen and the cursor rests right
        # below it. That lets us update it in place (move up, rewrite, or delete
        # the block with the standard "Delete Line" escape) so the transcript
        # never grows while the agent is mid-turn.
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
        """Write a single line to the stream.

        This is the one place every line of output is written, so it is also the
        *commit point* for the transient region: if a transient block is
        currently live (the tool-call area / usage line sitting at the bottom of
        the screen in non-verbose mode), any permanent output here must first
        erase that block in place -- otherwise the block would be stranded above
        the new content and the next update could not clear it correctly (that
        was exactly what made the screen keep scrolling).

        We erase it with the standard CSI-n-M (Delete Line) escape, which shifts
        the screen up to fill the gap. Crucially we do that *without* moving the
        cursor, so the text we are about to print lands right where the transient
        block used to be (the block is overwritten in place, not below it). This
        keeps the transient block permanently pinned to the bottom of the
        screen -- the only place it can be updated without scrolling.
        """
        if self._transient_lines > 0:
            self._delete_transient_block()
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
        """Whether we can move/erase the cursor (a real colour, utf-8 TTY).

        Only on such a terminal can we redraw the transient region in place;
        otherwise (piped / ASCII output) we fall back to plain scrolling emit.
        """
        return self.color_enabled and self.unicode_enabled

    def _delete_transient_block(self):
        """Erase the current transient block from the screen.

        Uses the standard CSI n M (Delete Line, DL) escape, which removes n
        lines and shifts everything below them up to fill the gap -- no blank
        lines, no scroll, no leftover.

        The cursor is always assumed to be sitting on the blank line directly
        below the block (the normal resting state right after an :meth:`emit` of
        an n-line block). To put the cursor on the block's *first* line, where
        DL acts, we move up n lines first; DL then deletes the block in place
        and leaves the cursor on what was the first line below it, exactly where
        the next :meth:`emit` should land. Only called while a block is live.
        """
        n = self._transient_lines
        if n > 0:
            self.stream.write(f"\033[{n}A")   # move up onto the block's first line
            self.stream.write(f"\033[{n}M")   # CSI n M = delete those n lines
            self.stream.flush()
        self._transient_lines = 0

    def clear_transient(self):
        """Erase any pending transient output, leaving a clean screen."""
        if self._can_clear_cursor() and self._transient_lines > 0:
            self._delete_transient_block()

    def _render_transient_block(self, text=""):
        """Render *text* into the live region, replacing the current block in
        place. Assumes the cursor already sits just below an existing block (or
        there is none) and that nothing permanent follows it.

        The new block is printed at the same spot and its height recorded so the
        next update can delete exactly that many lines. Each line is padded to
        the full width so a shorter follow-up fully overwrites any characters
        left behind by a longer previous render.
        """
        lines = ("" if text is None else str(text)).split("\n")
        padded = [line + " " * max(0, self.width - _visual_len(line)) for line in lines]
        if self._can_clear_cursor():
            self._delete_transient_block()
            self.emit("\n".join(padded))
            self._transient_lines = len(lines)
        else:
            self.emit("\n".join(padded))

    def emit_transient(self, text=""):
        """Render *text* into the live region, replacing the previous transient
        block in place so the screen does not scroll. Used for the tool-call
        area and the per-call usage line in non-verbose mode: those flash while
        the agent is mid-turn and disappear -- without leaving a growing
        transcript -- as soon as the next real (assistant / user) turn is drawn.

        On a real colour, utf-8 terminal the block is deleted and reprinted at
        the same spot (via the Delete-Line escape), so the transcript stays
        pinned to a fixed height. On piped / ASCII output we cannot move the
        cursor, so we simply emit the freshly rendered block (it scrolls like
        any other line, but nothing is lost or mangled).
        """
        self._render_transient_block(text)

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

    def question_box(self, title, lines):
        """Render a prominent question/confirmation box (yellow/orange)."""
        self.print_box(lines, title=title, fg="yellow")

    def plan_box(self, lines, title="Plan for the current goal"):
        """Render a generated plan (the fixed TUI appends it to the core view)."""
        self.status_box(lines, title=title, fg="cyan", dim=True)

    def status_box(self, lines, title=None, fg="cyan", dim=False):
        """Render a bordered box of free-form *status* lines (used for /plan).

        Unlike :meth:`turn` (a conversation turn) this renders the loop-owned
        tracking files as a single, glanceable box: the goal and the current
        plan side by side, with the last completion review under a fixed
        section. Each line is shown verbatim (so a numbered markdown plan stays
        a numbered list). ``dim`` renders the lines dimmed so the box reads as
        secondary, reference content rather than a live turn. Best-effort: a
        blank line is rendered as an empty cell so plan sections stay visible.
        """
        styled_lines = []
        for line in lines:
            text = "" if line is None else str(line)
            if dim:
                styled_lines.append(self.style(text, dim=True))
            else:
                styled_lines.append(text)
        self.print_box(styled_lines, title=title, fg=fg)

    @staticmethod
    def _oneline(text, limit=None):
        """Collapse ``text`` to one line (delegates to :func:`utils.oneline`)."""
        return oneline(text, limit)

    # -- semantic renderers -------------------------------------------------
    def banner(self, title, lines, fg="cyan"):
        self.print_box(lines, title=title, fg=fg)

    def turn(self, role, content, accent="cyan", badge=None):
        """Render a single conversation turn (user / assistant / system).

        ``badge`` is an optional, already-rendered string shown on the header
        line's right (after the role) -- used for the context-occupancy bar, so
        "how full is the window" sits next to the turn that was just produced
        rather than on a separate line. It is emitted verbatim (the caller owns
        any styling), and nothing is drawn when it is empty.
        """
        header = f"{self.glyph('role')} {role}"
        rendered = self.style(header, fg=accent, bold=True) if accent \
            else self.style(role, bold=True)
        if badge:
            rendered = f"{rendered}  {badge}"
        self.emit(rendered)
        text = "" if content is None else str(content)
        if text == "":
            self.emit("   (no content)")
            return
        for line in text.splitlines():
            self.emit("   " + line)

    def context_bar(self, fraction, width=10, warn=0.60, danger=0.85):
        """Render a small "how full is the context window" bar.

        ``fraction`` is the share of the window currently in use (0.0-1.0;
        anything outside is clamped, so a rough estimate can never produce a
        broken bar). Returns e.g. ``█████░░░░░ 48%`` -- or ``#####----- 48%``
        on a non-unicode (piped / ASCII) console, consistent with every other
        glyph in this module.

        The fill is coloured by pressure: green below ``warn``, amber up to
        ``danger``, red from there on. The caller passes those two thresholds as
        fractions of the window so they can line up with whatever *actually*
        happens at high occupancy (compaction), instead of a fixed percentage
        that may sit far away from it.

        Returns "" for an unusable ``fraction`` (``None``, or a non-number) so
        the caller can drop it into a header unconditionally.
        """
        try:
            frac = float(fraction)
        except (TypeError, ValueError):
            return ""
        frac = min(max(frac, 0.0), 1.0)
        pct = int(round(frac * 100))
        filled = min(max(int(round(frac * width)), 0), width)
        bar = self.glyph("bar_full") * filled + self.glyph("bar_empty") * (width - filled)
        fg = "green" if frac < warn else ("yellow" if frac < danger else "red")
        # The percentage is left unstyled and outside the colour span so the
        # digits stay legible on any terminal theme.
        return f"{self.style(bar, fg=fg)} {pct}%"

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

    # -- conversation flow --------------------------------------------------
    def divider(self):
        """Render a thin horizontal delimiter line.

        Used to separate consecutive conversation turns so each agent reply is
        visually boxed off from the next one.
        """
        self.emit(self.style("-" * self.width, fg="gray", dim=True))

    def short_divider(self, char="-", width=8, fg="gray"):
        """Render a short horizontal delimiter line (not the full width).

        Used to set off the user's own "You:" turn from the preceding assistant
        reply with a light, unobtrusive rule rather than a full-width divider.
        Defaults to a plain ASCII dash so it renders consistently regardless of
        the console's unicode setting.
        """
        self.emit(self.style(char * width, fg=fg, dim=True))

    def clear_screen(self):
        """Clear the terminal and move the cursor to the top-left.

        Implements the /clear command: it wipes the current view but keeps the
        session history intact, so subsequent turns continue from where the
        conversation is now. The full-screen TUI overrides this to preserve its
        fixed dashboard.
        """
        self.clear_transient()
        self.stream.write("\033[2J\033[H")
        self.stream.flush()

    def start_thinking(self, message="Thinking"):
        """Start a background "… thinking" spinner (non-scrolling).

        Runs a short animation in a daemon thread that rewrites a single live
        line in place via the transient region, so the agent appears to be
        working while the LLM is being called. On a real colour/utf-8 TTY it
        animates in place (no scrolling); on piped / ASCII output it simply
        prints one static "… thinking" line and starts no thread. Stop it with
        :meth:`stop_thinking` before drawing the next real turn.
        """
        self.stop_thinking()  # be defensive: never leak more than one spinner
        if not self._can_clear_cursor():
            self.emit(self.style(f"… {message}", fg="cyan", dim=True))
            return
        self._thinking_stop = threading.Event()
        self._thinking_thread = threading.Thread(
            target=self._spin, args=(message,), daemon=True)
        self._thinking_thread.start()

    def _spin(self, message):
        """Animation loop: rewrite the single transient line every frame.

        Reuses the transient-region machinery (:meth:`emit_transient`) so the
        cursor math (move up, overwrite, no scroll) is handled for us and stays
        consistent with how tool-call / usage transient output is cleared. The
        leading ellipsis dots grow with the frame counter to make the wait feel
        dynamic (… then … … then … … …, and back to …).
        """
        stop = self._thinking_stop
        i = 0
        while not stop.is_set():
            # Grow the leading dots: 1, 2, then 3, and hold.
            dots = "… " * min((i % 3) + 1, 3)
            i += 1
            self.emit_transient(self.style(f"{dots}{message}", fg="cyan", dim=True))
            stop.wait(0.3)
        # Replace the spinner line with a blank so the following real output
        # lands cleanly. The blank stays tracked as the transient block and is
        # cleared by the next :meth:`emit` (e.g. the assistant turn's divider).
        self.emit_transient(None)

    def stop_thinking(self):
        """Stop the "… thinking" spinner started by :meth:`start_thinking`.

        Signals the animation thread to finish its current frame and exit, then
        joins it, so no writes race with the main thread's subsequent output.
        Safe to call when no spinner is running (a no-op)."""
        stop = self._thinking_stop
        thread = self._thinking_thread
        self._thinking_stop = None
        self._thinking_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def start_tool_call(self, message="calling tool", tool_name=None):
        """Start a background "… calling tool" spinner (non-scrolling).

        Shown while the agent executes a tool call, mirroring
        :meth:`start_thinking` which covers the LLM request. Runs a short
        animation in a daemon thread that rewrites a single live line in place
        via the transient region, so the wait looks alive without scrolling the
        transcript. On a real colour/utf-8 TTY it animates in place; on piped /
        ASCII output it simply prints one static line and starts no thread.
        Stop it with :meth:`stop_tool_call` before drawing the next turn.
        """
        self.stop_tool_call()  # be defensive: never leak more than one spinner
        if not self._can_clear_cursor():
            self.emit(self.style(f"\u2026 {message}:{tool_name}", fg="magenta", dim=True))
            return
        self._tool_call_stop = threading.Event()
        self._tool_call_thread = threading.Thread(
            target=self._spin_tool_call, args=(message,), daemon=True)
        self._tool_call_thread.start()

    def _spin_tool_call(self, message):
        """Animation loop for the tool-call spinner (mirror of :meth:`_spin`)."""
        stop = self._tool_call_stop
        i = 0
        while not stop.is_set():
            dots = "\u2026 " * min((i % 3) + 1, 3)
            i += 1
            self.emit_transient(self.style(f"{dots}{message}", fg="magenta", dim=True))
            stop.wait(0.3)
        # Replace the spinner line with a blank so the following real output
        # lands cleanly; the blank stays tracked as the transient block and is
        # cleared by the next :meth:`emit` (e.g. the next assistant turn).
        self.emit_transient(None)

    def stop_tool_call(self):
        """Stop the "… calling tool" spinner started by :meth:`start_tool_call`.

        Signals the animation thread to finish its current frame and exit, then
        joins it, so no writes race with the main thread's subsequent output.
        Safe to call when no spinner is running (a no-op)."""
        stop = self._tool_call_stop
        thread = self._tool_call_thread
        self._tool_call_stop = None
        self._tool_call_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def tool_group(self, name, params=None, result=None, status="ran"):
        """Render one attempted tool call: its name, parameters and result."""
        self.emit(self.style(f"{self.glyph('gear')} {name}", fg="magenta", bold=True))
        for key, value in (params or {}).items():
            snippet = self._oneline(str(value), 100)
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
                self.emit(f"    {arrow} {self._oneline(first, 300)}")

    def tool_call_display(self, name, params=None, result=None, transient=True):
        """Render one attempted tool call.

        In verbose mode (``transient=False``) the full detail (name, parameters
        and result) is printed permanently via :meth:`tool_group`. In the quiet,
        default (non-verbose) mode we instead leave a small *persistent* summary
        in the transcript -- the yellow "tool" label with the tool name on the
        line below -- rather than flashing the call in the live region and then
        clearing it away completely.
        """
        self._tool_history.append((name, params, result))
        if not transient:
            self.tool_group(name, params, result)
            return
        label = self.style("tool", fg="yellow", bold=True) if self.color_enabled else "tool"
        self.emit(label)
        self.emit(f"    {name}: {str(params)[:35]}")

    def expand_tool(self, index=1):
        """Print the full detail of a recent tool call (``/expand [n]``)."""
        try:
            index = max(int(index), 1)
        except (TypeError, ValueError):
            index = 1
        if not self._tool_history:
            self.note("No tool calls have run yet.", "warn")
            return
        if index > len(self._tool_history):
            self.note(
                f"No tool call {index} back; only {len(self._tool_history)} "
                f"are remembered.", "warn")
            return
        name, params, result = self._tool_history[-index]
        self.rule(title=f"tool call {index} back", char="═", fg="cyan")
        self.tool_group(name, params, result)

    def usage_inline(self, usage, llm_calls=None):
        """A compact one-liner describing token usage.

        ``usage`` is a single call's numbers. ``llm_calls``, when given, is the
        session-wide number of LLM calls made so far and is appended as a
        trailing ``| 12 LLM calls`` field -- the per-call figures alone cannot
        say how much work the session has actually done (a turn can cost one
        call or a dozen, and compaction / planning calls are invisible in the
        token totals once their segment is closed).
        """
        if not usage:
            text = "input=n/a (server did not report usage)"
        else:
            text = (f"input={usage.get('prompt_tokens', 0)} "
                    f"output={usage.get('completion_tokens', 0)} "
                    f"toolcalls={usage.get('tool_calls', 0)} "
                    f"total={usage.get('total_tokens', 0)}")
        if llm_calls is not None:
            calls = int(llm_calls)
            text += f" | {plural(calls, 'LLM call')}"
        label = self.style("tokens", fg="cyan", dim=True) if self.color_enabled else "tokens"
        return f"{label}: {text}"

    def usage_line(self, usage, transient=None, llm_calls=None):
        """Emit the one-line usage readout.

        Printed permanently in verbose mode; in non-verbose mode (or when
        ``transient`` is True) it goes into the live region that is cleared on
        the next transient emit, so per-call usage does not accumulate.
        ``llm_calls`` is passed through to :meth:`usage_inline`.
        """
        if transient is None:
            transient = not self.verbose
        line = self.usage_inline(usage, llm_calls=llm_calls)
        self.emit_transient(line) if transient else self.emit(line)

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

    def history_list(self, rows, title="Conversation history", current=None):
        """Render the high-level recorder as a numbered list of turns.

        Each row is (turn number, what happened, location). The recorder is a
        flat, high-level log (see AgentSession), so this is a quick way to see
        the whole conversation at a glance and pick a turn to inspect or rewind
        to. The location column shows which ``history_{n}.json`` segment and
        message index the turn lives in, i.e. the end-of-turn snapshot a reset
        restores from.
        """
        # Each row is a 3-tuple (turn, what, location); render every cell as a
        # string for display while keeping the tuple shape so it can be unpacked
        # below. (A None row, if any, is dropped so a stray empty entry never
        # crashes the width maths or the unpacking.)
        rows = [tuple("" if c is None else str(c) for c in r) for r in rows]
        turn_w = max((len(r[0]) for r in rows), default=0)
        what_w = max((len(r[1]) for r in rows), default=0)
        total = len(rows)
        header = f"{title} ({total} turns)"
        if current:
            header += f" — on turn {current} of {total}"
        header += ":"
        self.emit(self.style(header, bold=True, fg="cyan"))
        for turn, what, where in rows:
            turn_cell = self.style(turn.rjust(turn_w), fg="magenta", bold=True) if self.color_enabled else turn.rjust(turn_w)
            what_cell = self.style(self._oneline(what, what_w), dim=True) if self.color_enabled else what.ljust(what_w)
            where_cell = self.style(where, fg="gray", dim=True) if self.color_enabled else where
            self.emit(f"{turn_cell}  {what_cell}  {where_cell}")
        self.emit(self.style("(commands: /history <n> to inspect, /reset <n> to rewind)", dim=True, fg="cyan"))

    def history_detail(self, record, messages, total=None):
        """Render a single turn's full recorded messages.

        Used by ``/history <n>`` to show exactly what the conversation looked
        like at the end of that turn. Each message is boxed by role, and the
        tool-call markers are surfaced so the detail mirrors the transcript the
        user saw at the time.
        """
        turn = record.get("turn", "?")
        what = record.get("what", "?")
        where = f"history_{record.get('history_file', 0)}.json at index {record.get('index', 0)}"
        heading = f"Turn {turn}"
        if total:
            heading += f" of {total}"
        heading += f" ({what}) — {where}"
        self.emit(self.style(heading, bold=True, fg="cyan"))
        self.emit(self.style("-" * self.width, fg="gray", dim=True))
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            self.emit(self.style(f"{self.glyph('role')} {role}", fg="magenta", bold=True))
            if role == "tool":
                self.emit(f"    tool result for id {msg.get('tool_call_id', '')}: {self._oneline(content, 200)}")
            elif content:
                for line in str(content).splitlines():
                    self.emit(f"    {line}")
            else:
                self.emit("    (no content)")
            calls = msg.get("tool_calls")
            if calls:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in calls)
                self.emit(self.style(f"    {self.glyph('gear')} tool call(s): {names}", fg="yellow"))
        self.emit(self.style("Type /reset %s to rewind here." % turn, dim=True, fg="cyan"))

    def welcome(self, version, session_id, cache_dir, allowed_dir,
                server_url, model, permission_mode, agent_mode, shell_mode,
                compaction_strategy=None, plan_strategy=None, plan_mode=False):
        """Render the startup info banner."""
        planning = plan_strategy or "plan-verify-step"
        if not plan_mode:
            planning += " (plan mode off)"
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
            f"compaction  : {compaction_strategy or 'summarise'}",
            f"planning    : {planning}",
        ]
        self.banner(f"MiniCodeCLI v{version}", lines, fg="cyan")

    def farewell(self, version, rows, sign_off=None):
        """Render the exit banner: a sign-off plus a summary of the session state.

        Deliberately narrower than :meth:`welcome`. Everything the welcome banner
        shows is about how the session *will* run -- server, model, permissions,
        allowed dir -- and all of it is irrelevant once the session is over. What
        the user can still act on is what the session *left behind*: which
        folder it was saved in, which history segments exist, how much was said
        and what it cost. So the box carries only the version, a good-bye line
        and those ``rows``.

        ``rows`` is a list of ``(label, value)`` pairs, laid out in a single
        colon-aligned column like the welcome banner; the caller supplies the
        pairs because it owns the state (the console stays free of session
        policy).
        """
        label_w = max((len(str(label)) for label, _ in rows), default=0)
        lines = [sign_off or "Goodbye, and thanks for using MiniCodeCLI!", ""]
        lines += [f"{str(label).ljust(label_w)} : {value}" for label, value in rows]
        # Yellow, not the welcome banner's cyan: this is the one box that
        # *closes* the session, so it reads as a distinct, valedictory note
        # rather than the startup banner reappearing. Colours only apply on a
        # colour-capable TTY (see :meth:`box`), so piped/ASCII output is
        # unchanged.
        self.banner(f"MiniCodeCLI v{version} - goodbye", lines, fg="yellow")

    def announce_api_key(self, present):
        if present:
            self.note("API key found (MINI_CODE_API_KEY).", "ok")
        else:
            self.note("No API key found. Set MINI_CODE_API_KEY='sk-...' if your server needs one.",
                      "warn")

    def announce_shell(self, prompts_disabled=False):
        if prompts_disabled:
            self.note("Shell execution is ENABLED. Every command is checked by the "
                      "safety classifier: destructive commands are blocked, but no "
                      "confirmation is asked (--sudo).", "warn")
            return
        self.note("Shell execution is ENABLED. Every command is checked by the safety "
                  "classifier: destructive commands are blocked, risky ones ask for confirmation.",
                  "warn")

    # -- interactive input --------------------------------------------------
    def read_line(self, prompt_text="", multiline=False, tui=False):
        """Print an (already-styled) prompt and read input from stdin.

        Returns the typed text, or ``None`` on EOF / Ctrl-C. Routing every
        interactive prompt (the "You:" line, permission answers, ...) through
        the console keeps all input handling in one place.

        ``multiline=True`` uses a small TTY line editor: Enter sends, while
        Shift+Enter (and Ctrl-J / LF) inserts a newline. ``tui=True`` enables
        the richer editor (up/down history, cursor movement, Home/End,
        Ctrl-A/E/K/U, Ctrl-L); it implies the multiline key handling. On a
        non-TTY stdin both fall back to a normal ``input()`` line.
        """
        if self.parent_console is not None:
            # A swallowed console (e.g. a sub-agent's) has its output redirected
            # away from the terminal; route both the prompt and the typed answer
            # through the console that actually owns the screen. The prompt
            # string carries its own styling, so it renders identically here or
            # there.
            return self.parent_console.read_line(
                prompt_text, multiline=multiline, tui=tui)
        if multiline or tui:
            try:
                return self._read_multiline(prompt_text, tui=tui)
            except (EOFError, KeyboardInterrupt):
                return None
        try:
            return input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            return None

    def _read_multiline(self, prompt_text="", tui=False):
        """Read one (possibly multi-line) turn from a TTY.

        Uses termios only for the duration of the prompt. Enter (CR) submits;
        LF / Ctrl-J / the Shift+Enter CSI-u sequence insert a newline. In TUI
        mode the line is redrawn as it is edited, with up/down history and
        cursor movement. Ctrl-C still raises KeyboardInterrupt (ISIG stays
        enabled), and the original terminal settings are restored in
        ``finally``. Anything that is not a TTY (tests, pipes, Windows without
        termios) falls back to ``input()``.
        """
        import codecs
        import select

        try:
            import termios
        except ImportError:  # pragma: no cover - Windows
            return input(prompt_text)
        try:
            if not sys.stdin.isatty():
                return input(prompt_text)
            fd = sys.stdin.fileno()
            previous = termios.tcgetattr(fd)
        except (AttributeError, OSError, ValueError, termios.error):
            return input(prompt_text)

        attrs = termios.tcgetattr(fd)
        # Keep ISIG (so Ctrl-C still interrupts) and OPOST (normal newlines),
        # but read key-by-key without canonical line buffering or echo. ICRNL
        # and IXON are *input* flags: clear them there, so Enter stays CR (the
        # submit key) and Ctrl-S/Ctrl-Q do not freeze the prompt.
        attrs[0] &= ~(getattr(termios, "ICRNL", 0)
                      | getattr(termios, "IXON", 0))
        attrs[3] &= ~(termios.ICANON | termios.ECHO)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0

        buffer = []
        cursor = 0
        history = list(self._input_history)
        history_pos = len(history)
        draft = ""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        last_render = {"lines": 0, "cursor_line": 0}

        def write(text):
            self.stream.write(text)
            self.stream.flush()

        # Bytes read while peeking past a CR (to swallow a CRLF) but not
        # consumed belong here, so the next key is never silently dropped --
        # even when it arrives after this call has already returned.
        pending = self._input_pending

        def read_one():
            if pending:
                return pending.pop(0)
            return os.read(fd, 1)

        def read_escape():
            """Read the rest of an escape sequence without waiting for lone Esc.

            CSI/SS3 sequences end on a final byte in ``@``..``~``; stopping
            there matters for the TUI editor, otherwise the next key (often
            Enter) is swallowed into an arrow-key sequence.
            """
            sequence = b""
            while len(sequence) < 12:
                if not pending:
                    ready, _w, _x = select.select([fd], [], [], 0.01)
                    if not ready:
                        break
                byte = read_one()
                sequence += byte
                if len(sequence) == 1:
                    if byte not in (b"[", b"O"):
                        break
                    continue
                code = byte[0]
                if code < 0x20 or 0x40 <= code <= 0x7e:
                    break
            return sequence

        def text_value():
            return "".join(buffer)

        def line_start():
            return text_value().rfind("\n", 0, cursor) + 1

        def line_end():
            found = text_value().find("\n", cursor)
            return len(buffer) if found < 0 else found

        def set_buffer(value):
            nonlocal cursor
            buffer[:] = list(value)
            cursor = len(buffer)

        def insert(value):
            nonlocal cursor
            if not value:
                return
            buffer[cursor:cursor] = list(value)
            cursor += len(value)

        def render():
            """Redraw the whole input block and place the cursor."""
            if not tui:
                return
            if last_render["lines"]:
                write("\r")
                if last_render["cursor_line"]:
                    write(f"\x1b[{last_render['cursor_line']}A")
                write("\x1b[J")
            text = prompt_text + text_value()
            lines = text.split("\n")
            for index, line in enumerate(lines):
                write(line)
                if index < len(lines) - 1:
                    write("\n")
            prefix = prompt_text + "".join(buffer[:cursor])
            prefix_lines = prefix.split("\n")
            cursor_line = len(prefix_lines) - 1
            column = _visual_len(prefix_lines[-1])
            up = len(lines) - 1 - cursor_line
            if up > 0:
                write(f"\x1b[{up}A")
            write("\r")
            if column > 0:
                write(f"\x1b[{column}C")
            last_render["lines"] = len(lines)
            last_render["cursor_line"] = cursor_line

        def history_up():
            nonlocal history_pos, draft
            if not history:
                return
            if history_pos == len(history):
                draft = text_value()
            if history_pos > 0:
                history_pos -= 1
                set_buffer(history[history_pos])
                render()

        def history_down():
            nonlocal history_pos
            if not history or history_pos >= len(history):
                return
            history_pos += 1
            set_buffer(draft if history_pos == len(history)
                       else history[history_pos])
            render()

        def submit():
            nonlocal cursor
            cursor = len(buffer)
            if tui:
                render()
            write("\n")
            text = text_value()
            if tui and text.strip():
                if not self._input_history or self._input_history[-1] != text:
                    self._input_history.append(text)
                    if len(self._input_history) > 100:
                        del self._input_history[0]
            return text

        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            if tui:
                render()
            else:
                write(prompt_text)
            while True:
                raw = read_one()
                if not raw:  # EOF
                    if tui and buffer:
                        cursor = len(buffer)
                        render()
                        write("\n")
                    return None if not buffer else text_value()
                if raw == b"\r":
                    # Some terminals send CRLF; consume the LF so it does not
                    # appear as a spurious empty line on the next prompt, but
                    # push any other following byte back for the next read.
                    if pending:
                        nxt = read_one()
                    else:
                        ready = select.select([fd], [], [], 0.01)[0]
                        nxt = os.read(fd, 1) if ready else b""
                    if nxt not in (b"", b"\n"):
                        pending.insert(0, nxt)
                    return submit()
                if raw == b"\n":       # Ctrl-J / terminals that send LF
                    if tui:
                        insert("\n")
                        render()
                    else:
                        buffer.append("\n")
                        write("\n... ")
                    continue
                if raw == b"\x1b":      # CSI-u / modifyOtherKeys Shift+Enter
                    seq = read_escape()
                    if (seq in (b"[13;2u", b"[13;2~", b"[27;2;13~")
                            or b"13;2" in seq or b";2;13" in seq):
                        if tui:
                            insert("\n")
                            render()
                        else:
                            buffer.append("\n")
                            write("\n... ")
                        continue
                    if tui:
                        if seq in (b"[A", b"OA"):
                            history_up()
                        elif seq in (b"[B", b"OB"):
                            history_down()
                        elif seq in (b"[C", b"OC"):
                            cursor = min(cursor + 1, len(buffer))
                            render()
                        elif seq in (b"[D", b"OD"):
                            cursor = max(cursor - 1, 0)
                            render()
                        elif seq in (b"[H", b"OH", b"[1~", b"[7~"):
                            cursor = line_start()
                            render()
                        elif seq in (b"[F", b"OF", b"[4~", b"[8~"):
                            cursor = line_end()
                            render()
                        elif seq in (b"[3~",):
                            if cursor < len(buffer):
                                del buffer[cursor]
                                render()
                    continue
                if raw in (b"\x7f", b"\b"):  # backspace
                    if tui:
                        if cursor > 0:
                            del buffer[cursor - 1]
                            cursor -= 1
                            render()
                    elif buffer and buffer[-1] != "\n":
                        buffer.pop()
                        write("\b \b")
                    continue
                if raw == b"\x03":      # Ctrl-C (if ISIG were disabled)
                    raise KeyboardInterrupt
                if raw == b"\x04":      # Ctrl-D
                    if not buffer:
                        return None
                    if tui and cursor < len(buffer):
                        del buffer[cursor]
                        render()
                    continue
                if tui and raw == b"\x01":  # Ctrl-A
                    cursor = line_start()
                    render()
                    continue
                if tui and raw == b"\x05":  # Ctrl-E
                    cursor = line_end()
                    render()
                    continue
                if tui and raw == b"\x0b":  # Ctrl-K
                    del buffer[cursor:]
                    render()
                    continue
                if tui and raw == b"\x15":  # Ctrl-U
                    del buffer[:cursor]
                    cursor = 0
                    render()
                    continue
                if tui and raw == b"\x0c":  # Ctrl-L
                    write("\x1b[2J\x1b[H")
                    last_render["lines"] = 0
                    last_render["cursor_line"] = 0
                    render()
                    continue
                for char in decoder.decode(raw):
                    if char == "\n":
                        if tui:
                            insert(char)
                            render()
                        else:
                            buffer.append(char)
                            write("\n... ")
                    elif char == "\t" or char.isprintable():
                        if tui:
                            insert(char)
                            render()
                        else:
                            buffer.append(char)
                            write(char)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, previous)
            except (OSError, ValueError, termios.error):
                pass

    def _permission_prompt_and_answer(self, name, params_str, shell_cmd, tier, reason):
        """Draw the permission box and read the answer on the right console.

        Shared by :meth:`ask_permission`, so the box, the prompt and the input
        all land on one console. For a *swallowed* console (a sub-agent's, whose
        output is redirected to a buffer) the box is drawn on the console that
        actually owns the terminal (``parent_console``) instead -- so the human
        sees it and answers it -- and the parent tool-call spinner is restored
        afterwards. For a normal console it simply operates on ``self``.
        """
        console = self.parent_console if self.parent_console is not None else self
        console.clear_transient()
        lines = [f"Permission needed for tool: {name}",
                 f"Parameters : {params_str}"]
        if shell_cmd is not None:
            snippet = self._oneline(shell_cmd, 120)
            if tier == "ask":
                lines.append(f"Command    : {snippet}")
                lines.append(f"           : risky - needs your confirmation ({reason})")
            else:
                lines.append(f"Command    : {snippet}")
        console.question_box("Question to User", lines)
        prompt = self.style("Proceed? Type 'y' or 'yes' and press Enter: ",
                            fg="yellow", bold=True)
        answer = console.read_line(prompt)
        if console is not self:
            # The box was shown on the parent; restore its tool-call spinner (it
            # was still running while the box was on screen) so the screen is
            # consistent once the sub-agent continues.
            console.stop_tool_call()
            console.start_tool_call()
        return (answer or "").strip().lower() in ("y", "yes")

    def ask_permission(self, name, params_str, shell_cmd=None, tier=None, reason=None):
        """Draw a permission-prompt box and read the user's answer (y / yes).

        The box *and* the input both go through this console. Any previous
        transient tool output is cleared first so the box renders cleanly. For a
        swallowed console (a sub-agent's) the whole box is rendered on the
        console that owns the terminal (its ``parent_console``) instead, so the
        human sees it and answers it -- the main agent responds in place of the
        sub-agent. Returns True to proceed.
        """
        return self._permission_prompt_and_answer(
            name, params_str, shell_cmd, tier, reason)


def tui_mode(value):
    """Normalise ``--tui`` / legacy bool flags to ``auto``, ``on`` or ``off``."""
    if value is True:
        return "on"
    if value is False or value is None:
        return "off"
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return "on"
    if text in ("0", "false", "no", "off"):
        return "off"
    return "auto"


class TuiConsole(Console):
    """A full-screen, fixed-layout console used by ``--tui``.

    Unlike the scrolling :class:`Console`, this keeps a tiny state model and
    redraws the whole terminal on every change:

        +-- Summary box (version/session/model/context) --+
        |                                                 |
        | last conversation content + transient status    |
        |                                                 |
        +-- input box + one-line help -------------------+

    Nothing is appended below the screen: a new turn replaces the body, and
    tool/thinking status is a transient line that disappears when the operation
    finishes. The alternate screen is restored on exit.
    """

    is_fixed = True

    @classmethod
    def supported(cls):
        """Whether the current terminal is a sensible default for the TUI."""
        try:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                return False
        except (AttributeError, OSError, ValueError):
            return False
        term = os.environ.get("TERM", "").strip().lower()
        if not term or term in ("dumb", "unknown"):
            return False
        known = (
            "xterm", "screen", "tmux", "rxvt", "alacritty", "kitty",
            "wezterm", "foot", "konsole", "vte", "gnome", "iterm",
            "eterm", "ansi", "linux", "cygwin", "vt100", "vt220",
        )
        return term.startswith(known)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = False
        self._summary_lines = []
        self._turns = deque(maxlen=40)
        self._turn_id = 0
        self._current_assistant = None
        self._current_tool_entry = None
        self._last_user = ""
        self._last_assistant = ""
        self._info_provider = None
        self._render_lock = threading.RLock()
        self._thinking_stop = None
        self._thinking_thread = None
        self._tool_line = ""
        self._status_line = ""
        self._notice_line = ""
        self._notices = []
        self._overlay_lines = []
        self._question_active = False
        self._question_title = "Question to User"
        self._question_lines = []
        self._last_frame_lines = []
        self._last_frame_size = None
        self._scroll_offset = 0
        self._last_body_height = 1
        self._total_body_lines = 0
        self._input_buffer = []
        self._input_cursor = 0
        self._input_active = False
        self._history_pos = 0
        self._old_winch = None

    # -- lifecycle ---------------------------------------------------------

    def start_tui(self):
        try:
            if not self.stream.isatty():
                return False
        except (AttributeError, OSError):
            return False
        self._active = True
        self._last_frame_lines = []
        self._last_frame_size = None
        if hasattr(signal, "SIGWINCH"):
            try:
                self._old_winch = signal.getsignal(signal.SIGWINCH)
                signal.signal(signal.SIGWINCH, self._on_winch)
            except (ValueError, OSError):
                self._old_winch = None
        # Ask xterm-compatible terminals to distinguish modified Enter keys.
        # Shift+Enter then arrives as CSI 27;2;13~ instead of a plain CR.
        self._write("\x1b[?1049h\x1b[2J\x1b[H"
                    "\x1b[>4;2m\x1b[>1u\x1b[?25l")
        self.render()
        return True

    def _on_winch(self, *_args):
        if self._active:
            self.render()

    def stop_tui(self):
        if not self._active:
            return
        self.stop_thinking()
        self._active = False
        self._last_frame_lines = []
        self._last_frame_size = None
        if self._old_winch is not None and hasattr(signal, "SIGWINCH"):
            try:
                signal.signal(signal.SIGWINCH, self._old_winch)
            except (ValueError, OSError):
                pass
            self._old_winch = None
        self._write("\x1b[<u\x1b[>4;0m\x1b[?25h\x1b[r\x1b[?1049l")

    def _write(self, text):
        self.stream.write(text)
        self.stream.flush()

    def _size(self):
        size = shutil.get_terminal_size((max(self.width, 40), 24))
        return max(int(size.columns), 20), max(int(size.lines), 8)

    # -- state updates -----------------------------------------------------

    def set_summary(self, lines):
        self._summary_lines = [str(line) for line in (lines or [])]
        self.render()

    def load_history(self, turns):
        """Populate the core view with user/assistant turns from a restored session."""
        self._turns.clear()
        self._turn_id = 0
        self._current_assistant = None
        for role, content in list(turns or [])[-40:]:
            if role == "You":
                self._turn_id += 1
            record = {"role": role, "content": content,
                      "turn_id": self._turn_id,
                      "tools": [], "background": []}
            self._turns.append(record)
            if role == "You":
                self._last_user = content
            elif role == "Assistant":
                self._last_assistant = content
                self._current_assistant = record
        self._scroll_offset = 0
        self.render()

    def turn(self, role, content, accent="cyan", badge=None):
        if not self._active:
            return super().turn(role, content, accent=accent, badge=badge)
        text = "" if content is None else str(content)
        if role == "You":
            self._turn_id += 1
            self._current_assistant = None
            self._current_tool_entry = None
            self._last_user = text
            self._turns.append({"role": role, "content": text,
                                "turn_id": self._turn_id,
                                "tools": [], "background": []})
            self._overlay_lines = []
            self._notice_line = ""
            self._notices = []
            self._scroll_offset = 0
        elif role == "Assistant":
            record = {"role": role, "content": text,
                      "turn_id": self._turn_id,
                      "tools": [], "background": []}
            self._last_assistant = text
            self._current_assistant = record
            self._turns.append(record)
            self._overlay_lines = []
            self._scroll_offset = 0
        else:
            self._notice_line = text
        self.render()

    def note(self, msg, level="info"):
        if not self._active:
            return super().note(msg, level)
        self._notices.append((level, str(msg)))
        del self._notices[:-3]
        self.render()

    def _ensure_assistant_record(self):
        if self._current_assistant is None:
            self._current_assistant = {
                "role": "Assistant", "content": "",
                "turn_id": self._turn_id, "tools": [], "background": []}
            self._turns.append(self._current_assistant)
        return self._current_assistant

    def tool_call_display(self, name, params=None, result=None, transient=True):
        """Record the call and show it as a transient line in the fixed TUI."""
        self._tool_history.append((name, params, result))
        if not self._active:
            return super().tool_call_display(name, params, result,
                                             transient=transient)
        record = self._ensure_assistant_record()
        entry = self._current_tool_entry
        if entry is None or entry.get("name") != name:
            entry = {"name": name, "background": ""}
            record["tools"].append(entry)
        entry["name"] = name
        if name == "background" and str((params or {}).get("action", "")).lower() == "start":
            label = ((params or {}).get("command")
                     or (params or {}).get("subagent")
                     or (params or {}).get("task") or "task")
            entry["background"] = str(label)
            record["background"].append(str(label))
        self._current_tool_entry = None
        self._tool_line = f"tool {name}: {str(params)[:60]}"
        self.render()

    def start_thinking(self, message="Thinking"):
        if not self._active:
            return super().start_thinking(message)
        self.stop_thinking()
        frames = ("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if self.unicode_enabled else "|/-\\")
        stop = threading.Event()

        def spin():
            index = 0
            while not stop.is_set():
                self._status_line = f"{frames[index % len(frames)]} {message}…"
                index += 1
                self.render()
                stop.wait(0.12)
            self._status_line = ""
            self.render()

        self._thinking_stop = stop
        self._thinking_thread = threading.Thread(target=spin, daemon=True)
        self._thinking_thread.start()

    def stop_thinking(self):
        if not self._active:
            return super().stop_thinking()
        stop = self._thinking_stop
        thread = self._thinking_thread
        self._thinking_stop = None
        self._thinking_thread = None
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._status_line = ""
        self.render()

    def start_tool_call(self, message="calling tool", tool_name=None):
        if not self._active:
            return super().start_tool_call(message, tool_name=tool_name)
        record = self._ensure_assistant_record()
        entry = {"name": tool_name or "tool", "background": ""}
        record["tools"].append(entry)
        self._current_tool_entry = entry
        self._tool_line = f"… {message}: {tool_name or ''}".strip()
        self.render()

    def stop_tool_call(self):
        if not self._active:
            return super().stop_tool_call()
        self._tool_line = ""
        self.render()

    def usage_line(self, usage, transient=None, llm_calls=None):
        if not self._active:
            return super().usage_line(usage, transient=transient,
                                      llm_calls=llm_calls)
        # The fixed view keeps the body for conversation content; token usage
        # remains available through /context and the normal (non-TUI) console.
        return

    def emit_transient(self, text=""):
        if not self._active:
            return super().emit_transient(text)
        self._status_line = str(text or "")
        self.render()

    def clear_transient(self):
        if not self._active:
            return super().clear_transient()
        self._status_line = ""
        self.render()

    def emit(self, text=""):
        if not self._active:
            return super().emit(text)
        self._overlay_lines.append("" if text is None else str(text))
        self.render()

    def rule(self, title=None, char="═", fg="cyan"):
        if not self._active:
            return super().rule(title, char=char, fg=fg)
        # A titled rule starts a new command output. An untitled rule is a
        # closing separator (``print_help`` emits one at the end); it must not
        # clear the output that was just produced.
        if title:
            self._scroll_offset = 0
            self._overlay_lines = [self.style(f"[{title}]", fg=fg, bold=True)]
            self.render()

    def divider(self):
        if not self._active:
            return super().divider()

    def short_divider(self, char="-", width=8, fg="gray"):
        if not self._active:
            return super().short_divider(char=char, width=width, fg=fg)

    def print_box(self, lines, title=None, fg="cyan"):
        if not self._active:
            return super().print_box(lines, title=title, fg=fg)
        self._scroll_offset = 0
        styled = (list(lines or []) if fg != "yellow"
                  else [self.style(str(line), fg="yellow") for line in lines or []])
        self._overlay_lines = ([self.style(f"[{title}]", fg=fg, bold=True)]
                               if title else []) + styled
        self.render()

    def question_box(self, title, lines):
        if not self._active:
            return super().question_box(title, lines)
        self._question_title = title or "Question to User"
        self._question_lines = list(lines or [])
        self._question_active = True
        self._overlay_lines = []
        self._scroll_offset = 0
        self.render()

    def append_block(self, label, text):
        """Append a labelled block (e.g. a generated plan) to the core view."""
        if not self._active:
            return
        self._turns.append({"role": str(label), "content": str(text),
                            "turn_id": self._turn_id,
                            "tools": [], "background": []})
        self._overlay_lines = []
        self._scroll_offset = 0
        self.render()

    def plan_box(self, lines, title="Plan for the current goal"):
        if not self._active:
            return super().plan_box(lines, title=title)
        self.append_block("Plan", "\n".join(str(line) for line in lines or []))

    def status_box(self, lines, title=None, fg="cyan", dim=False):
        if not self._active:
            return super().status_box(lines, title=title, fg=fg, dim=dim)
        self._scroll_offset = 0
        self._overlay_lines = ([self.style(f"[{title}]", fg=fg, bold=True)]
                               if title else []) + list(lines or [])
        self.render()

    def clear_screen(self):
        if not self._active:
            return super().clear_screen()
        self._last_user = ""
        self._last_assistant = ""
        self._overlay_lines = []
        self._notice_line = ""
        self._notices = []
        self.render()

    # -- rendering ---------------------------------------------------------

    @staticmethod
    def _fit(text, width):
        text = "" if text is None else str(text)
        if _visual_len(text) <= width:
            return text
        plain = _ANSI_RE.sub("", text)
        limit = max(width - 3, 0)
        out, used = [], 0
        for char in plain:
            char_width = _visual_len(char)
            if used + char_width > limit:
                break
            out.append(char)
            used += char_width
        return "".join(out) + "..."

    def _box(self, title, body, width, fg="cyan"):
        h = self.glyph("h")
        v = self.glyph("v")
        tl, tr = self.glyph("tl"), self.glyph("tr")
        bl, br = self.glyph("bl"), self.glyph("br")
        label = f" {title} " if title else ""
        top_fill = max(width - 2 - _visual_len(label), 0)
        top = self.style(tl + label + h * top_fill + tr, fg=fg, bold=True)
        bottom = self.style(bl + h * max(width - 2, 0) + br, fg=fg, bold=True)
        border = self.style(v, fg=fg, bold=True)
        lines = [top]
        inner = max(width - 4, 1)
        for line in body:
            text = self._fit(line, inner)
            lines.append(border + " " + _pad_visual(text, inner) + " " + border)
        lines.append(bottom)
        return lines

    def _wrap(self, text, width):
        out = []
        text = _ANSI_RE.sub("", str(text or ""))
        for line in text.splitlines() or [""]:
            if not line:
                out.append("")
                continue
            out.extend(textwrap.wrap(line, width=max(width, 1)) or [""])
        return out

    def _input_lines(self):
        text = "".join(self._input_buffer)
        return text.split("\n") if text else [""]

    def _notice_style(self, level, msg):
        icons = {"info": self.glyph("info"), "warn": self.glyph("warn"),
                 "error": self.glyph("error"), "ok": self.glyph("ok")}
        colors = {"info": "cyan", "warn": "yellow", "error": "red",
                  "ok": "green"}
        return self.style(f"{icons.get(level, self.glyph('info'))} {msg}",
                          fg=colors.get(level, "cyan"),
                          dim=(level == "info"))

    def _content_lines(self, width):
        if self._overlay_lines:
            return [self._fit(line, width) for line in self._overlay_lines]
        lines = []
        assistant_counts = {}
        for record in self._turns:
            if record.get("role") == "Assistant":
                turn = record.get("turn_id")
                assistant_counts[turn] = assistant_counts.get(turn, 0) + 1
        for record in self._turns:
            role = record.get("role", "?")
            content = record.get("content", "")
            if role == "You":
                role_color = "cyan"
            elif role == "Assistant":
                role_color = "green"
            elif role == "Plan":
                role_color = "yellow"
            else:
                role_color = "magenta"
            lines.append(self.style(f"{self.glyph('role')} {role}",
                                    fg=role_color, bold=True))
            if role == "Assistant":
                tool_names = [entry.get("name", "?")
                              for entry in record.get("tools", [])]
                background = [label for label in record.get("background", [])
                              if label]
                tools_text = ", ".join(tool_names) if tool_names else "none"
                bg_text = ", ".join(background) if background else "none"
                messages = assistant_counts.get(record.get("turn_id"), 1)
                lines.append(self.style(
                    f"  ↳ tools: {tools_text} · background: {bg_text} · "
                    f"messages: {messages}", dim=True))
            lines.extend("  " + line for line in self._wrap(content, width))
            lines.append(self.style(self.glyph("h") * max(width, 1), dim=True))
        if lines and self._turns:
            lines.pop()             # no separator after the newest turn
        if not lines:
            lines.append(self.style(
                "No conversation yet. Type a message; /help lists commands.",
                dim=True))
        if self._tool_line:
            lines.append(self.style(f"{self.glyph('gear')} {self._tool_line}",
                                    fg="magenta", dim=True))
        if self._status_line:
            lines.append(self.style(self._status_line, fg="cyan", dim=True))
        notices = [self._notice_style(level, msg)
                   for level, msg in self._notices[-3:]]
        if self._notice_line:
            notices.append(self.style(self._notice_line, fg="yellow", dim=True))
        if notices:
            lines.append(self.style(self.glyph("h") * max(width, 1), dim=True))
            lines.extend(notices)
        return [self._fit(line, width) for line in lines]

    def _body_lines(self, height, width):
        lines = self._content_lines(width)
        self._last_body_height = max(height, 1)
        self._total_body_lines = len(lines)
        max_offset = max(len(lines) - height, 0)
        self._scroll_offset = min(max(self._scroll_offset, 0), max_offset)
        if self._scroll_offset and max_offset:
            end = max_offset - self._scroll_offset + height
            start = max(end - height, 0)
            view = lines[start:end]
        else:
            view = lines[-height:]
        return (view + [""] * height)[:height]

    def scroll_body(self, lines):
        """Scroll the middle pane by ``lines`` (positive = up)."""
        self._scroll_offset = max(0, self._scroll_offset + int(lines))
        self.render()

    def scroll_page(self, pages):
        """Scroll the middle pane by roughly one page (positive = up)."""
        page = max(self._last_body_height - 1, 1)
        self.scroll_body(pages * page)

    def dismiss_overlay(self):
        """Escape from command output or a scrolled view back to the core view."""
        if self._question_active:
            return
        if self._overlay_lines:
            self._overlay_lines = []
            self._scroll_offset = 0
            self.render()
        elif self._scroll_offset:
            self._scroll_offset = 0
            self.render()

    def set_info_provider(self, provider):
        """Set a callable returning a one-line session summary for the header."""
        self._info_provider = provider

    def render(self):
        if not self._active:
            return
        with self._render_lock:
            self._render_locked()

    def _render_locked(self):
        cols, rows = self._size()
        summary = [str(line) for line in (self._summary_lines
                                          or ["Mini-Code"])]
        if self._info_provider is not None:
            try:
                info = self._info_provider()
            except Exception:
                info = ""
            if info:
                summary.append(self.style(info, dim=True))
        header = self._box("Mini-Code", summary[:max(rows // 3, 3)], cols,
                           fg="cyan")

        input_lines = self._input_lines()
        max_input = max(1, min(rows // 2, rows - 6))
        if len(input_lines) > max_input:
            input_lines = input_lines[-max_input:]
        if self._scroll_offset:
            hint_text = (f"scrolled {self._scroll_offset}/"
                         f"{self._total_body_lines} lines · "
                         f"Shift+↓ / Esc to bottom")
        else:
            hint_text = ("Enter send · Shift+Enter newline · ↑/↓ history · "
                         "Shift+↑/↓ scroll · Esc close command · /help")
        hint = self.style(hint_text, dim=True)
        input_title = "Answer" if self._question_active else "You"
        input_fg = "yellow" if self._question_active else "green"
        body_box = input_lines + [hint]
        input_box = self._box(input_title, body_box, cols, fg=input_fg)
        question_box = []
        if self._question_active and self._question_lines:
            question_box = self._box(
                self._question_title or "Question to User",
                self._question_lines[:max(rows // 3, 3)], cols, fg="yellow")
        body_height = max(rows - len(header) - len(question_box)
                          - len(input_box), 1)
        body = self._body_lines(body_height, max(cols - 4, 1))
        lines = header + body + question_box + input_box
        lines = (lines + [""] * rows)[:rows]
        padded = [_pad_visual(self._fit(line, cols), cols) for line in lines]
        self._write("\x1b[?25l")
        if self._last_frame_size != (cols, rows):
            self._write("\x1b[2J\x1b[H")
            self._write("\n".join(padded))
        else:
            for row, line in enumerate(padded, start=1):
                if line != self._last_frame_lines[row - 1]:
                    self._write(f"\x1b[{row};1H\x1b[2K{line}")
        self._last_frame_lines = padded
        self._last_frame_size = (cols, rows)
        if self._input_active:
            input_top = len(header) + body_height + len(question_box)
            prefix = "".join(self._input_buffer[:self._input_cursor])
            cursor_line = prefix.count("\n")
            column = len(prefix.rsplit("\n", 1)[-1])
            row = min(input_top + 2 + cursor_line, rows)
            col = min(3 + column, cols)
            self._write(f"\x1b[{row};{col}H\x1b[?25h")
        else:
            self._write("\x1b[?25l")

    # -- full-screen input -------------------------------------------------

    def _read_tui_input(self, prompt_text=""):
        import codecs
        import select

        try:
            import termios
        except ImportError:  # pragma: no cover - Windows
            return input(prompt_text)
        try:
            if not sys.stdin.isatty():
                return input(prompt_text)
            fd = sys.stdin.fileno()
            previous = termios.tcgetattr(fd)
        except (AttributeError, OSError, ValueError, termios.error):
            return input(prompt_text)

        attrs = termios.tcgetattr(fd)
        attrs[0] &= ~(getattr(termios, "ICRNL", 0)
                      | getattr(termios, "IXON", 0))
        attrs[3] &= ~(termios.ICANON | termios.ECHO)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0

        buffer = []
        cursor = 0
        history = list(self._input_history)
        history_pos = len(history)
        draft = ""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._notice_line = prompt_text.strip() if prompt_text.strip() else ""
        self._input_buffer = buffer
        self._input_cursor = cursor
        self._input_active = True
        self.render()

        def read_one():
            return os.read(fd, 1)

        def read_escape():
            sequence = b""
            while len(sequence) < 12:
                ready, _w, _x = select.select([fd], [], [], 0.01)
                if not ready:
                    break
                byte = read_one()
                sequence += byte
                if len(sequence) == 1:
                    if byte not in (b"[", b"O"):
                        break
                    continue
                code = byte[0]
                if code < 0x20 or 0x40 <= code <= 0x7e:
                    break
            return sequence

        def sync():
            self._input_buffer = buffer
            self._input_cursor = cursor
            self.render()

        def set_buffer(value):
            nonlocal cursor
            buffer[:] = list(value)
            cursor = len(buffer)
            sync()

        def line_start():
            return "".join(buffer).rfind("\n", 0, cursor) + 1

        def line_end():
            found = "".join(buffer).find("\n", cursor)
            return len(buffer) if found < 0 else found

        def submit_current():
            nonlocal cursor
            text = "".join(buffer)
            if text.strip() and (not self._input_history
                                 or self._input_history[-1] != text):
                self._input_history.append(text)
                if len(self._input_history) > 100:
                    del self._input_history[0]
            buffer[:] = []
            cursor = 0
            self._input_active = False
            self._input_buffer = []
            self._input_cursor = 0
            self._notice_line = ""
            self.render()
            return text

        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            # Discard keys typed while the agent was busy; otherwise an arrow
            # pressed during an LLM/tool call replays at the next prompt and
            # scrambles the fixed view.
            try:
                termios.tcflush(fd, termios.TCIFLUSH)
            except (OSError, ValueError, termios.error):
                pass
            self._input_pending.clear()
            while True:
                raw = read_one()
                if not raw:
                    self._input_active = False
                    self.render()
                    return None
                if raw == b"\r":
                    return submit_current()
                if raw == b"\n":
                    buffer[cursor:cursor] = ["\n"]
                    cursor += 1
                    sync()
                    continue
                if raw == b"\x1b":
                    seq = read_escape()
                    if (seq in (b"[13;2u", b"[13;2~", b"[27;2;13~",
                                b"[13;2;13~", b"[27;2;13u")
                            or b"13;2" in seq or b";2;13" in seq
                            or seq == b"\r"):
                        buffer[cursor:cursor] = ["\n"]
                        cursor += 1
                        sync()
                        continue
                    if seq in (b"[13u", b"[13;1u"):
                        # kitty/CSI-u unmodified Enter, if the terminal reports it
                        return submit_current()
                    if seq.startswith(b"[") and seq.endswith(b"u"):
                        # Kitty/CSI-u control keys. Enabling the protocol makes
                        # Ctrl+letter and Escape unambiguous; map the handful
                        # this editor uses back to their raw behaviour.
                        parts = seq[1:-1].split(b";")
                        try:
                            code = int(parts[0])
                        except (TypeError, ValueError):
                            code = None
                        mod = parts[1].split(b":")[0] if len(parts) > 1 else b""
                        if code == 27:
                            self.dismiss_overlay()
                        elif code == 127:
                            if cursor > 0:
                                del buffer[cursor - 1]
                                cursor -= 1
                                sync()
                        elif code == 9:
                            buffer[cursor:cursor] = ["\t"]
                            cursor += 1
                            sync()
                        elif code in (13, 109):
                            return submit_current()
                        elif code in (10, 106):
                            buffer[cursor:cursor] = ["\n"]
                            cursor += 1
                            sync()
                        elif mod.startswith(b"5"):
                            if code == 99:              # Ctrl-C
                                raise KeyboardInterrupt
                            if code == 100:             # Ctrl-D
                                if not buffer:
                                    self._input_active = False
                                    self.render()
                                    return None
                                if cursor < len(buffer):
                                    del buffer[cursor]
                                    sync()
                            elif code == 97:             # Ctrl-A
                                cursor = line_start()
                                sync()
                            elif code == 101:            # Ctrl-E
                                cursor = line_end()
                                sync()
                            elif code == 107:            # Ctrl-K
                                del buffer[cursor:]
                                sync()
                            elif code == 117:            # Ctrl-U
                                del buffer[:cursor]
                                cursor = 0
                                sync()
                            elif code == 108:            # Ctrl-L
                                self._overlay_lines = []
                                self._last_user = ""
                                self._last_assistant = ""
                                self._notice_line = ""
                                sync()
                            elif code == 116:            # Ctrl-T
                                self.expand_tool(1)
                                sync()
                        continue
                    if seq in (b"[1;2A", b"[a"):
                        self.scroll_body(1)
                        continue
                    if seq in (b"[1;2B", b"[b"):
                        self.scroll_body(-1)
                        continue
                    if seq in (b"[5~",):
                        self.scroll_page(1)
                        continue
                    if seq in (b"[6~",):
                        self.scroll_page(-1)
                        continue
                    if not seq:
                        self.dismiss_overlay()
                        continue
                    if seq in (b"[A", b"OA"):
                        if history:
                            if history_pos == len(history):
                                draft = "".join(buffer)
                            if history_pos > 0:
                                history_pos -= 1
                                set_buffer(history[history_pos])
                    elif seq in (b"[B", b"OB"):
                        if history_pos < len(history):
                            history_pos += 1
                            set_buffer(draft if history_pos == len(history)
                                       else history[history_pos])
                    elif seq in (b"[C", b"OC"):
                        cursor = min(cursor + 1, len(buffer))
                        sync()
                    elif seq in (b"[D", b"OD"):
                        cursor = max(cursor - 1, 0)
                        sync()
                    elif seq in (b"[H", b"OH", b"[1~", b"[7~"):
                        cursor = line_start()
                        sync()
                    elif seq in (b"[F", b"OF", b"[4~", b"[8~"):
                        cursor = line_end()
                        sync()
                    elif seq in (b"[3~",):
                        if cursor < len(buffer):
                            del buffer[cursor]
                            sync()
                    continue
                if raw in (b"\x7f", b"\b"):
                    if cursor > 0:
                        del buffer[cursor - 1]
                        cursor -= 1
                        sync()
                    continue
                if raw == b"\x03":
                    raise KeyboardInterrupt
                if raw == b"\x04":
                    if not buffer:
                        self._input_active = False
                        self.render()
                        return None
                    if cursor < len(buffer):
                        del buffer[cursor]
                        sync()
                    continue
                if raw == b"\x01":
                    cursor = line_start()
                    sync()
                    continue
                if raw == b"\x05":
                    cursor = line_end()
                    sync()
                    continue
                if raw == b"\x0b":
                    del buffer[cursor:]
                    sync()
                    continue
                if raw == b"\x15":
                    del buffer[:cursor]
                    cursor = 0
                    sync()
                    continue
                if raw == b"\x0c":
                    self._overlay_lines = []
                    self._last_user = ""
                    self._last_assistant = ""
                    self._notice_line = ""
                    sync()
                    continue
                if raw == b"\x14":  # Ctrl-T: expand the most recent tool
                    self.expand_tool(1)
                    sync()
                    continue
                for char in decoder.decode(raw):
                    if char == "\n":
                        buffer[cursor:cursor] = [char]
                        cursor += 1
                    elif char == "\t" or char.isprintable():
                        buffer[cursor:cursor] = [char]
                        cursor += 1
                    sync()
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, previous)
            except (OSError, ValueError, termios.error):
                pass

    def read_line(self, prompt_text="", multiline=False, tui=False):
        if not self._active:
            return super().read_line(prompt_text, multiline=multiline, tui=tui)
        try:
            return self._read_tui_input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            self._input_active = False
            return None
        finally:
            if self._question_active:
                self._question_active = False
                self._question_lines = []
                self.render()


# One shared default console used throughout the package.
console = Console(color=True, unicode=True)
