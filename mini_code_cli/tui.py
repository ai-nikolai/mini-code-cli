"""mini_code_cli.tui -- the opt-in fixed-screen terminal UI.

The scrolling :class:`mini_code_cli.ui.Console` streams turns into the normal
terminal scrollback. :class:`TuiConsole` instead keeps a small state model and
redraws a fixed frame: dashboard, core conversation, transient tool/thinking
status, questions, and the input box. It deliberately subclasses ``Console`` so
both UIs share styling, glyphs, the context bar, tool history and the
``/expand`` helper; only presentation and input are replaced here.
"""

import os
import shutil
import signal
import sys
import textwrap
import threading
from collections import deque

from .ui import (Console, _ANSI_RE, _pad_visual, _read_escape_sequence,
                 _visual_len)


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

    @property
    def active(self):
        return self._active

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
        # Up/down prompt history for this full-screen console.
        self._input_history = []
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
                    seq = _read_escape_sequence(fd, read_one, [], select)
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
