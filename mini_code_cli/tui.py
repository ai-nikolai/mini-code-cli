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

from .ui import (Console, LOGO, TAGLINE, _ANSI_RE, _pad_visual,
                 _read_escape_sequence, _stop_thread, _visual_len)


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

    @property
    def active(self):
        return self._active

    @staticmethod
    def supported():
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
        self._scroll_stop = None
        self._scroll_thread = None
        self._summary_lines = []
        self._turns = deque(maxlen=40)
        self._turn_id = 0
        self._current_assistant = None
        self._current_tool_entry = None
        self._info_provider = None
        self._render_lock = threading.RLock()
        self._thinking_stop = None
        self._thinking_thread = None
        self._tool_line = ""
        self._status_line = ""
        self._notice_line = ""
        self._notices = []
        # Notes/warnings emitted before the first turn belong in the core
        # view, above the conversation (or its "no conversation yet" hint).
        self._startup_notices = []
        self._overlay_lines = []
        # Command output overlays are anchored to the bottom (like a rolling
        # log). Interactive pickers set this so their table starts at the top
        # and scrolls down with the selection.
        self._overlay_top = False
        # A focused sub-window for editing one setting. It replaces the whole
        # frame so no dashboard/notice boxes compete with the input.
        self._edit_active = False
        self._edit_title = ""
        self._edit_current = ""
        self._edit_choices = ()
        self._edit_choice_mode = False
        self._edit_options = ()
        self._edit_selected = 0
        self._edit_typed = ""
        self._edit_mask = False
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
        # Enter the alternate screen and hide the cursor while we own the
        # terminal. The enhanced keyboard protocols are enabled only around
        # prompt input (see _read_tui_input) so Ctrl+C stays a signal.
        self._write("\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l")
        self.render()
        self.start_scroll_listener()
        return True

    def _on_winch(self, *_args):
        if self._active:
            self.render()

    def stop_tui(self):
        if not self._active:
            return
        self.stop_scroll_listener()
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
            if role == "Assistant":
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
        if not self._turns and not self._overlay_lines:
            self._startup_notices.append((level, str(msg)))
            self.render()
            return
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
        _stop_thread(stop, thread)
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
            self._overlay_top = False
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
        self._overlay_top = False
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
        self._overlay_top = False
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
        self._overlay_top = False
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
        self._overlay_top = False
        self._overlay_lines = ([self.style(f"[{title}]", fg=fg, bold=True)]
                               if title else []) + list(lines or [])
        self.render()

    def clear_screen(self):
        if not self._active:
            return super().clear_screen()
        # In the fixed TUI, /clear behaves like Escape in command mode: close
        # any command output/overlay, return a scrolled view to the bottom, and
        # drop transient info/warning notices. The core conversation stays
        # visible -- only the scrolling console clears the terminal itself.
        self._overlay_lines = []
        self._overlay_top = False
        self._scroll_offset = 0
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
        if self._startup_notices:
            lines.extend(self._notice_style(level, msg)
                         for level, msg in self._startup_notices)
            lines.append(self.style(self.glyph("h") * max(width, 1),
                                    dim=True))
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
        if not self._turns:
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
        if self._overlay_lines and self._overlay_top:
            # A picker table is already windowed around its selection; show it
            # from the first line instead of anchoring the tail to the bottom.
            self._scroll_offset = 0
            return (lines[:height] + [""] * height)[:height]
        if self._overlay_lines:
            # Command output (help/history/context/...) keeps its title line
            # pinned while the rest scrolls underneath it.
            header = lines[:1]
            body = lines[1:]
            available = max(height - len(header), 1)
            max_offset = max(len(body) - available, 0)
            self._scroll_offset = min(max(self._scroll_offset, 0), max_offset)
            if self._scroll_offset and max_offset:
                end = max_offset - self._scroll_offset + available
                start = max(end - available, 0)
                view = body[start:end]
            else:
                view = body[-available:]
            return (header + view + [""] * height)[:height]
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
            self._overlay_top = False
            self._scroll_offset = 0
            self.render()
        elif self._scroll_offset:
            self._scroll_offset = 0
            self.render()

    def set_info_provider(self, provider):
        """Set a callable returning a one-line session summary for the header."""
        self._info_provider = provider

    # -- scroll keys while the agent is busy -------------------------------

    def start_scroll_listener(self):
        """Read scroll/Escape keys while the model or a tool is working.

        The prompt editor owns stdin during input. Between prompts this small
        listener keeps only the scrolling keys alive, so the middle pane can be
        browsed while a long LLM/tool call is running. It enables no enhanced
        keyboard protocol, so Ctrl+C still reaches the process as SIGINT.
        """
        if (self._scroll_thread is not None or not self._active
                or not sys.stdin.isatty()):
            return
        stop = threading.Event()

        def listen():
            import select
            try:
                import termios
            except ImportError:  # pragma: no cover - Windows
                return
            try:
                fd = sys.stdin.fileno()
                previous = termios.tcgetattr(fd)
            except (AttributeError, OSError, ValueError, termios.error):
                return
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
                while not stop.is_set():
                    ready, _w, _x = select.select([fd], [], [], 0.1)
                    if not ready:
                        continue
                    raw = os.read(fd, 1)
                    if not raw:
                        break
                    if raw != b"\x1b":
                        continue
                    seq = _read_escape_sequence(
                        fd, lambda: os.read(fd, 1), [], select)
                    if seq in (b"[1;2A", b"[a"):
                        self.scroll_body(1)
                    elif seq in (b"[1;2B", b"[b"):
                        self.scroll_body(-1)
                    elif seq in (b"[5~",):
                        self.scroll_page(1)
                    elif seq in (b"[6~",):
                        self.scroll_page(-1)
                    elif not seq:
                        self.dismiss_overlay()
            except (OSError, ValueError, termios.error):
                pass
            finally:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, previous)
                except (OSError, ValueError, termios.error):
                    pass

        self._scroll_stop = stop
        self._scroll_thread = threading.Thread(target=listen, daemon=True)
        self._scroll_thread.start()

    def stop_scroll_listener(self):
        stop = self._scroll_stop
        thread = self._scroll_thread
        self._scroll_stop = None
        self._scroll_thread = None
        _stop_thread(stop, thread)

    def render(self):
        if not self._active:
            return
        with self._render_lock:
            self._render_locked()

    def _render_locked(self):
        cols, rows = self._size()
        if self._edit_active:
            self._render_edit_locked(cols, rows)
            return
        # The ASCII mark is a persistent banner above every view, not part of
        # the dashboard box: it stays visible while settings are open too.
        banner = self._banner_lines(cols, rows)
        # On a short terminal the conversation and input matter more than the
        # dashboard; the banner + input still leave a usable body. The hint is
        # dropped only when even that does not fit.
        hide_header = rows < 18
        show_hint = rows >= 14
        if hide_header:
            header = []
        else:
            summary = [str(line) for line in (self._summary_lines
                                              or ["Mini-Code"])]
            if self._info_provider is not None:
                try:
                    info = self._info_provider()
                except Exception:
                    info = ""
                if info:
                    summary.append(self.style(info, dim=True))
            max_summary = max(rows // 4, 3)
            header = self._box("Mini-Code", summary[:max_summary], cols,
                               fg="cyan")

        input_lines = self._input_lines()
        max_input = max(1, min(rows // 2,
                               rows - len(banner) - len(header) - 4))
        cursor_line = 0
        column = 0
        if self._input_active:
            prefix = "".join(self._input_buffer[:self._input_cursor])
            cursor_line = prefix.count("\n")
            # Keep the caret's line visible instead of always showing the tail
            # of a tall input: the visible window follows the cursor, then
            # clamps to the end of the buffer.
            start = max(0, cursor_line - max_input + 1)
            start = min(start, max(0, len(input_lines) - max_input))
            input_lines = input_lines[start:start + max_input]
            visible_cursor_line = cursor_line - start
            # CJK and other wide glyphs occupy two cells, so use display width
            # rather than the number of code points.
            column = _visual_len(prefix.rsplit("\n", 1)[-1])
        else:
            visible_cursor_line = 0
            input_lines = input_lines[-max_input:]
        input_title = "Answer" if self._question_active else "You"
        input_fg = "yellow" if self._question_active else "green"
        body_box = list(input_lines)
        if show_hint:
            if self._scroll_offset:
                hint_text = (f"scrolled {self._scroll_offset}/"
                             f"{self._total_body_lines} lines · "
                             f"↑/↓ or Shift+↑/↓ scroll · Esc to bottom")
            elif self._overlay_lines:
                hint_text = ("↑/↓ or Shift+↑/↓ scroll · Esc close command · "
                             "/help")
            else:
                hint_text = ("Enter send · Shift+Enter newline · ↑/↓ history · "
                             "Shift+↑/↓ scroll · Esc close command · /help")
            body_box.append(self.style(hint_text, dim=True))
        input_box = self._box(input_title, body_box, cols, fg=input_fg)
        question_box = []
        if self._question_active and self._question_lines:
            question_box = self._box(
                self._question_title or "Question to User",
                self._question_lines[:max(rows // 3, 3)], cols, fg="yellow")
        body_height = max(rows - len(banner) - len(header)
                          - len(question_box) - len(input_box), 1)
        body = self._body_lines(body_height, max(cols - 4, 1))
        lines = banner + header + body + question_box + input_box
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
            input_top = (len(banner) + len(header) + body_height
                         + len(question_box))
            row = min(input_top + 2 + visible_cursor_line, rows)
            # Text starts at column 3 (border + space); stop at the inner edge
            # so a long line never puts the caret on the right border.
            col = max(3, min(3 + column, cols - 2))
            self._write(f"\x1b[{row};{col}H\x1b[?25h")
        else:
            self._write("\x1b[?25l")

    def _center(self, text, width):
        """Center an already-styled line, ignoring ANSI escape width."""
        pad = max((width - _visual_len(text)) // 2, 0)
        return " " * pad + text

    def _banner_lines(self, cols, rows):
        """Return the centered, unboxed ASCII mark used by --tui off.

        The art is skipped on narrow terminals, exactly like the scrolling
        welcome banner. It is shown above every view, including the focused
        settings editor.
        """
        art_width = max(_visual_len(line) for line in LOGO)
        if cols < art_width + 4:
            return []
        # Center the block once: centering each logo line separately would
        # misalign the 42- and 43-column rows relative to each other.
        pad = " " * max((cols - art_width) // 2, 0)
        lines = [pad + self.style(line, fg="cyan", bold=True)
                 for line in LOGO]
        lines.append(pad + self.style(TAGLINE, dim=True))
        # No blank spacer: keep the mark one line closer to the main box.
        return lines

    def _render_edit_locked(self, cols, rows):
        """Draw the focused setting window plus the persistent banner."""
        banner = self._banner_lines(cols, rows)
        if self._edit_choice_mode:
            head = [self.style(f"[{self._edit_title or 'Edit setting'}]",
                               fg="cyan", bold=True)]
            if self._edit_current not in ("", None):
                head.append(self.style(f"current: {self._edit_current}",
                                       dim=True))
            hint = "↑/↓ or number, Enter save · type 'exit' or Esc back"
            if self._edit_typed:
                hint = f"typed: {self._edit_typed} · {hint}"
            head.append(self.style(hint, dim=True))
            body = []
            for index, option in enumerate(self._edit_options):
                marker = ">" if index == self._edit_selected else " "
                body.append(f"{marker} {index + 1}. {option}")
            box = self._box("Value", body, cols, fg="green")
            lines = (banner + head + box + [""] * rows)[:rows]
            padded = [_pad_visual(self._fit(line, cols), cols)
                      for line in lines]
            self._write("\x1b[2J\x1b[H" + "\n".join(padded))
            self._last_frame_lines = padded
            self._last_frame_size = (cols, rows)
            self._write("\x1b[?25l")
            return

        head = [self.style(f"[{self._edit_title or 'Edit setting'}]",
                           fg="cyan", bold=True)]
        if self._edit_current not in ("", None):
            head.append(self.style(f"current: {self._edit_current}",
                                   dim=True))
        if self._edit_choices:
            head.append(self.style(
                "choices: " + ", ".join(str(c) for c in self._edit_choices),
                dim=True))
        head.append(self.style(
            "Enter save · type 'exit' or Esc to go back", dim=True))

        max_input = max(1, min(rows // 2,
                               rows - len(banner) - len(head) - 4))
        if self._edit_mask:
            real = "".join(self._input_buffer)
            input_lines = ["*" * len(real)]
            visible_cursor_line = 0
            column = len(real) if self._input_active else 0
        else:
            input_lines = self._input_lines()
            if self._input_active:
                prefix = "".join(self._input_buffer[:self._input_cursor])
                cursor_line = prefix.count("\n")
                start = max(0, cursor_line - max_input + 1)
                start = min(start, max(0, len(input_lines) - max_input))
                input_lines = input_lines[start:start + max_input]
                visible_cursor_line = cursor_line - start
                column = _visual_len(prefix.rsplit("\n", 1)[-1])
            else:
                visible_cursor_line = 0
                input_lines = input_lines[-max_input:]
                column = 0

        box = self._box("Value", input_lines, cols, fg="green")
        lines = (banner + head + box + [""] * rows)[:rows]
        padded = [_pad_visual(self._fit(line, cols), cols) for line in lines]
        # The focused window owns the whole screen; a full repaint avoids
        # leaving slices of the settings table or dashboard behind it.
        self._write("\x1b[2J\x1b[H" + "\n".join(padded))
        self._last_frame_lines = padded
        self._last_frame_size = (cols, rows)
        if self._input_active:
            row = max(1, min(len(banner) + len(head) + 2
                             + visible_cursor_line, rows))
            col = max(3, min(3 + column, cols - 2))
            self._write(f"\x1b[{row};{col}H\x1b[?25h")
        else:
            self._write("\x1b[?25l")

    def settings_edit(self, name, current="", choices=(), secret=False,
                      current_label=None):
        """Edit one setting in a focused sub-window; return text or ``None``."""
        if not self._active:
            return super().settings_edit(
                name, current=current, choices=choices, secret=secret,
                current_label=current_label)
        self.stop_scroll_listener()
        self._edit_active = True
        self._edit_mask = bool(secret)
        self._edit_title = f"Setting: {name}"
        label = current if current_label is None else current_label
        self._edit_current = "" if label is None else str(label)
        self._edit_choices = tuple(choices or ())
        self._input_buffer = []
        self._input_cursor = 0
        self._input_active = True
        self._notice_line = ""
        self._overlay_lines = []
        self._overlay_top = False
        try:
            return self._read_tui_input("", mask=secret, remember=False)
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            self._edit_active = False
            self._edit_mask = False
            self._edit_title = ""
            self._edit_current = ""
            self._edit_choices = ()
            self._input_active = False
            self._input_buffer = []
            self._input_cursor = 0
            self._notice_line = ""
            # The edit window cleared the screen, so force a full repaint of
            # the main (or settings) view when we return.
            self._last_frame_size = None
            self.render()
            if self._active:
                self.start_scroll_listener()

    def settings_choose(self, name, options, current="", current_label=None):
        """Pick a fixed setting value in the focused sub-window.

        ``None`` means "go back" (Esc, q, or typing ``exit``). ``current`` is
        the raw value used to preselect an option; ``current_label`` is what
        the header shows (for example ``1.0 (default)``).
        """
        options = list(options or ())
        if not self._active or not options:
            return super().settings_choose(
                name, options, current=current, current_label=current_label)
        start = options.index(current) if current in options else 0
        self.stop_scroll_listener()
        self._edit_active = True
        self._edit_choice_mode = True
        self._edit_title = f"Setting: {name}"
        label = current if current_label is None else current_label
        self._edit_current = "" if label is None else str(label)
        self._edit_options = tuple(options)
        self._edit_selected = start
        self._edit_typed = ""
        self._edit_choices = ()
        self._input_active = False
        self._input_buffer = []
        self._input_cursor = 0
        self._notice_line = ""
        self._overlay_lines = []
        self._overlay_top = False

        def draw(selected):
            self._edit_selected = selected
            self.render()

        try:
            index = self._pick_index(draw, len(options), start,
                                     allow_exit_word=True)
            return None if index is None else options[index]
        except (EOFError, KeyboardInterrupt):
            return None
        finally:
            self._edit_active = False
            self._edit_choice_mode = False
            self._edit_title = ""
            self._edit_current = ""
            self._edit_choices = ()
            self._edit_options = ()
            self._edit_selected = 0
            self._edit_typed = ""
            self._input_active = False
            self._input_buffer = []
            self._input_cursor = 0
            self._notice_line = ""
            self._last_frame_size = None
            self.render()
            if self._active:
                self.start_scroll_listener()

    # -- full-screen input -------------------------------------------------

    def _read_tui_input(self, prompt_text="", mask=False, remember=True):
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
        # A focused setting editor must not recall previous chat commands.
        history = [] if self._edit_active else list(self._input_history)
        history_pos = len(history)
        draft = ""
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._notice_line = prompt_text.strip() if prompt_text.strip() else ""
        self._input_buffer = ["*"] * len(buffer) if mask else buffer
        self._input_cursor = cursor
        self._input_active = True
        self.render()

        def read_one():
            return os.read(fd, 1)

        def sync():
            # A masked prompt (an API key) keeps the real text in ``buffer``
            # but shows one star per character so nothing leaks on screen.
            self._input_buffer = ["*"] * len(buffer) if mask else buffer
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
            if (remember and text.strip() and (not self._input_history
                                              or self._input_history[-1] != text)):
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
            # Enable the enhanced keyboard protocols only while the prompt is
            # reading. Keeping them on during an LLM/tool call would encode
            # Ctrl+C as CSI-u and stop the terminal from sending SIGINT, so
            # Ctrl+C could not interrupt a thinking request.
            self._write("\x1b[>4;2m\x1b[>1u")
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
                if not raw:  # EOF
                    # Match the scrolling console: a final unterminated line is
                    # still input, not a discard. Only an empty buffer means
                    # "quit".
                    if buffer:
                        text = "".join(buffer)
                        self._input_buffer = []
                        self._input_cursor = 0
                        self._input_active = False
                        return text
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
                    if seq and not seq.startswith((b"[", b"O")):
                        # Esc followed quickly by a printable key is not an
                        # escape sequence; insert the key rather than swallowing
                        # it. Feeding the byte through the incremental decoder
                        # also keeps a multibyte UTF-8 character intact (the
                        # first byte may decode to ""; the rest arrives on the
                        # next loop iteration).
                        decoded = decoder.decode(seq)
                        text = "".join(c for c in decoded
                                       if c in ("\n", "\t") or c.isprintable())
                        if text:
                            buffer[cursor:cursor] = list(text)
                            cursor += len(text)
                            sync()
                        elif decoded:
                            self.dismiss_overlay()
                        continue
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
                            if self._edit_active:
                                return None
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
                                self._notice_line = ""
                                sync()
                            elif code == 116:            # Ctrl-T
                                if not self._edit_active:
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
                        if self._edit_active:
                            return None
                        self.dismiss_overlay()
                        continue
                    if seq in (b"[A", b"OA"):
                        if self._overlay_lines:
                            # A command/history overlay owns the arrows: scroll
                            # it instead of recalling prompt history.
                            self.scroll_body(1)
                        elif history:
                            if history_pos == len(history):
                                draft = "".join(buffer)
                            if history_pos > 0:
                                history_pos -= 1
                                set_buffer(history[history_pos])
                    elif seq in (b"[B", b"OB"):
                        if self._overlay_lines:
                            self.scroll_body(-1)
                        elif history_pos < len(history):
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
                    self._notice_line = ""
                    sync()
                    continue
                if raw == b"\x14":  # Ctrl-T: expand the most recent tool
                    if not self._edit_active:
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
            self._write("\x1b[<u\x1b[>4;0m")
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, previous)
            except (OSError, ValueError, termios.error):
                pass

    def _picker_window(self, header, body, selected, status=None):
        """Render a top-anchored picker window that follows ``selected``.

        The body is scrolled so the selected row stays visible; unlike command
        output, the window starts at the top instead of showing only its tail.
        ``status`` is an optional ``(text, colour)`` badge shown first so it
        survives truncation on narrow terminals.
        """
        visible = max(int(getattr(self, "_last_body_height", 1) or 1), 1)
        # Leave one row for the header when possible; at height 1 show the
        # selected row itself rather than a header with no room for the list.
        show_header = visible >= 2
        window = max(visible - 1, 1) if show_header else 1
        total = len(body)
        top = getattr(self, "_picker_top", 0)
        if selected < top:
            top = selected
        elif selected >= top + window:
            top = selected - window + 1
        top = max(0, min(top, max(0, total - window)))
        self._picker_top = top
        end = min(top + window, total)
        lines = body[top:end]
        if show_header:
            label = header
            if total:
                label += f"  {top + 1}-{end}/{total}"
            line = self.style(label, fg="cyan", bold=True)
            if status is not None:
                text, fg = status
                line = (self.style(str(text), fg=fg, bold=True)
                        + "  " + line)
            lines = [line] + lines
        elif status is not None:
            # A one-row body cannot fit a separate header; keep the status
            # badge and the selected row together.
            text, fg = status
            lines = [self.style(str(text), fg=fg, bold=True)
                     + "  " + (lines[0] if lines else "")]
        self._overlay_top = True
        self._overlay_lines = lines
        self.render()

    def _pick_index(self, draw, count, default=0, allow_exit_word=False,
                    on_left=None, on_right=None, on_save=None, on_all=None,
                    on_defaults=None):
        """Run a blocking raw-key picker and return a 0-based index.

        ``draw(selected)`` renders the overlay. Up/down move, a digit jumps to
        that row, left/right call the optional change callbacks, ``s`` calls
        ``on_save`` (when set), Enter confirms, and q/Esc cancels (returning
        ``None``). With ``allow_exit_word``, typing ``exit`` also cancels --
        used by the focused settings sub-window.
        """
        if not self._active or count <= 0:
            return None
        import select
        import termios

        selected = max(0, min(int(default), count - 1))
        # The scroll listener runs between prompts and also reads stdin. If it
        # stays alive here it steals the arrow-key bytes, so the picker sees
        # only (or part of) the ESC sequence and treats it as a cancel.
        self.stop_scroll_listener()
        fd = None
        previous = None
        try:
            try:
                fd = sys.stdin.fileno()
                previous = termios.tcgetattr(fd)
            except (AttributeError, OSError, ValueError, termios.error):
                return None
            attrs = termios.tcgetattr(fd)
            attrs[3] &= ~(termios.ICANON | termios.ECHO)
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSADRAIN, attrs)
            typed = []
            draw(selected)
            while True:
                ready, _, _ = select.select([fd], [], [], 1.0)
                if not ready:
                    continue
                raw = os.read(fd, 1)
                if not raw:
                    return None
                if raw in (b"q", b"Q"):
                    return None
                if raw in (b"a", b"A") and on_all is not None:
                    on_all()
                    draw(selected)
                    continue
                if raw in (b"d", b"D") and on_defaults is not None:
                    on_defaults()
                    draw(selected)
                    continue
                if raw in (b"s", b"S") and on_save is not None:
                    on_save()
                    draw(selected)
                    continue
                if allow_exit_word and raw.isalpha():
                    typed.append(raw.decode("utf-8", "ignore").lower())
                    if "".join(typed).endswith("exit"):
                        return None
                    self._edit_typed = "".join(typed[-12:])
                    draw(selected)
                    continue
                if allow_exit_word and raw in (b"\x7f", b"\b"):
                    if typed:
                        typed.pop()
                        self._edit_typed = "".join(typed[-12:])
                        draw(selected)
                    continue
                if raw == b"\x1b":
                    seq = _read_escape_sequence(fd, lambda: os.read(fd, 1),
                                                [], select)
                    if seq.endswith(b"A"):
                        selected = (selected - 1) % count
                        draw(selected)
                        continue
                    if seq.endswith(b"B"):
                        selected = (selected + 1) % count
                        draw(selected)
                        continue
                    if seq.endswith(b"D"):
                        if on_left is not None:
                            selected = on_left(selected)
                        draw(selected)
                        continue
                    if seq.endswith(b"C"):
                        if on_right is not None:
                            selected = on_right(selected)
                        draw(selected)
                        continue
                    return None
                if raw in b"123456789":
                    index = int(raw) - 1
                    if 0 <= index < count:
                        selected = index
                        draw(selected)
                    continue
                if raw in (b"\r", b"\n"):
                    return selected
        finally:
            if fd is not None and previous is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSADRAIN, previous)
                except (OSError, ValueError, termios.error):
                    pass
            self._overlay_lines = []
            self._overlay_top = False
            self._picker_top = 0
            self.render()
            if self._active:
                self.start_scroll_listener()

    def choose(self, title, options, default=1, prompt="> ",
               allow_exit=False):
        """Arrow-key picker for the fixed view; returns 1-based index or None."""
        if not self._active or not options:
            return super().choose(title, options, default=default,
                                  prompt=prompt, allow_exit=allow_exit)
        options = list(options)
        start = max(0, min(int(default) - 1, len(options) - 1))
        self._picker_top = 0

        def draw(selected):
            body = []
            for index, option in enumerate(options):
                marker = ">" if index == selected else " "
                body.append(f"{marker} {index + 1}. {option}")
            header = (f"[{title}] ↑/↓ or number, Enter select, q cancel"
                      if title else "[select] ↑/↓ or number, Enter, q")
            self._picker_window(header, body, selected)

        index = self._pick_index(draw, len(options), start,
                                 allow_exit_word=allow_exit)
        return None if index is None else index + 1

    def settings_pick(self, rows, immutable_lines=(), selected=0,
                      on_change=None, on_save=None, dirty=False,
                      on_defaults=None, on_all=None):
        """Show the settings table and return the selected row, or None.

        Up/down or a number select, Enter opens the focused editor, left/right
        rotate a toggle or fixed-choice value in place, ``s`` saves staged
        settings, and q/Esc cancels. ``on_change(row, value, rows)`` stages an
        inline change; it may replace the ``rows`` list so dependent rows (e.g.
        plan and plan-strategy) stay in sync. ``on_save(rows)`` persists the
        staged batch and refreshes the table. ``dirty`` may be a bool or a
        callable; the header shows a green ``saved`` or orange ``unsaved`` badge.
        ``a`` calls ``on_all(rows)`` (the CLI ``--all`` preset) and ``d``
        calls ``on_defaults(rows)`` to stage the defaults.
        """
        if not self._active or not rows:
            return None
        # Immutable settings are part of the same scrollable table so they can
        # be reached with the arrows on a small screen. They are selectable but
        # reject every edit.
        immutable_rows = []
        for offset, line in enumerate(immutable_lines):
            display = str(line)
            immutable_rows.append({
                "number": len(rows) + offset + 1,
                "name": display.split(":", 1)[0].strip(),
                "value": display,
                "display": display,
                "raw": None,
                "mutable": False,
                "cache": "none",
                "group": "Immutable / session",
                "spec": None,
                "immutable": True,
            })

        def combined():
            return list(rows) + immutable_rows

        total = len(rows) + len(immutable_rows)
        self._picker_top = 0
        start = max(0, min(int(selected), total - 1))

        def draw(selected):
            cols, _rows = self._size()
            available = max(cols - 4, 20)
            body = []
            selected_body = 0
            selectable = 0
            group_seen = None
            for row in combined():
                group = row.get("group", "Other")
                if group != group_seen:
                    group_seen = group
                    body.append(self.style(f"-- {group} --", fg="cyan",
                                           bold=True))
                if selectable == selected:
                    selected_body = len(body)
                marker = ">" if selectable == selected else " "
                if row.get("immutable"):
                    body.append(self.style(self._fit(
                        f"{marker} {row['number']:2}. {row['display']}",
                        available), dim=True))
                    selectable += 1
                    continue
                cache = (f"  cache:{row['cache']}"
                         if row.get("cache", "none") != "none" else "")
                # Shrink the columns on narrow terminals instead of letting
                # the table run past the visible body. Drop the cache suffix
                # first when it would squeeze the value column.
                prefix = 6          # marker + space + "NN. "
                if cache and available - prefix - len(cache) < 40:
                    cache = ""
                remaining = max(available - prefix - len(cache), 19)
                name_w = min(22, max(8, remaining // 3))
                value_w = max(8, remaining - name_w - 1)
                name = self._fit(row["name"], name_w)
                value = self._fit(row.get("display", row["value"]), value_w)
                body.append(self._fit(
                    f"{marker} {row['number']:2}. {name:<{name_w}} "
                    f"{value:<{value_w}}{cache}", available))
                selectable += 1
            is_dirty = dirty() if callable(dirty) else bool(dirty)
            status = (("unsaved", "yellow") if is_dirty
                      else ("saved", "green"))
            self._picker_window(
                "↑↓←→ Enter · s save · a --all · d default · q",
                body, selected_body, status=status)

        def change(selected, step):
            row = combined()[selected]
            spec = row.get("spec")
            if spec is None or not row.get("mutable") or not spec.mutable:
                return selected
            if spec.kind == "bool":
                value = not bool(row.get("raw"))
            elif spec.kind == "choice" and spec.choices:
                choices = list(spec.choices)
                current = row.get("raw")
                index = choices.index(current) if current in choices else 0
                value = choices[(index + step) % len(choices)]
            else:
                return selected
            if on_change is not None:
                on_change(row, value, rows)
            else:
                row["raw"] = value
                row["value"] = (("on" if value else "off")
                                if spec.kind == "bool" else str(value))
            return selected

        def left(selected):
            return change(selected, -1)

        def right(selected):
            return change(selected, 1)

        def save():
            if on_save is not None:
                on_save(rows)

        def defaults():
            if on_defaults is not None:
                on_defaults(rows)

        def all_preset():
            if on_all is not None:
                on_all(rows)

        index = self._pick_index(
            draw, total, start, on_left=left, on_right=right,
            on_save=save if on_save is not None else None,
            on_all=all_preset if on_all is not None else None,
            on_defaults=defaults if on_defaults is not None else None)
        return None if index is None else combined()[index]

    def read_secret(self, prompt_text="API key: "):
        """Read a secret into the fixed-view input line without echoing it."""
        if not self._active:
            return super().read_secret(prompt_text)
        self.stop_scroll_listener()
        try:
            return self._read_tui_input(prompt_text, mask=True, remember=False)
        except (EOFError, KeyboardInterrupt):
            self._input_active = False
            return None
        finally:
            if self._question_active:
                self._question_active = False
                self._question_lines = []
                self.render()
            if self._active:
                self.start_scroll_listener()

    def read_line(self, prompt_text="", multiline=False, tui=False):
        if not self._active:
            return super().read_line(prompt_text, multiline=multiline, tui=tui)
        self.stop_scroll_listener()
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
            if self._active:
                self.start_scroll_listener()
