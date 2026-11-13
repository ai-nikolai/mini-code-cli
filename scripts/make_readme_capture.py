#!/usr/bin/env python3
"""Regenerate the terminal capture embedded in the README.

The asset is an SVG rather than a PNG so it stays crisp on GitHub, needs no
image library, and can be regenerated from this repository alone:

    python3 scripts/make_readme_capture.py

The text is produced by the real :class:`mini_code_cli.ui.Console` renderer (the
welcome banner, conversation turns, tool calls, usage line and a notice), so
the capture cannot drift away from the shipped interface.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_code_cli import __version__, ui  # noqa: E402

OUT = ROOT / "docs" / "assets" / "mini-code-demo.svg"
OUT_PNG = ROOT / "docs" / "assets" / "mini-code-demo.png"
FONT = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono-Bold.ttf"

# GitHub-dark terminal palette.
FG = {
    "black": "#484f58",
    "red": "#ff7b72",
    "green": "#3fb950",
    "yellow": "#d29922",
    "blue": "#58a6ff",
    "magenta": "#bc8cff",
    "cyan": "#39c5cf",
    "white": "#c9d1d9",
    "gray": "#8b949e",
    "default": "#c9d1d9",
}
BG = {
    "black": "#161b22",
    "red": "#3d1518",
    "green": "#12351d",
    "yellow": "#3d2d0a",
    "blue": "#10243d",
    "magenta": "#2a1740",
    "cyan": "#0d3235",
}
_ANSI = re.compile(r"\033\[([0-9;]*)m")


def _capture() -> str:
    """One representative session, rendered by the real console."""
    stream = io.StringIO()
    console = ui.Console(stream=stream, color=True, unicode=True,
                         width=96, verbose=True)
    console.welcome(
        __version__,
        "a1b2c3d4",
        "~/.cache/mini/sessions/a1b2c3d4",
        "~/projects/demo",
        "http://localhost:30000",
        "deepseek-v4-flash",
        "auto (safe)",
        "manual",
        "on",
        compaction_strategy="summarise",
        plan_strategy="plan-verify-step",
        plan_mode=False,
    )
    console.rule(title="turn 1", char="═", fg="cyan")
    console.turn("You", "Add a /health endpoint to app.py and a test for it.",
                 accent="cyan")
    console.turn(
        "Assistant",
        "I'll inspect the app, add the endpoint, then run the tests.",
        accent="green",
    )
    console.tool_group("read_file", {"file_path": "app.py"}, "18 lines")
    console.tool_group(
        "edit",
        {"file_path": "app.py",
         "old_string": 'return {"status": "ok"}',
         "new_string": 'return {"status": "ok", "service": "demo"}'},
        "replaced 1 occurrence",
    )
    console.tool_group("run_shell_command", {"command": "pytest -q"},
                       "2 passed in 0.31s")
    console.usage_line(
        {"prompt_tokens": 18_420, "completion_tokens": 412,
         "tool_calls": 3, "total_tokens": 18_832},
        llm_calls=7,
    )
    console.note("Tip: /help lists commands · /history browses turns.", "info")
    return stream.getvalue()


def _runs(text: str):
    """Split one line into ``(text, style)`` runs, tracking SGR state."""
    style = {"bold": False, "dim": False, "fg": None, "bg": None}
    runs = []
    pos = 0
    for match in _ANSI.finditer(text):
        if match.start() > pos:
            runs.append((text[pos:match.start()], dict(style)))
        codes = [c for c in match.group(1).split(";") if c]
        for code in codes:
            value = int(code)
            if value == 0:
                style = {"bold": False, "dim": False, "fg": None, "bg": None}
            elif value == 1:
                style["bold"] = True
            elif value == 2:
                style["dim"] = True
            elif value in (3, 4):
                continue
            elif 30 <= value <= 37:
                style["fg"] = next(
                    name for name, n in ui._FG.items() if n == value)
            elif 90 <= value <= 97:
                style["fg"] = {90: "gray", 91: "red", 92: "green",
                               93: "yellow", 94: "blue", 95: "magenta",
                               96: "cyan", 97: "white"}[value]
            elif 40 <= value <= 47:
                style["bg"] = next(
                    name for name, n in ui._BG.items() if n == value)
        pos = match.end()
    if pos < len(text):
        runs.append((text[pos:], dict(style)))
    return runs


def _html_escape(text: str) -> str:
    return escape(text).replace(" ", "&#160;")


def _svg(text: str) -> str:
    lines = text.rstrip("\n").split("\n")
    cols = max((ui._visual_len(line) for line in lines), default=1)
    cell_w, line_h = 8.45, 20
    pad_x, pad_y, header = 26, 46, 34
    width = int(cols * cell_w + pad_x * 2)
    height = int(len(lines) * line_h + pad_y + header)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="DejaVu Sans Mono, Menlo, Consolas, monospace" '
        f'font-size="14">',
        '<defs>',
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="8" stdDeviation="12" '
        'flood-color="#000" flood-opacity="0.35"/>',
        '</filter>',
        '</defs>',
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" '
        'fill="#0d1117" filter="url(#shadow)"/>',
        f'<rect x="0" y="0" width="{width}" height="{header}" rx="10" '
        'fill="#161b22"/>',
        f'<rect x="0" y="{header - 10}" width="{width}" height="10" '
        'fill="#161b22"/>',
        '<circle cx="20" cy="17" r="5.5" fill="#ff5f56"/>',
        '<circle cx="38" cy="17" r="5.5" fill="#ffbd2e"/>',
        '<circle cx="56" cy="17" r="5.5" fill="#27c93f"/>',
        f'<text x="{width // 2}" y="22" fill="#8b949e" font-size="12" '
        'text-anchor="middle">mini-code — ~/projects/demo</text>',
    ]
    y = header + 20
    for line in lines:
        if line:
            spans = []
            for run, style in _runs(line):
                if not run:
                    continue
                fill = FG.get(style.get("fg") or "default", FG["default"])
                attrs = [f'fill="{fill}"']
                if style.get("bold"):
                    attrs.append('font-weight="600"')
                if style.get("dim"):
                    attrs.append('opacity="0.68"')
                if style.get("bg"):
                    # SVG has no per-tspan background; a faint underline is a
                    # readable hint without changing the monospace grid.
                    attrs.append(f'text-decoration="underline"')
                    attrs.append(
                        f'style="text-decoration-color:'
                        f'{BG.get(style["bg"], "#30363d")}"')
                spans.append(
                    f'<tspan {" ".join(attrs)}>{_html_escape(run)}</tspan>')
            parts.append(
                f'<text x="{pad_x}" y="{y}" xml:space="preserve">'
                + "".join(spans) + "</text>")
        y += line_h
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _png(text: str) -> bool:
    """Render the same capture as a PNG, when Pillow is available."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False

    lines = text.rstrip("\n").split("\n")
    font = ImageFont.truetype(FONT, 14)
    bold = ImageFont.truetype(FONT_BOLD, 14)
    title_font = ImageFont.truetype(FONT, 12)
    pad_x, header, line_h = 26, 34, 20
    body_w = max((font.getlength(ui._ANSI_RE.sub("", line))
                  for line in lines), default=1)
    width = int(body_w + pad_x * 2)
    height = int(header + len(lines) * line_h + 18)

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width, height), radius=10, fill="#0d1117")
    draw.rounded_rectangle((0, 0, width, header), radius=10, fill="#161b22")
    draw.rectangle((0, header - 10, width, header), fill="#161b22")
    for cx, color in ((20, "#ff5f56"), (38, "#ffbd2e"), (56, "#27c93f")):
        draw.ellipse((cx - 5.5, 11.5, cx + 5.5, 22.5), fill=color)
    title = "mini-code — ~/projects/demo"
    draw.text(((width - title_font.getlength(title)) / 2, 10),
              title, font=title_font, fill="#8b949e")

    def blend(color):
        fg = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        bg = (13, 17, 23)
        return tuple(int(bg[j] + (fg[j] - bg[j]) * 0.68) for j in range(3))

    y = header + 8
    for line in lines:
        x = pad_x
        for run, style in _runs(line):
            if not run:
                continue
            color = FG.get(style.get("fg") or "default", FG["default"])
            fill = blend(color) if style.get("dim") else color
            use = bold if style.get("bold") else font
            if style.get("bg"):
                bg = BG.get(style["bg"], "#30363d")
                draw.rounded_rectangle(
                    (x - 2, y + 1, x + font.getlength(run) + 2, y + line_h - 1),
                    radius=3, fill=bg)
            draw.text((x, y), run, font=use, fill=fill)
            x += font.getlength(run)
        y += line_h
    OUT_PNG = ROOT / "docs" / "assets" / "mini-code-demo.png"
    image.save(OUT_PNG)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")
    return True


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    capture = _capture()
    OUT.write_text(_svg(capture), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    _png(capture)


if __name__ == "__main__":
    main()
