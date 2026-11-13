#!/usr/bin/env python3
"""Regenerate the animated terminal demo used for promotion.

The frames are produced by the real console renderer, then rendered with
Pillow (the only optional dependency):

    python3 scripts/make_readme_gif.py

If Pillow is not installed the PNG/SVG capture script still works; this GIF
step is skipped.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from mini_code_cli import __version__, ui  # noqa: E402
from make_readme_capture import render_image  # noqa: E402

OUT = ROOT / "docs" / "assets" / "mini-code-demo.gif"
PROMPT = "Add a /health endpoint to app.py and a test for it."
ASSISTANT = "I'll inspect the app, add the endpoint, then run the tests."


def _console():
    return ui.Console(stream=io.StringIO(), color=True, unicode=True,
                      width=96, verbose=True)


def _welcome(console):
    console.welcome(
        __version__, "a1b2c3d4",
        "~/.cache/mini/sessions/a1b2c3d4",
        "~/projects/demo", "http://localhost:30000",
        "deepseek-v4-flash", "auto (safe)", "manual", "on",
        compaction_strategy="summarise",
        plan_strategy="plan-verify-step", plan_mode=False,
    )


def screen(chars=None, phase="done"):
    console = _console()
    _welcome(console)
    if chars is not None:
        caret = "▌" if chars < len(PROMPT) else ""
        console.emit(console.style("You › ", bold=True, fg="cyan")
                     + PROMPT[:chars] + caret)
        return console.stream.getvalue()

    console.rule(title="turn 1", char="═", fg="cyan")
    console.turn("You", PROMPT, accent="cyan")
    if phase in ("thinking", "assistant", "tools", "done"):
        dots = {"thinking": ".", "assistant": "..", "tools": "...",
                "done": ""}.get(phase, "")
        console.emit(console.style(f"… Thinking{dots}", dim=True, fg="cyan"))
    if phase in ("assistant", "tools", "done"):
        console.turn("Assistant", ASSISTANT, accent="green")
    if phase in ("tools", "done"):
        console.tool_group("read_file", {"file_path": "app.py"}, "18 lines")
    if phase == "done":
        console.tool_group(
            "edit",
            {"file_path": "app.py",
             "old_string": 'return {"status": "ok"}',
             "new_string": 'return {"status": "ok", "service": "demo"}'},
            "replaced 1 occurrence")
        console.tool_group("run_shell_command", {"command": "pytest -q"},
                           "2 passed in 0.31s")
        console.usage_line(
            {"prompt_tokens": 18_420, "completion_tokens": 412,
             "tool_calls": 3, "total_tokens": 18_832}, llm_calls=7)
        console.note("Tip: /help lists commands · /history browses turns.",
                     "info")
    return console.stream.getvalue()


def main():
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; skipping the GIF")
        return

    final = screen()
    min_lines = len(final.rstrip("\n").split("\n"))
    frames, durations = [], []

    def add(text, ms):
        frames.append(render_image(text, min_lines=min_lines, min_cols=96))
        durations.append(ms)

    add(screen(chars=0), 700)
    for chars in (8, 16, 24, 32, 40, 48, len(PROMPT)):
        add(screen(chars=chars), 110)
    add(screen(chars=len(PROMPT)), 550)
    for phase, ms in (("thinking", 320), ("assistant", 420),
                      ("tools", 520), ("done", 1800)):
        add(screen(phase=phase), ms)

    paletted = [f.convert("RGB").convert(
        "P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        OUT, save_all=True, append_images=paletted[1:],
        duration=durations, loop=0, disposal=2, optimize=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
