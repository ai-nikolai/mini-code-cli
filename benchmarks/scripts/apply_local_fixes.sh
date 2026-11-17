#!/usr/bin/env bash
# Re-apply the benchmark-local fixes that live in git-ignored checkouts.
#
# `external_repos/harbor` and `external_repos/terminal-bench-2-1` are ignored
# clones. A fresh setup can therefore lose these small compatibility fixes. This
# script is idempotent and safe to run after setup:
#
#   benchmarks/scripts/apply_local_fixes.sh
#
# It does not touch task assertions, timeouts, memory limits or the task set.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 - "$ROOT" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
changed = []


def replace_once(path, old, new, marker):
    if not path.is_file():
        return
    text = path.read_text()
    if marker in text:
        return
    if old not in text:
        print(f"  ! pattern not found in {path}", file=sys.stderr)
        return
    path.write_text(text.replace(old, new, 1))
    changed.append(path)


# 1. Harbor's Pi agent must separate the prompt from options. A task whose
#    instruction starts with '-' was parsed as a Pi option ("Unknown option").
pi = root / "external_repos/harbor/src/harbor/agents/installed/pi.py"
replace_once(
    pi,
    '                f"{cli_flags}"\n'
    '                f"{escaped_instruction} "\n',
    '                f"{cli_flags}"\n'
    '                f"-- {escaped_instruction} "\n',
    'f"-- {escaped_instruction} "',
)

# 2. Do not classify arbitrary agent prose ("rate limit") as an API rate limit,
#    and report container memory kills as ordinary non-zero exits.
base = root / "external_repos/harbor/src/harbor/agents/installed/base.py"
old_patterns = (
    '        ErrorPattern(r"rate.?limit", ApiRateLimitError),\n'
    '        ErrorPattern(r"too many requests", ApiRateLimitError),\n'
)
new_patterns = (
    '        # Do not match the bare phrase "rate limit": agents discuss it in\n'
    '        # their reasoning, which used to hide a memory kill as a rate limit.\n'
    '        ErrorPattern(\n'
    '            r"(?i)(?:API Error:\\s*429|HTTP(?:\\s+status)?[:\\s]*429|"\n'
    '            r"status(?: code)?[:\\s]*429|rate.?limit(?:[ _-]?(?:exceeded|reached))|"\n'
    '            r"too many requests)",\n'
    '            ApiRateLimitError,\n'
    '        ),\n'
    '        ErrorPattern(\n'
    '            r"(?i)MemoryLimitExceededError|Memory limit exceeded",\n'
    '            NonZeroAgentExitCodeError,\n'
    '        ),\n'
)
replace_once(base, old_patterns, new_patterns,
             "rate.?limit(?:[ _-]?(?:exceeded|reached))")

# 3. fix-code-vulnerability installs pytest at verify time; use the same
#    interpreter for pip and pytest. The bare `pip`/`python` pair can point at
#    different interpreters in the SIF, leaving `--ctrf` unrecognised.
for path in (
    root / "external_repos/terminal-bench-2-1/fix-code-vulnerability/tests/test.sh",
    root / "experiments/tb-staged/fix-code-vulnerability/tests/test.sh",
):
    if not path.is_file():
        continue
    text = path.read_text()
    original = text
    if "python -m pip install pytest==8.4.1" not in text:
        text = text.replace(
            "pip install pytest==8.4.1 pytest-json-ctrf==0.3.5",
            "python -m pip install pytest==8.4.1 pytest-json-ctrf==0.3.5",
        )
    text = text.replace("python -m pytest -p ctrf.main --ctrf",
                        "python -m pytest --ctrf")
    if text != original:
        path.write_text(text)
        changed.append(path)

if changed:
    for path in changed:
        print(f"patched: {path.relative_to(root)}")
else:
    print("local benchmark fixes already applied")
PY

# The Claude adapter lives in the tracked tree, not external_repos.
echo "tracked: benchmarks/agents/claude_code.py (CLAUDE_CODE_MAX_OUTPUT_TOKENS)"
