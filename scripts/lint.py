#!/usr/bin/env python3
"""A tiny, dependency-free linter for this repo.

Three checks, all from the standard library -- nothing to install:

1. every ``*.py`` file **compiles** (``py_compile``);
2. its indentation is **unambiguous** (``tabnanny``: a tab that could be read
   as spaces is exactly the kind of bug that is invisible in a diff);
3. the text is **tidy**: no trailing whitespace, no tab characters anywhere,
   and a newline at end of file.

Usage::

    python3 scripts/lint.py                 # the package, the tests, scripts/
    python3 scripts/lint.py a.py b/ c.py    # specific files / directories

Relative paths are resolved against the repository root, so the command is the
same from anywhere. Exits ``1`` if anything was reported, ``0`` if clean.
"""

import contextlib
import io
import os
import py_compile
import sys
import tabnanny
import tempfile

#: Directories linted when no target is given (relative to the repo root).
DEFAULT_TARGETS = ("mini_code_cli", "benchmarks", "tests", "scripts")

#: Directories never worth walking into. ``benchmarks/`` keeps generated trees
#: (run artifacts, staged benchmark tasks, derived images) that are git-ignored
#: and are not this repository's code -- linting vendored task files is noise.
SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "runs", "results", "datasets", "vendor",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def python_files(target):
    """Yield every ``.py`` file named by ``target`` (a file or a directory)."""
    if os.path.isfile(target):
        yield target
        return
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def rel(path):
    """``path`` relative to the repo root when it lives inside it, else as-is."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:              # different drive on Windows
        return path


def check_compiles(path, scratch):
    """Return one message if ``path`` does not compile, else an empty list."""
    out = os.path.join(scratch, "out.pyc")
    try:
        py_compile.compile(path, cfile=out, doraise=True)
        return []
    except py_compile.PyCompileError as e:
        return [f"{rel(path)}: does not compile: {e.msg.strip()}"]
    except OSError as e:
        return [f"{rel(path)}: could not be compiled: {e}"]
    finally:
        try:
            os.unlink(out)
        except OSError:
            pass


def check_indentation(path):
    """Return one message per ambiguous-indentation problem tabnanny reports."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            tabnanny.check(path)
    except Exception as e:                     # best-effort: never crash the lint
        return [f"{rel(path)}: tabnanny failed: {e}"]
    return [f"{rel(path)}: {line.strip()}"
            for line in buf.getvalue().splitlines() if line.strip()]


def check_text(path):
    """Return the tidy-text findings (trailing space, tabs, missing final \\n)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    problems = []
    if text and not text.endswith("\n"):
        problems.append(f"{rel(path)}: missing newline at end of file")
    for number, line in enumerate(text.splitlines(), 1):
        if line != line.rstrip():
            problems.append(f"{rel(path)}:{number}: trailing whitespace")
        if "\t" in line:
            problems.append(f"{rel(path)}:{number}: tab character")
    return problems


def main(argv):
    targets = argv or list(DEFAULT_TARGETS)
    paths, missing = [], []
    for target in targets:
        full = target if os.path.isabs(target) else os.path.join(ROOT, target)
        if os.path.exists(full):
            paths.extend(python_files(full))
        else:
            missing.append(target)

    problems = [f"lint: no such file or directory: {m}" for m in missing]
    scratch = tempfile.mkdtemp(prefix="mini-lint-")
    try:
        for path in paths:
            problems.extend(check_compiles(path, scratch))
            problems.extend(check_indentation(path))
            problems.extend(check_text(path))
    finally:
        try:
            os.rmdir(scratch)
        except OSError:
            pass

    for problem in problems:
        print(problem)
    print(f"lint: {len(paths)} file(s) checked, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
