"""Detect the patched upstream checkouts this repository is meant to run against.

We carry our own branches of three upstream projects -- ``harbor`` (the
Terminal-Bench harness), ``SWE-bench`` (the evaluation harness) and
``terminal-bench-2`` (the task set) -- because two behaviours cannot be expressed
from the outside: harbor's Apptainer backend needed to accept a task that only
ships a Dockerfile, and ``swebench.harness.run_evaluation`` needed a container
backend that is not the Docker SDK.

The checkouts are *not* vendored into this history (they are large, and their
branches -- pushed to github.com/ai-nikolai -- are the source of truth; harbor is
installed straight from its branch by ``benchmarks/requirements.txt``). This module
answers the one question the suites have: *is the patched code what will actually
run?* -- by checking which file the imported module came from, not by trusting a
directory to exist.
"""

import functools
import importlib.util
import os
import subprocess

#: checkout directory name (relative to the repository root) -> which upstream
CHECKOUTS = {
    "harbor": "harbor",
    "swebench": "SWE-bench",
    "terminal_bench": "terminal-bench-2-1",
}

#: the branch we build on, per checkout (see benchmarks/scripts/setup_external_repos.sh)
BRANCHES = {
    "harbor": "mini-code/apptainer",
    "swebench": "mini-code/apptainer",
    "terminal_bench": "mini-code/bench",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root():
    return _REPO_ROOT


def checkout_path(name):
    """The checkout directory for ``name``, or ``None`` when it is absent."""
    path = os.path.join(_REPO_ROOT, CHECKOUTS[name])
    return path if os.path.isdir(path) else None


def _git(path, *args):
    try:
        result = subprocess.run(["git", "-C", path, *args], capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def state(name):
    """``{repo, path, branch, sha, dirty, present}`` for one checkout."""
    path = checkout_path(name)
    if path is None:
        return {"repo": CHECKOUTS[name], "path": None, "present": False,
                "branch": None, "sha": None, "dirty": None}
    return {
        "repo": CHECKOUTS[name],
        "path": path,
        "present": True,
        "branch": _git(path, "rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _git(path, "rev-parse", "--short=12", "HEAD"),
        "dirty": bool(_git(path, "status", "--porcelain")),
    }


def _imported_from(module):
    """The file a module resolves to, or ``None`` when it is not importable."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    origin = spec.origin or ""
    return origin if origin.endswith(".py") else None


def installed_from_checkout(name):
    """True when the *imported* package comes from our checkout, not PyPI.

    This is stronger than "the directory exists": an editable install of the fork
    and a stale site-packages copy look identical from the outside, and running the
    wrong one is exactly the failure this module exists to prevent.
    """
    path = checkout_path(name)
    if path is None:
        return False
    module = {"harbor": "harbor", "swebench": "swebench"}.get(name)
    if module is None:
        return True
    origin = _imported_from(module)
    return bool(origin and os.path.commonpath([path, origin]) == path)


def harbour_fork_active():
    """The patched harbor (Dockerfile-only tasks, apptainer cache dirs) is in use."""
    return installed_from_checkout("harbor")


def swebench_fork_active():
    """The patched swebench (``--backend apptainer``) is in use."""
    if not installed_from_checkout("swebench"):
        return False
    try:
        from swebench.harness.container_backends import BACKENDS
    except Exception:  # noqa: BLE001 - an unpatched checkout has no such module
        return False
    return "apptainer" in BACKENDS


def _interpreter_for(binary):
    """The Python that a console script uses (its shebang), or ``None``."""
    try:
        with open(binary, "rb") as handle:
            first = handle.readline().decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    return parts[0] if parts else None


@functools.lru_cache(maxsize=8)
def cli_origin(binary):
    """Where the ``harbor`` console script imports its package from.

    The CLI runs in its *own* interpreter, so ``import harbor`` here says nothing
    about it: harbor may be installed editable in the env that owns the script
    while this process cannot import it at all.
    """
    if not binary:
        return None
    python = _interpreter_for(binary)
    if not python or not os.path.exists(python):
        return None
    try:
        result = subprocess.run(
            [python, "-c", "import harbor; print(harbor.__file__)"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def cli_uses_fork(binary):
    """True when the ``harbor`` script on PATH imports our checkout."""
    origin = cli_origin(binary)
    path = checkout_path("harbor")
    return bool(origin and path and os.path.commonpath([path, origin]) == path)


def summary():
    """Every checkout's state, plus whether the patched code is what will run."""
    payload = {name: state(name) for name in CHECKOUTS}
    payload["harbor"]["active"] = harbour_fork_active()
    payload["swebench"]["active"] = swebench_fork_active()
    payload["terminal_bench"]["active"] = payload["terminal_bench"]["present"]
    return payload


def describe():
    """Human-readable lines for ``plan``/preflight, one per checkout."""
    lines = []
    for name, info in summary().items():
        if not info["present"]:
            lines.append(f"{name}: no checkout ({CHECKOUTS[name]}/) -- "
                         f"falling back to the in-repo adapter")
            continue
        branch = info["branch"] or "detached"
        dirty = " (dirty)" if info["dirty"] else ""
        state_text = "in use" if info["active"] else "present, not imported"
        lines.append(f"{name}: {CHECKOUTS[name]} @ {branch} {info['sha']}"
                     f"{dirty} -- {state_text}")
    return lines


def preflight_problems():
    """Blocking problems, phrased as what to do about them."""
    problems = []
    if not harbour_fork_active() and checkout_path("harbor"):
        problems.append(
            "the harbor checkout exists but is not the installed package; install "
            "it with `pip install -e harbor --no-deps` in the environment that runs "
            "harbor, or the unpatched PyPI release will be used")
    if not swebench_fork_active() and checkout_path("swebench"):
        problems.append(
            "the SWE-bench checkout exists but its container backends are not "
            "importable; install it with `pip install -e SWE-bench --no-deps`")
    return problems
