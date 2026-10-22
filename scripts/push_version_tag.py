#!/usr/bin/env python3
"""Tag the current commit with the version in ``pyproject.toml`` and push it.

Releasing used to mean remembering two things by hand: bump ``project.version``
*and* create the matching ``v<version>`` tag before pushing. Forgetting the
second is silent -- the release exists only in the working tree until someone
notices -- and re-running a tag push by hand fails with an unhelpful
"already exists" once the tag is there. So this script does the mechanical part
and is safe to run repeatedly:

  * the version comes from ``pyproject.toml`` (the same source of truth the
    welcome banner reads), never from a CLI argument, so a tag can never
    disagree with the released code;
  * if the tag is **already on the remote** it prints one line and exits ``0``
    (nothing to do is the normal, expected outcome of a second run);
  * otherwise it creates the annotated tag at ``HEAD`` (or reuses an existing
    local tag that already points there) and pushes just that tag.

It refuses to push a local tag that points at a *different* commit: that means
the version was bumped without a new tag name, and pushing it would move the
release backwards.

Standard library only, like ``scripts/lint.py``, because this has to work on a
fresh checkout with nothing installed.

Usage::

    python3 scripts/push_version_tag.py              # tag HEAD and push it
    python3 scripts/push_version_tag.py --dry-run    # say what it would do
    python3 scripts/push_version_tag.py --remote up  # use another remote

Exits ``0`` when the tag is on the remote (pushed now or already there) and
``1`` on any refusal or git failure, always with a message saying why.
"""

import argparse
import os
import re
import subprocess
import sys

#: Where the version lives, relative to the repository root.
PYPROJECT = "pyproject.toml"

#: ``version = "1.2.3"`` inside the ``[project]`` table (the 3.10 fallback).
VERSION_RE = re.compile(r"""^\s*version\s*=\s*["']([^"']+)["']""")

#: A version we are willing to turn into a tag name.
VERSION_OK_RE = re.compile(r"^[0-9][0-9A-Za-z.+_-]*$")

#: Repository root, set by :func:`main`. Every git call runs there, so ``origin``
#: and the relative paths mean the same thing however the script is invoked.
ROOT = ""


class TagError(Exception):
    """A refusal or a failure, carrying the message to show the user."""


def git(*args):
    """Run a git command at the repository root; return its stdout.

    Raises :class:`TagError` with git's own stderr on a non-zero exit, so the
    user sees *why* a command failed instead of a bare traceback.
    """
    result = subprocess.run(["git", *args], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise TagError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def repo_root():
    """The top level of the working tree this script lives in."""
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                            text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise TagError(f"not inside a git working tree: {detail}")
    return result.stdout.strip()


def read_version(path):
    """Return ``project.version`` from ``pyproject.toml``.

    ``tomllib`` (Python 3.11+) parses the file properly. On 3.10 -- which this
    project still supports -- a narrow regex over the ``[project]`` table stands
    in: it follows the section headers, so a ``version`` key in some other table
    is ignored rather than mistaken for the release number. Either way a missing
    or unusable version is an error, never a guess.
    """
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as e:
        raise TagError(f"cannot read {path}: {e}")

    version = ""
    try:
        import tomllib  # Python >= 3.11
    except ImportError:  # pragma: no cover - exercised on 3.10 only
        tomllib = None
    if tomllib is not None:
        try:
            version = str(tomllib.loads(raw.decode("utf-8"))["project"]["version"])
        except (KeyError, TypeError):
            raise TagError(f"{path} has no version in its [project] table")
        except Exception as e:
            raise TagError(f"cannot parse {path}: {e}")
    else:  # pragma: no cover - exercised on 3.10 only
        section = ""
        for line in raw.decode("utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped
                continue
            if section != "[project]":
                continue
            match = VERSION_RE.match(line)
            if match:
                version = match.group(1)
                break
        if not version:
            raise TagError(f"{path} has no version in its [project] table")

    version = version.strip()
    if not VERSION_OK_RE.match(version):
        raise TagError(f"refusing to tag an odd version string: {version!r}")
    return version


def remote_tag(tag, remote):
    """``(present, commit)`` for ``tag`` on ``remote``.

    ``ls-remote`` is the only reliable "is it already there?" check: listing
    local tags says nothing about the remote, and letting ``git push`` fail
    instead would mean reading "already exists" out of an error message.

    ``commit`` is the commit the tag describes -- the peeled ``^{}`` line that
    ``ls-remote`` emits for an annotated tag -- or ``""`` when it cannot be
    determined. The refspec is a *glob* on purpose: given the exact ref, git
    prints only the tag object's own sha, which never matches a commit and so
    would make every annotated tag look like it describes nothing.
    """
    out = git("ls-remote", "--tags", remote, f"refs/tags/{tag}*")
    present, commit = False, ""
    for line in out.splitlines():
        sha, _, ref = line.partition("\t")
        ref = ref.strip()
        if ref == f"refs/tags/{tag}^{{}}":
            present, commit = True, sha.strip()
        elif ref == f"refs/tags/{tag}":
            present = True
    return present, commit


def local_tag_commit(tag):
    """The commit the *local* ``tag`` points at, or ``""`` if there is none."""
    result = subprocess.run(["git", "rev-parse", "-q", "--verify",
                             f"refs/tags/{tag}^{{commit}}"],
                            cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def push_tag(version, remote="origin", dry_run=False):
    """Ensure ``v<version>`` is on ``remote``; return the process exit code."""
    tag = f"v{version}"
    head = git("rev-parse", "HEAD")

    print(f"version {version} (from {PYPROJECT}) -> tag {tag}")

    present, on_remote = remote_tag(tag, remote)
    if present:
        # The interesting (and expected) case on a second run. Say *where* it
        # points: a tag left over from an earlier commit is otherwise invisible,
        # and it is exactly what a forgotten version bump looks like.
        where = f" at {on_remote[:12]}" if on_remote else ""
        note = ""
        if on_remote and on_remote != head:
            note = (f" -- note: that is not HEAD ({head[:12]}); bump the version "
                    f"in {PYPROJECT} if this commit needs its own tag")
        print(f"{tag} already exists on {remote}{where}{note}; skipping.")
        return 0

    local = local_tag_commit(tag)
    if local and local != head:
        raise TagError(
            f"local tag {tag} points at {local[:12]}, but HEAD is {head[:12]}.\n"
            f"Refusing to push a tag that does not describe this commit.\n"
            f"Inspect it with: git show {tag}\n"
            f"If it is wrong, move it with: git tag -f -a {tag} -m 'release "
            f"{version}' {head[:12]}")

    if dry_run:
        action = "would push existing" if local else "would create and push"
        print(f"dry run: {action} {tag} at {head[:12]} to {remote}.")
        return 0

    if local:
        print(f"local tag {tag} already points at {head[:12]}; pushing it.")
    else:
        git("tag", "-a", tag, "-m", f"release {version}", head)
        print(f"created annotated tag {tag} at {head[:12]}.")

    git("push", remote, f"refs/tags/{tag}")
    print(f"pushed {tag} to {remote}.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Tag the current commit with the version in "
                    f"{PYPROJECT} and push the tag.")
    parser.add_argument("--remote", default="origin",
                        help="remote to push the tag to (default: origin)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would happen without creating or "
                             "pushing anything")
    args = parser.parse_args(argv)

    global ROOT
    try:
        ROOT = repo_root()
        version = read_version(os.path.join(ROOT, PYPROJECT))
        return push_tag(version, remote=args.remote, dry_run=args.dry_run)
    except TagError as e:
        print(f"push_version_tag: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
