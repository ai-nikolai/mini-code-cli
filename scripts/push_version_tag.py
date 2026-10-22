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

Two more refusals protect the release *workflow*, which only publishes tags
whose commit is on ``main`` (``.github/workflows/release.yml`` re-checks this,
but failing there costs a whole CI run and leaves a red mark on the repo):

  * ``HEAD`` must be **exactly the tip** of ``<remote>/main`` (fetched fresh,
    not the possibly stale local ref). A commit that is merely *on* main is not
    enough -- tagging an older commit would release code that is not what main
    points at -- and a commit that is not on main at all would be rejected by
    CI, so the script says so here instead;
  * ``pyproject.toml`` must have no uncommitted changes: the version is read
    from the working tree, so tagging with a dirty file could publish a version
    that exists in no commit.

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


def current_branch():
    """The checked-out branch name, or ``"HEAD"`` when it is detached."""
    try:
        return git("rev-parse", "--abbrev-ref", "HEAD")
    except TagError:
        return "HEAD"


def is_ancestor(commit, descendant):
    """True when ``commit`` is reachable from ``descendant``.

    ``git merge-base --is-ancestor`` answers with its exit code, so it cannot go
    through :func:`git`, which treats every non-zero exit as an error.
    """
    result = subprocess.run(["git", "merge-base", "--is-ancestor",
                             commit, descendant],
                            cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode == 0


def remote_branch_head(branch, remote):
    """The commit ``<remote>/<branch>`` points at *right now*, fetched fresh.

    The local ``refs/remotes/<remote>/<branch>`` can be arbitrarily stale (the
    very bug this guard exists for: a tag pushed seconds before main was
    fast-forwarded), so the remote is asked directly rather than trusting what
    the last fetch happened to record.
    """
    try:
        git("fetch", "--no-tags", remote, f"refs/heads/{branch}")
        return git("rev-parse", "FETCH_HEAD^{commit}")
    except TagError as e:
        raise TagError(f"cannot resolve {remote}/{branch} "
                       f"(is the remote/branch name right?): {e}")


def pyproject_is_dirty():
    """True when ``pyproject.toml`` has uncommitted or untracked changes."""
    out = git("status", "--porcelain", "--", PYPROJECT)
    return bool(out.strip())


def check_releasable(remote):
    """Refuse unless ``HEAD`` is the tip of ``<remote>/main`` and the tree is clean.

    Runs *after* the "tag already on the remote" early return, so a repeated run
    stays a successful no-op even if the checkout has since moved on. Every
    branch is checked so the message names the one condition that actually
    failed, rather than a generic "cannot release".
    """
    head = git("rev-parse", "HEAD")
    main_head = remote_branch_head("main", remote)

    if not is_ancestor(head, main_head):
        branch = current_branch()
        if branch == "main":
            # The other half of the same mistake: releasing from a local main
            # that has commits the remote has not got yet.
            raise TagError(
                f"HEAD ({head[:12]}) is not on {remote}/main ({main_head[:12]}), "
                f"even though you are on 'main': it has commits that have not "
                f"been pushed.\n"
                f"The release workflow only publishes tags whose commit is on "
                f"main, so this tag would fail in CI.\n"
                f"Push them first: git push {remote} main (then re-run this "
                f"script).")
        where = ("HEAD is detached" if branch == "HEAD"
                 else f"the checked-out branch is {branch!r}")
        raise TagError(
            f"HEAD ({head[:12]}) is not on {remote}/main ({main_head[:12]}), "
            f"and {where}.\n"
            f"The release workflow only publishes tags whose commit is on "
            f"main, so this tag would fail in CI -- fix it here instead.\n"
            f"Merge or push this work onto main first, e.g.:\n"
            f"  git switch main && git merge --ff-only {head[:12]}\n"
            f"then re-run this script from main.")

    if head != main_head:
        raise TagError(
            f"HEAD ({head[:12]}) is behind {remote}/main ({main_head[:12]}).\n"
            f"Tagging it would release an older commit than main, and the "
            f"version in {PYPROJECT} would describe code that is not main's "
            f"tip.\n"
            f"Run 'git pull --ff-only' and re-run this script.")

    if pyproject_is_dirty():
        raise TagError(
            f"{PYPROJECT} has uncommitted or untracked changes.\n"
            f"This script reads the version from the working tree, but a tag "
            f"describes a committed state -- releasing now would publish a "
            f"version that exists in no commit.\n"
            f"Commit (or restore) {PYPROJECT} first.")


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

    # Nothing to push is settled above, so from here the tag *will* be created:
    # it must describe main's tip and a committed version, or CI would reject it
    # (this is the check that catches tagging from a feature branch, or tagging
    # a local commit that has not reached main yet). Enforced for --dry-run too,
    # otherwise the dry run would promise a push that a real run refuses.
    check_releasable(remote)

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
