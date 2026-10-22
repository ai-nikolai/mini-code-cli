"""Tests for ``scripts/push_version_tag.py`` -- the release tag helper.

The script is driven as a *subprocess* against a real, throwaway git repository
(a bare "origin" plus a clone), because that is the thing that matters: it reads
``pyproject.toml``, talks to git, and refuses to tag anything that is not the
tip of ``<remote>/main``. Mocking git would test the mocks.

Every case builds its own pair of directories under ``tempfile`` and removes
them afterwards; nothing here touches the real repository or the network (the
"remote" is a local bare repo on disk).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "push_version_tag.py")

# A commit identity, passed per-command so the tests never depend on (or
# modify) the developer's git config.
IDENTITY = ["-c", "user.email=tests@example.invalid", "-c", "user.name=Test"]


def git(cwd, *args):
    """Run git in ``cwd`` and return stdout; raise with stderr on failure."""
    result = subprocess.run(["git", *IDENTITY, *args], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


class ReleaseScriptTest(unittest.TestCase):
    """One bare origin + one working clone, torn down per test."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="push-tag-")
        self.origin = os.path.join(self.root, "origin.git")
        self.work = os.path.join(self.root, "work")
        subprocess.run(["git", "init", "-q", "--bare", "-b", "main", self.origin],
                       check=True)
        subprocess.run(["git", "init", "-q", "-b", "main", self.work], check=True)
        self.write_pyproject("0.1.0")
        self.commit("initial")
        git(self.work, "remote", "add", "origin", self.origin)
        git(self.work, "push", "-q", "origin", "main")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    # -- helpers ---------------------------------------------------------

    def write_pyproject(self, version):
        path = os.path.join(self.work, "pyproject.toml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('[project]\nname = "demo"\nversion = "%s"\n' % version)

    def commit(self, message):
        git(self.work, "add", "-A")
        git(self.work, "commit", "-qm", message)

    def push(self):
        git(self.work, "push", "-q", "origin", "HEAD:main")

    def run_script(self, *args):
        return subprocess.run([sys.executable, SCRIPT, *args], cwd=self.work,
                              text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)

    def remote_tags(self):
        return git(self.origin, "tag", "--list").split()

    # -- happy path ------------------------------------------------------

    def test_it_creates_and_pushes_an_annotated_tag(self):
        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("created annotated tag v0.1.0", result.stdout)
        self.assertIn("pushed v0.1.0 to origin", result.stdout)
        self.assertEqual(self.remote_tags(), ["v0.1.0"])
        # Annotated, with the repo's conventional message -- not a lightweight
        # tag pointing straight at the commit.
        self.assertEqual(git(self.origin, "cat-file", "-t", "v0.1.0"), "tag")
        self.assertIn("release 0.1.0", git(self.origin, "cat-file", "-p", "v0.1.0"))

    def test_a_second_run_is_a_successful_no_op(self):
        self.assertEqual(self.run_script().returncode, 0)

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already exists on origin", result.stdout)
        self.assertIn("skipping", result.stdout)
        self.assertEqual(self.remote_tags(), ["v0.1.0"])   # still exactly one

    def test_the_skip_path_wins_even_when_the_checkout_has_moved_on(self):
        # Once *this* version's tag is out, a later checkout (local commits,
        # an unpushed branch state, a dirty tree) must not turn the release
        # script into an error: "nothing to do" is the normal outcome of a
        # repeated run, and the guard only governs new tags.
        self.assertEqual(self.run_script().returncode, 0)
        with open(os.path.join(self.work, "NOTES.md"), "w", encoding="utf-8") as f:
            f.write("local scratch, not committed\n")
        git(self.work, "switch", "-q", "-c", "scratch/branch")

        result = self.run_script()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("skipping", result.stdout)

    def test_dry_run_reports_without_creating_anything(self):
        result = self.run_script("--dry-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("would create and push v0.1.0", result.stdout)
        self.assertEqual(self.remote_tags(), [])
        self.assertEqual(git(self.work, "tag", "--list"), "")

    # -- refusals --------------------------------------------------------

    def test_an_unpushed_commit_on_main_is_refused(self):
        # The bug this guard exists for: tag a local commit that has not reached
        # the remote yet. The release workflow would reject the tag, after a CI
        # run; here it must fail immediately and create nothing.
        self.write_pyproject("0.2.0")
        self.commit("bump but do not push")

        result = self.run_script()

        self.assertEqual(result.returncode, 1)
        self.assertIn("have not been pushed", result.stderr)
        self.assertIn("git push origin main", result.stderr)
        self.assertEqual(self.remote_tags(), [])
        self.assertEqual(git(self.work, "tag", "--list"), "")

    def test_a_feature_branch_is_refused_and_named(self):
        git(self.work, "switch", "-q", "-c", "feature/x")
        self.write_pyproject("0.2.0")
        self.commit("work in progress")

        result = self.run_script()

        self.assertEqual(result.returncode, 1)
        self.assertIn("'feature/x'", result.stderr)
        self.assertIn("is not on origin/main", result.stderr)
        self.assertEqual(self.remote_tags(), [])

    def test_a_checkout_behind_main_is_refused(self):
        # Move origin ahead, then rewind the clone: HEAD is a *valid* ancestor
        # of main but not its tip, so the tag would describe older code.
        self.write_pyproject("0.2.0")
        self.commit("bump")
        self.push()
        git(self.work, "reset", "-q", "--hard", "HEAD~1")

        result = self.run_script()

        self.assertEqual(result.returncode, 1)
        self.assertIn("is behind origin/main", result.stderr)
        self.assertIn("git pull --ff-only", result.stderr)
        self.assertEqual(self.remote_tags(), [])

    def test_a_dirty_pyproject_is_refused(self):
        # The version is read from the working tree, so tagging with an
        # uncommitted version would publish something no commit contains.
        self.write_pyproject("0.9.9")

        result = self.run_script()

        self.assertEqual(result.returncode, 1)
        self.assertIn("uncommitted or untracked changes", result.stderr)
        self.assertEqual(self.remote_tags(), [])

    def test_a_missing_version_is_an_error_not_a_guess(self):
        path = os.path.join(self.work, "pyproject.toml")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write('[project]\nname = "demo"\n')
        self.commit("drop the version")

        result = self.run_script()

        self.assertEqual(result.returncode, 1)
        self.assertIn("has no version in its [project] table", result.stderr)
        self.assertEqual(self.remote_tags(), [])


class VersionAgreementTest(unittest.TestCase):
    """The script and the package must read the same version from the file.

    They are two implementations on purpose (the script has to run on a fresh
    checkout where the package is not installed, and it refuses to guess from
    installed metadata), so this is the assertion that keeps them honest.
    """

    def test_the_script_reads_the_package_version(self):
        import importlib.util

        from mini_code_cli import __version__

        spec = importlib.util.spec_from_file_location("push_version_tag", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        version = module.read_version(os.path.join(root, "pyproject.toml"))
        self.assertEqual(version, __version__)


if __name__ == "__main__":
    unittest.main()
